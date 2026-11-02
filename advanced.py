"""
Advanced parametric-insurance analytics (builds on pricing.py).

 1. Multi-site data (5 Saurashtra locations, NASA POWER)
 2. Phase-aware indices: seasonal deficit, longest dry spell, max 3-day rainfall
 3. Multi-peril contract priced with a Gaussian copula over fitted marginals
 4. Bootstrap parameter uncertainty on the premium
 5. Mann-Kendall / Sen's slope trend test
 6. Walk-forward (out-of-sample) backtest
 7. Soil-water-bucket yield proxy -> basis-risk metrics (illustrative, FAO-33 style)
 8. Spatial basis risk matrix across sites
 9. Strike/exit grid search under a premium cap
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

import pricing as pr

HERE = os.path.dirname(os.path.abspath(__file__))

SITES = {
    "Rajkot": (22.30, 70.80),
    "Junagadh": (21.52, 70.46),
    "Jamnagar": (22.47, 70.07),
    "Amreli": (21.60, 71.22),
    "Surendranagar": (22.73, 71.65),
}
INDEX_COLS = ["deficit", "dryspell", "excess"]


# ----------------------------------------------------------------------------
# 1. MULTI-SITE DATA
# ----------------------------------------------------------------------------
def synthetic_sites(names, start=pr.START_YEAR, end=pr.END_YEAR, seed=11) -> dict:
    """Correlated synthetic monsoon rainfall for several sites (offline demo only)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{start}-01-01", f"{end}-12-31")
    n = len(dates)
    in_season = np.asarray(dates.month.isin(pr.SEASON_MONTHS))
    years = np.asarray(dates.year)
    yf = {y: rng.lognormal(0, 0.28) for y in range(start, end + 1)}
    yfac = np.array([yf[y] for y in years])
    zs = np.zeros(n)
    eps = rng.standard_normal(n)
    for i in range(1, n):  # shared persistent regional weather state
        zs[i] = 0.55 * zs[i - 1] + np.sqrt(1 - 0.55**2) * eps[i]
    out = {}
    for name in names:
        sy = {y: rng.lognormal(0, 0.12) for y in range(start, end + 1)}
        syf = np.array([sy[y] for y in years])
        scale = rng.uniform(0.85, 1.15)
        lat = 0.8 * zs + 0.6 * rng.standard_normal(n)
        p_wet = np.clip(0.45 * np.sqrt(yfac * syf), 0.01, 0.9)
        wet = in_season & (lat < stats.norm.ppf(p_wet))
        amt = rng.gamma(0.8, 11 * np.sqrt(yfac * syf) * scale)
        out[name] = pd.DataFrame({"date": dates, "rain_mm": np.where(wet, amt, 0.0)})
    return out


def load_sites(refresh=False) -> tuple[dict, str]:
    """Daily rainfall for every site. Uses cache -> NASA POWER -> synthetic fallback."""
    data = {}
    try:
        for name, (lat, lon) in SITES.items():
            path = os.path.join(HERE, f"rainfall_{name}.csv")
            if name == "Rajkot" and os.path.exists(pr.CACHE) and not refresh:
                path = pr.CACHE
            if os.path.exists(path) and not refresh:
                data[name] = pd.read_csv(path, parse_dates=["date"])
            else:
                df = pr.fetch_nasa_power(lat, lon)
                df.to_csv(os.path.join(HERE, f"rainfall_{name}.csv"), index=False)
                data[name] = df
        return data, "NASA POWER"
    except Exception as e:
        print(f"[warn] site download failed ({type(e).__name__}); using SYNTHETIC data.")
        return synthetic_sites(list(SITES)), "SYNTHETIC (demo only)"


# ----------------------------------------------------------------------------
# 2. INDICES
# ----------------------------------------------------------------------------
def _longest_run(flags) -> int:
    best = cur = 0
    for v in flags:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def annual_indices(daily: pd.DataFrame, dry_threshold_mm=2.5) -> pd.DataFrame:
    """Per year: deficit (Jun-Sep total mm), dryspell (longest run <2.5 mm/day, Jul1-Sep15),
    excess (max 3-day rainfall in season, mm)."""
    d = daily.copy()
    d["rain_mm"] = d["rain_mm"].fillna(0.0)
    rows = {}
    for yr, g in d.groupby(d["date"].dt.year):
        s = g[g["date"].dt.month.isin(pr.SEASON_MONTHS)]
        w = g[(g["date"] >= f"{yr}-07-01") & (g["date"] <= f"{yr}-09-15")]
        rows[yr] = {
            "deficit": s["rain_mm"].sum(),
            "dryspell": _longest_run((w["rain_mm"].values < dry_threshold_mm)),
            "excess": s["rain_mm"].rolling(3).sum().max(),
        }
    return pd.DataFrame(rows).T.rename_axis("year")


# ----------------------------------------------------------------------------
# 3. MULTI-PERIL CONTRACT + GAUSSIAN COPULA
# ----------------------------------------------------------------------------
def _frac(x, strike, exit_):
    """Linear payout fraction; works for lower-tail (exit<strike) and upper-tail (exit>strike)."""
    return np.clip((np.asarray(x, dtype=float) - strike) / (exit_ - strike), 0, 1)


@dataclass
class MultiPerilContract:
    limit: float
    deficit: tuple  # (strike, exit) mm   -- pays when LOW
    dryspell: tuple  # (strike, exit) days -- pays when HIGH
    excess: tuple  # (strike, exit) mm   -- pays when HIGH
    weights: tuple = (0.5, 0.3, 0.2)  # share of sum insured per leg (sums to 1)

    def leg_payouts(self, d, s, e) -> np.ndarray:
        legs = [_frac(d, *self.deficit), _frac(s, *self.dryspell), _frac(e, *self.excess)]
        return np.column_stack([self.limit * w * f for w, f in zip(self.weights, legs)])

    def payout_df(self, idx: pd.DataFrame) -> pd.Series:
        tot = self.leg_payouts(idx["deficit"], idx["dryspell"], idx["excess"]).sum(axis=1)
        return pd.Series(tot, index=idx.index, name="payout")


def default_multi_contract(idx: pd.DataFrame, limit=100_000.0, weights=(0.5, 0.3, 0.2),
                           pcts=((20, 5), (80, 95), (90, 98))) -> MultiPerilContract:
    legs = []
    for col, (ps, pe) in zip(INDEX_COLS, pcts):
        strike, exit_ = (float(np.percentile(idx[col], p)) for p in (ps, pe))
        if np.isclose(strike, exit_):
            exit_ += 1.0 if exit_ >= strike else -1.0
        legs.append((strike, exit_))
    return MultiPerilContract(limit, *legs, weights=weights)


def _fit_marginal(x: np.ndarray):
    """Best of gamma / lognormal / Weibull / Gumbel by AIC."""
    best = None
    for dist, fixed in ((stats.gamma, True), (stats.lognorm, True), (stats.weibull_min, True), (stats.gumbel_r, False)):
        try:
            params = dist.fit(x, floc=0) if fixed else dist.fit(x)
            ll = np.sum(dist.logpdf(x, *params))
            if not np.isfinite(ll):
                continue
            aic = 2 * (len(params) - int(fixed)) - 2 * ll
            if best is None or aic < best[2]:
                best = (dist.name, dist(*params), aic)
        except Exception:
            continue
    return best


class CopulaModel:
    """Fitted marginals + Gaussian copula on rank-transformed indices."""

    def __init__(self, idx: pd.DataFrame):
        self.marginals = {c: _fit_marginal(idx[c].values) for c in INDEX_COLS}
        n = len(idx)
        u = np.column_stack([stats.rankdata(idx[c]) / (n + 1) for c in INDEX_COLS])
        self.corr = np.corrcoef(stats.norm.ppf(u).T)

    def simulate(self, n: int, rng) -> np.ndarray:
        z = rng.multivariate_normal(np.zeros(3), self.corr, size=n)
        u = np.clip(stats.norm.cdf(z), 1e-9, 1 - 1e-9)
        return np.column_stack([self.marginals[c][1].ppf(u[:, i]) for i, c in enumerate(INDEX_COLS)])


def price_multi(contract: MultiPerilContract, model: CopulaModel, n_sims=100_000,
                risk_load=0.25, expense_load=0.20, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    sims = model.simulate(n_sims, rng)
    legs = contract.leg_payouts(sims[:, 0], sims[:, 1], sims[:, 2])
    tot = legs.sum(axis=1)
    el, sd = tot.mean(), tot.std()
    premium = (el + risk_load * sd) / (1 - expense_load)
    q99 = np.quantile(tot, 0.99)
    return {
        "expected_loss": el, "sd_loss": sd, "prob_payout": (tot > 0).mean(),
        "var_99": q99, "tvar_99": tot[tot >= q99].mean(),
        "premium": premium, "premium_rate_pct": 100 * premium / contract.limit,
        "loss_ratio": el / premium if premium else np.nan,
        "leg_expected": dict(zip(INDEX_COLS, legs.mean(axis=0))),
        "sims_total": tot,
    }


# ----------------------------------------------------------------------------
# 4. PARAMETER UNCERTAINTY (BOOTSTRAP)
# ----------------------------------------------------------------------------
def bootstrap_premium(idx: pd.DataFrame, contract: MultiPerilContract, n_boot=150, n_sims=20_000,
                      risk_load=0.25, expense_load=0.20, seed=1) -> pd.DataFrame:
    """Resample years with replacement, refit marginals + copula, reprice. Thresholds held fixed."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        samp = idx.iloc[rng.integers(0, len(idx), len(idx))].reset_index(drop=True)
        try:
            m = CopulaModel(samp)
            r = price_multi(contract, m, n_sims, risk_load, expense_load, seed=b)
            rows.append({"premium": r["premium"], "expected_loss": r["expected_loss"], "prob_payout": r["prob_payout"]})
        except Exception:
            continue
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. TREND (NON-STATIONARITY)
# ----------------------------------------------------------------------------
def trend_test(series: pd.Series) -> dict:
    """Mann-Kendall (via Kendall tau vs time) + Theil-Sen slope."""
    tau, p = stats.kendalltau(series.index.values, series.values)
    slope = stats.theilslopes(series.values, series.index.values)[0]
    return {"tau": tau, "p_value": p, "sen_slope_per_year": slope,
            "pct_per_decade": 100 * 10 * slope / series.mean(), "significant_5pct": bool(p < 0.05)}


# ----------------------------------------------------------------------------
# 6. WALK-FORWARD BACKTEST
# ----------------------------------------------------------------------------
def backtest(index: pd.Series, min_train=15, strike_pct=20, exit_pct=5, limit=100_000.0,
             risk_load=0.25, expense_load=0.20) -> pd.DataFrame:
    """Each year: price using ONLY earlier years, then compare with that year's realised payout."""
    rows = []
    for t in index.index[min_train:]:
        train = index[index.index < t]
        c = pr.default_contract(train.values, limit, strike_pct, exit_pct)
        fit = pr.fit_distributions(train.values)[0]
        p = pr.price_contract(c, train.values, fit, n_sims=20_000, risk_load=risk_load, expense_load=expense_load)
        rows.append({"year": t, "index_mm": index[t], "premium": p["premium"],
                     "expected_loss": p["expected_loss_mc"], "payout": float(c.payout(index[t]))})
    df = pd.DataFrame(rows).set_index("year")
    df["cum_premium"] = df["premium"].cumsum()
    df["cum_payout"] = df["payout"].cumsum()
    df["cum_loss_ratio"] = df["cum_payout"] / df["cum_premium"]
    return df


# ----------------------------------------------------------------------------
# 7. YIELD PROXY + BASIS RISK
# ----------------------------------------------------------------------------
# Illustrative FAO-33 style parameters (assumptions, NOT calibrated to a crop):
PHASES = {  # name: (start MM-DD, end MM-DD, Kc, Ky)
    "establishment": ("06-15", "07-31", 0.50, 0.40),
    "flowering": ("08-01", "08-31", 1.05, 0.80),
    "maturity": ("09-01", "09-30", 0.70, 0.40),
}


def relative_yield(daily: pd.DataFrame, et0=4.5, soil_cap=100.0, waterlog_mm=150.0) -> pd.Series:
    """Daily soil-water bucket -> phase water-stress -> relative yield in [0,1].
    ky*(1-ETa/ETc) per phase, multiplied across phases, plus a waterlogging penalty."""
    d = daily.copy()
    d["rain_mm"] = d["rain_mm"].fillna(0.0)
    out = {}
    for yr, g in d.groupby(d["date"].dt.year):
        g = g[(g["date"] >= f"{yr}-06-01") & (g["date"] <= f"{yr}-09-30")]
        soil, ry = 20.0, 1.0
        eta_by, etc_by = {k: 0.0 for k in PHASES}, {k: 0.0 for k in PHASES}
        for date, r in zip(g["date"], g["rain_mm"]):
            mmdd = date.strftime("%m-%d")
            phase = next((k for k, (a, b, *_ ) in PHASES.items() if a <= mmdd <= b), None)
            kc = PHASES[phase][2] if phase else 0.2
            etc = kc * et0
            soil = min(soil_cap, soil + r)
            eta = min(etc, soil)
            soil -= eta
            if phase:
                eta_by[phase] += eta
                etc_by[phase] += etc
        for k, (_, _, _, ky) in PHASES.items():
            ry *= 1 - ky * (1 - eta_by[k] / etc_by[k])
        r3 = g["rain_mm"].rolling(3).sum().max()
        ry *= 1 - 0.3 * float(np.clip((r3 - waterlog_mm) / waterlog_mm, 0, 1))
        out[yr] = ry
    return pd.Series(out, name="rel_yield").rename_axis("year")


def farm_loss(rel_yield: pd.Series, limit: float, deductible=0.15) -> pd.Series:
    """Insurable loss: yield shortfall beyond a 15% deductible, scaled to the sum insured."""
    return limit * np.clip(1 - rel_yield / (1 - deductible), 0, 1)


def basis_metrics(payout: pd.Series, loss: pd.Series, limit: float) -> dict:
    payout, loss = payout.align(loss, join="inner")
    loss_years = loss > 0.10 * limit
    under = loss_years & (payout < 0.5 * loss)
    return {
        "corr": float(np.corrcoef(payout, loss)[0, 1]) if payout.std() > 0 and loss.std() > 0 else np.nan,
        "rmse_inr": float(np.sqrt(np.mean((payout - loss) ** 2))),
        "avg_shortfall_inr": float(np.mean(np.maximum(loss - payout, 0))),
        "avg_overpay_inr": float(np.mean(np.maximum(payout - loss, 0))),
        "loss_years": int(loss_years.sum()),
        "under_compensated_years": int(under.sum()),
    }


# ----------------------------------------------------------------------------
# 8. SPATIAL BASIS RISK
# ----------------------------------------------------------------------------
def spatial_basis(deficit_by_site: dict, strike_pct=20, exit_pct=5, drought_spi=-1.0):
    """hit[ref, farm] = P(contract at `ref` pays | `farm` has SPI drought). Also index correlation."""
    names = list(deficit_by_site)
    contracts, spi = {}, {}
    for n, s in deficit_by_site.items():
        contracts[n] = pr.default_contract(s.values, 1.0, strike_pct, exit_pct)
        spi[n] = pr.spi_like(s, pr.fit_distributions(s.values)[0])
    hit = pd.DataFrame(index=names, columns=names, dtype=float)
    for ref in names:
        paid = pd.Series(contracts[ref].payout(deficit_by_site[ref].values) > 0, index=deficit_by_site[ref].index)
        for farm in names:
            dr = spi[farm] <= drought_spi
            hit.loc[ref, farm] = paid[dr].mean() if dr.any() else np.nan
    corr = pd.DataFrame({n: deficit_by_site[n] for n in names}).corr()
    return hit, corr


# ----------------------------------------------------------------------------
# 9. CONTRACT OPTIMISER
# ----------------------------------------------------------------------------
def sensitivity_grid(index: pd.Series, loss: pd.Series, limit=100_000.0, risk_load=0.25, expense_load=0.20,
                     strike_pcts=range(10, 41, 5), exit_pcts=range(2, 21, 3)) -> pd.DataFrame:
    fit = pr.fit_distributions(index.values)[0]
    rows = []
    for sp in strike_pcts:
        for ep in exit_pcts:
            if ep >= sp:
                continue
            c = pr.default_contract(index.values, limit, sp, ep)
            p = pr.price_contract(c, index.values, fit, n_sims=20_000, risk_load=risk_load, expense_load=expense_load)
            m = basis_metrics(pd.Series(c.payout(index.values), index=index.index), loss, limit)
            rows.append({"strike_pct": sp, "exit_pct": ep, "premium_rate_pct": p["premium_rate_pct"],
                         "prob_payout": p["prob_payout"], "rmse_inr": m["rmse_inr"],
                         "shortfall_inr": m["avg_shortfall_inr"], "corr": m["corr"]})
    return pd.DataFrame(rows)


def optimise(grid: pd.DataFrame, max_premium_rate_pct: float) -> pd.Series | None:
    ok = grid[grid["premium_rate_pct"] <= max_premium_rate_pct]
    return None if ok.empty else ok.sort_values("rmse_inr").iloc[0]


# ----------------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------------
def run_advanced(site="Rajkot", refresh=False, limit=100_000.0, risk_load=0.25, expense_load=0.20,
                 n_boot=150, max_premium_rate_pct=12.0) -> dict:
    sites, source = load_sites(refresh)
    idx_by_site = {n: annual_indices(df) for n, df in sites.items()}
    idx = idx_by_site[site]
    model = CopulaModel(idx)
    mp = default_multi_contract(idx, limit)
    mp_price = price_multi(mp, model, risk_load=risk_load, expense_load=expense_load)
    boot = bootstrap_premium(idx, mp, n_boot=n_boot, risk_load=risk_load, expense_load=expense_load)

    # deficit-only comparator
    dfc = pr.default_contract(idx["deficit"].values, limit)
    dfit = pr.fit_distributions(idx["deficit"].values)[0]
    df_price = pr.price_contract(dfc, idx["deficit"].values, dfit, risk_load=risk_load, expense_load=expense_load)

    ry = relative_yield(sites[site])
    loss = farm_loss(ry, limit)
    m_df = basis_metrics(pd.Series(dfc.payout(idx["deficit"].values), index=idx.index), loss, limit)
    m_mp = basis_metrics(mp.payout_df(idx), loss, limit)

    hit, corr = spatial_basis({n: v["deficit"] for n, v in idx_by_site.items()})
    grid = sensitivity_grid(idx["deficit"], loss, limit, risk_load, expense_load)
    return dict(source=source, site=site, sites=sites, idx_by_site=idx_by_site, idx=idx, model=model,
                mp=mp, mp_price=mp_price, boot=boot, df_contract=dfc, df_price=df_price,
                rel_yield=ry, loss=loss, metrics_deficit=m_df, metrics_multi=m_mp,
                trend=trend_test(idx["deficit"]), backtest=backtest(idx["deficit"], limit=limit),
                spatial_hit=hit, spatial_corr=corr, grid=grid,
                best=optimise(grid, max_premium_rate_pct))


if __name__ == "__main__":
    r = run_advanced()
    mp, p, b = r["mp"], r["mp_price"], r["boot"]["premium"]
    print(f"Data source: {r['source']} | site: {r['site']}\n")
    print("Marginals:", {c: (m[0], round(m[2], 1)) for c, m in r["model"].marginals.items()})
    print("Copula correlation:\n", pd.DataFrame(r["model"].corr, index=INDEX_COLS, columns=INDEX_COLS).round(2))
    print(f"\nMulti-peril premium: INR {p['premium']:,.0f} ({p['premium_rate_pct']:.1f}%)  "
          f"P(payout) {p['prob_payout']:.1%}  LR {p['loss_ratio']:.0%}")
    print("Expected loss by leg:", {k: round(v) for k, v in p["leg_expected"].items()})
    print(f"Bootstrap 90% CI on premium: INR {b.quantile(.05):,.0f} - {b.quantile(.95):,.0f} (median {b.median():,.0f})")
    t = r["trend"]
    print(f"\nTrend in seasonal rain: tau={t['tau']:.2f}, p={t['p_value']:.2f}, {t['pct_per_decade']:+.1f}%/decade, significant={t['significant_5pct']}")
    bt = r["backtest"]
    print(f"Backtest {bt.index.min()}-{bt.index.max()}: cumulative loss ratio {bt['cum_loss_ratio'].iloc[-1]:.0%}")
    print("\nBasis risk vs yield proxy:")
    print(pd.DataFrame({"deficit-only": r["metrics_deficit"], "multi-peril": r["metrics_multi"]}).round(2))
    print("\nSpatial hit-rate (row = contract site, col = farm site):\n", r["spatial_hit"].round(2))
    if r["best"] is None:
        print("\nNo contract in the grid meets the premium cap; raise the cap or lower the strike percentile.")
    else:
        print("\nBest contract under premium cap:\n", r["best"])
