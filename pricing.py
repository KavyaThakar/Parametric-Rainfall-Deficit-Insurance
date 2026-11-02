"""
Parametric rainfall-deficit insurance: data -> distribution -> contract -> price -> monitor.

Location : Rajkot, Gujarat (22.30 N, 70.80 E)  -- change LAT/LON for any site
Season   : Kharif (Jun-Sep), cumulative rainfall index
Data     : NASA POWER daily precipitation (free, no API key)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

LAT, LON = 22.30, 70.80
START_YEAR, END_YEAR = 1991, 2024
SEASON_MONTHS = (6, 7, 8, 9)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rainfall_daily.csv")


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def fetch_nasa_power(lat=LAT, lon=LON, start=START_YEAR, end=END_YEAR) -> pd.DataFrame:
    """Daily corrected precipitation (mm/day) from NASA POWER."""
    import requests

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": f"{start}0101",
        "end": f"{end}1231",
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    raw = r.json()["properties"]["parameter"]["PRECTOTCORR"]
    df = pd.DataFrame({"date": pd.to_datetime(list(raw.keys())), "rain_mm": list(raw.values())})
    df.loc[df["rain_mm"] < 0, "rain_mm"] = np.nan  # -999 = missing
    return df


def synthetic_rainfall(start=START_YEAR, end=END_YEAR, seed=7) -> pd.DataFrame:
    """Offline fallback so the pipeline can be tested without internet.
    Monsoon-like: wet-day probability and gamma intensities, Jun-Sep only."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{start}-01-01", f"{end}-12-31")
    rain = np.zeros(len(dates))
    year_factor = {y: rng.lognormal(0, 0.28) for y in range(start, end + 1)}
    for i, d in enumerate(dates):
        if d.month in SEASON_MONTHS:
            p_wet = 0.45 * year_factor[d.year] ** 0.5
            if rng.random() < min(p_wet, 0.9):
                rain[i] = rng.gamma(0.8, 11 * year_factor[d.year] ** 0.5)
    return pd.DataFrame({"date": dates, "rain_mm": rain})


def load_data(refresh=False, allow_synthetic=True) -> tuple[pd.DataFrame, str]:
    if os.path.exists(CACHE) and not refresh:
        df = pd.read_csv(CACHE, parse_dates=["date"])
        return df, "cached"
    try:
        df = fetch_nasa_power()
        df.to_csv(CACHE, index=False)
        return df, "NASA POWER"
    except Exception as e:  # network blocked etc.
        if not allow_synthetic:
            raise
        print(f"[warn] NASA POWER fetch failed ({e}); using SYNTHETIC data.")
        return synthetic_rainfall(), "SYNTHETIC (demo only)"


def seasonal_index(daily: pd.DataFrame) -> pd.Series:
    """Index = cumulative Jun-Sep rainfall (mm) per year."""
    d = daily[daily["date"].dt.month.isin(SEASON_MONTHS)].copy()
    d["year"] = d["date"].dt.year
    return d.groupby("year")["rain_mm"].sum().rename("season_rain_mm")


# ----------------------------------------------------------------------------
# 2. STATISTICAL MODELLING
# ----------------------------------------------------------------------------
@dataclass
class FitResult:
    name: str
    dist: stats.rv_continuous
    params: tuple
    aic: float
    ks_stat: float
    ks_p: float

    def frozen(self):
        return self.dist(*self.params)


def fit_distributions(x: np.ndarray) -> list[FitResult]:
    """Fit candidate distributions (location fixed at 0 for rainfall), rank by AIC."""
    out = []
    for name, dist in {"gamma": stats.gamma, "lognorm": stats.lognorm, "weibull": stats.weibull_min}.items():
        params = dist.fit(x, floc=0)
        ll = np.sum(dist.logpdf(x, *params))
        k = len(params) - 1  # loc fixed
        aic = 2 * k - 2 * ll
        ks = stats.kstest(x, dist.name, args=params)
        out.append(FitResult(name, dist, params, aic, ks.statistic, ks.pvalue))
    return sorted(out, key=lambda r: r.aic)


# ----------------------------------------------------------------------------
# 3. CONTRACT + PRICING
# ----------------------------------------------------------------------------
@dataclass
class Contract:
    strike: float  # mm: payout starts when index falls below this
    exit: float  # mm: full payout at or below this
    limit: float = 100_000.0  # sum insured (INR)

    def payout(self, index):
        index = np.asarray(index, dtype=float)
        frac = np.clip((self.strike - index) / (self.strike - self.exit), 0, 1)
        return frac * self.limit


def price_contract(
    contract: Contract,
    hist_index: np.ndarray,
    fit: FitResult,
    n_sims: int = 200_000,
    risk_load: float = 0.25,  # multiple of payout std dev added to expected loss
    expense_load: float = 0.20,  # share of final premium for expenses/margin
    seed: int = 42,
) -> dict:
    """
    Burn analysis (historical) + Monte Carlo (fitted distribution).
    Technical premium = (E[loss] + risk_load * SD[loss]) / (1 - expense_load)
    """
    burn = contract.payout(hist_index)
    rng = np.random.default_rng(seed)
    sims = fit.frozen().rvs(size=n_sims, random_state=rng)
    mc = contract.payout(sims)

    expected_loss = mc.mean()
    sd_loss = mc.std()
    premium = (expected_loss + risk_load * sd_loss) / (1 - expense_load)

    return {
        "burn_cost": burn.mean(),
        "expected_loss_mc": expected_loss,
        "sd_loss": sd_loss,
        "prob_payout": (mc > 0).mean(),
        "prob_full_payout": (mc >= contract.limit - 1e-9).mean(),
        "var_99": np.quantile(mc, 0.99),
        "tvar_99": mc[mc >= np.quantile(mc, 0.99)].mean() if (mc >= np.quantile(mc, 0.99)).any() else 0.0,
        "premium": premium,
        "premium_rate_pct": 100 * premium / contract.limit,
        "loss_ratio": expected_loss / premium if premium else np.nan,
        "sims": mc,
    }


def default_contract(hist_index: np.ndarray, limit=100_000.0, strike_pct=20, exit_pct=5) -> Contract:
    return Contract(
        strike=float(np.percentile(hist_index, strike_pct)),
        exit=float(np.percentile(hist_index, exit_pct)),
        limit=limit,
    )


# ----------------------------------------------------------------------------
# 4. TRIGGER VALIDATION / BASIS-RISK PROXY
# ----------------------------------------------------------------------------
def spi_like(hist_index: pd.Series, fit: FitResult) -> pd.Series:
    """SPI: gamma CDF -> standard normal z. SPI <= -1 is a standard 'drought' flag."""
    cdf = np.clip(fit.frozen().cdf(hist_index.values), 1e-6, 1 - 1e-6)
    return pd.Series(stats.norm.ppf(cdf), index=hist_index.index, name="spi")


def trigger_validation(hist_index: pd.Series, contract: Contract, fit: FitResult, drought_spi=-1.0) -> dict:
    """
    Compare contract trigger against an independent drought benchmark (SPI <= -1).
    Without farm-yield data this is a proxy for basis risk:
      false negative = drought year, no payout   (farmer's worst case)
      false positive = payout year, no drought   (insurer's cost)
    """
    spi = spi_like(hist_index, fit)
    drought = spi <= drought_spi
    paid = contract.payout(hist_index.values) > 0
    tp = int((drought.values & paid).sum())
    fn = int((drought.values & ~paid).sum())
    fp = int((~drought.values & paid).sum())
    tn = int((~drought.values & ~paid).sum())
    return {
        "drought_years": int(drought.sum()),
        "payout_years": int(paid.sum()),
        "hit_rate": tp / (tp + fn) if (tp + fn) else np.nan,
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "table": pd.DataFrame({"index_mm": hist_index, "spi": spi, "drought": drought, "payout_inr": contract.payout(hist_index.values)}),
    }


# ----------------------------------------------------------------------------
# 5. MONITORING (automation of claims trigger check)
# ----------------------------------------------------------------------------
def monitor_season(daily: pd.DataFrame, year: int, contract: Contract, as_of: str | None = None) -> dict:
    """Season-to-date rainfall vs. the trigger. Run daily during Jun-Sep."""
    d = daily[(daily["date"].dt.year == year) & daily["date"].dt.month.isin(SEASON_MONTHS)]
    if as_of:
        d = d[d["date"] <= pd.Timestamp(as_of)]
    to_date = float(d["rain_mm"].sum())
    final = (d["date"].max().month == 9 and d["date"].max().day == 30)
    return {
        "year": year,
        "as_of": d["date"].max().date() if len(d) else None,
        "rain_to_date_mm": round(to_date, 1),
        "strike_mm": round(contract.strike, 1),
        "exit_mm": round(contract.exit, 1),
        "season_complete": final,
        "indicative_payout_inr": round(float(contract.payout(to_date)), 0) if final else None,
        "status": ("FINAL: payout due" if contract.payout(to_date) > 0 else "FINAL: no payout") if final
        else ("ON TRACK (above strike)" if to_date > contract.strike else "BELOW STRIKE so far (season still open)"),
    }


# ----------------------------------------------------------------------------
# 6. ONE-CALL PIPELINE
# ----------------------------------------------------------------------------
def run_pipeline(refresh=False, strike_pct=20, exit_pct=5, limit=100_000.0, risk_load=0.25, expense_load=0.20):
    daily, source = load_data(refresh=refresh)
    idx = seasonal_index(daily)
    fits = fit_distributions(idx.values)
    best = fits[0]
    contract = default_contract(idx.values, limit, strike_pct, exit_pct)
    pricing = price_contract(contract, idx.values, best, risk_load=risk_load, expense_load=expense_load)
    validation = trigger_validation(idx, contract, best)
    return dict(daily=daily, source=source, index=idx, fits=fits, best=best,
                contract=contract, pricing=pricing, validation=validation)


if __name__ == "__main__":
    r = run_pipeline()
    p, c, v = r["pricing"], r["contract"], r["validation"]
    print(f"Data source : {r['source']}")
    print(f"Years       : {r['index'].index.min()}-{r['index'].index.max()} (n={len(r['index'])})")
    print(f"Season rain : mean {r['index'].mean():.0f} mm, sd {r['index'].std():.0f} mm")
    print("\nDistribution fits (best AIC first):")
    for f in r["fits"]:
        print(f"  {f.name:8s} AIC={f.aic:8.1f}  KS p={f.ks_p:.3f}")
    print(f"\nContract    : strike {c.strike:.0f} mm | exit {c.exit:.0f} mm | limit INR {c.limit:,.0f}")
    print(f"Burn cost   : INR {p['burn_cost']:,.0f}   Expected loss (MC): INR {p['expected_loss_mc']:,.0f}")
    print(f"P(payout)   : {p['prob_payout']:.1%}   VaR99: INR {p['var_99']:,.0f}   TVaR99: INR {p['tvar_99']:,.0f}")
    print(f"Premium     : INR {p['premium']:,.0f} ({p['premium_rate_pct']:.2f}% of sum insured)")
    print(f"Loss ratio  : {p['loss_ratio']:.1%}")
    print(f"\nTrigger check vs SPI<=-1 drought benchmark: hit rate {v['hit_rate']:.0%}, "
          f"false-alarm rate {v['false_alarm_rate']:.0%}  (TP={v['tp']} FN={v['fn']} FP={v['fp']} TN={v['tn']})")
    print("\nMonitor (latest season, as of 31 Aug):")
    print(monitor_season(r["daily"], int(r["index"].index.max()), c, as_of=f"{int(r['index'].index.max())}-08-31"))
