import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import pricing as pr
import advanced as ad

st.set_page_config(page_title="Parametric Rainfall Insurance", layout="wide")
st.title("Parametric Rainfall-Deficit Insurance: Rajkot (Kharif)")
st.caption("Price a rainfall-index contract, test its trigger, and monitor the live season.")


@st.cache_data(show_spinner="Loading rainfall data...")
def get_data():
    daily, source = pr.load_data()
    return daily, source, pr.seasonal_index(daily)


daily, source, idx = get_data()
if "SYNTHETIC" in source:
    st.warning("NASA POWER could not be reached. Showing SYNTHETIC demo data.")
else:
    st.success(f"Data source: {source}, {idx.index.min()}-{idx.index.max()}")

fits = pr.fit_distributions(idx.values)
names = [f.name for f in fits]

# ---- Sidebar: contract terms ----
st.sidebar.header("Contract terms")
strike_pct = st.sidebar.slider("Strike percentile (payout starts)", 5, 40, 20)
exit_pct = st.sidebar.slider("Exit percentile (full payout)", 1, 20, 5)
limit = st.sidebar.number_input("Sum insured (INR)", 10_000, 10_000_000, 100_000, step=10_000)
risk_load = st.sidebar.slider("Risk load (x payout SD)", 0.0, 1.0, 0.25, 0.05)
expense_load = st.sidebar.slider("Expense load (% of premium)", 0.0, 0.5, 0.20, 0.05)
dist_choice = st.sidebar.selectbox("Distribution", names, index=0)

if exit_pct >= strike_pct:
    st.error("Exit percentile must be below strike percentile.")
    st.stop()

fit = next(f for f in fits if f.name == dist_choice)
contract = pr.default_contract(idx.values, limit, strike_pct, exit_pct)
p = pr.price_contract(contract, idx.values, fit, risk_load=risk_load, expense_load=expense_load)
v = pr.trigger_validation(idx, contract, fit)

mode = st.sidebar.radio("Mode", ["Basic (single index)", "Advanced (multi-peril)"])

if mode.startswith("Advanced"):
    st.sidebar.header("Advanced")
    site = st.sidebar.selectbox("Site", list(ad.SITES))
    n_boot = st.sidebar.slider("Bootstrap resamples", 50, 300, 100, 50)
    cap = st.sidebar.slider("Premium cap for optimiser (% of sum insured)", 4.0, 25.0, 12.0, 0.5)


    @st.cache_resource(show_spinner="Running advanced analysis...")
    def run_adv(site, limit, risk_load, expense_load, n_boot, cap):
        return ad.run_advanced(site, limit=float(limit), risk_load=risk_load, expense_load=expense_load,
                               n_boot=n_boot, max_premium_rate_pct=cap)


    R = run_adv(site, limit, risk_load, expense_load, n_boot, cap)
    if "SYNTHETIC" in R["source"]:
        st.warning("Advanced mode is using SYNTHETIC multi-site data (NASA POWER unreachable).")
    P, B, T = R["mp_price"], R["boot"]["premium"], R["trend"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Multi-peril premium", f"INR {P['premium']:,.0f}", f"{P['premium_rate_pct']:.1f}% of sum insured")
    k2.metric("90% CI (bootstrap)", f"{B.quantile(.05):,.0f} - {B.quantile(.95):,.0f}")
    k3.metric("P(any payout)", f"{P['prob_payout']:.1%}")
    k4.metric("Loss ratio", f"{P['loss_ratio']:.0%}")

    a1, a2, a3, a4, a5 = st.tabs(["Multi-peril", "Uncertainty & trend", "Backtest", "Basis risk", "Spatial & optimiser"])

    with a1:
        mp = R["mp"]
        st.write(f"**Legs** (share of sum insured: deficit {mp.weights[0]:.0%}, dry spell {mp.weights[1]:.0%}, excess {mp.weights[2]:.0%})")
        st.dataframe(pd.DataFrame({
            "trigger (strike)": [f"< {mp.deficit[0]:.0f} mm", f"> {mp.dryspell[0]:.0f} days", f"> {mp.excess[0]:.0f} mm"],
            "full payout (exit)": [f"<= {mp.deficit[1]:.0f} mm", f">= {mp.dryspell[1]:.0f} days", f">= {mp.excess[1]:.0f} mm"],
            "fitted marginal": [R["model"].marginals[c][0] for c in ad.INDEX_COLS],
            "expected loss (INR)": [round(P["leg_expected"][c]) for c in ad.INDEX_COLS],
        }, index=["Seasonal deficit", "Longest dry spell", "Max 3-day rain"]))
        corr = pd.DataFrame(R["model"].corr, index=ad.INDEX_COLS, columns=ad.INDEX_COLS)
        st.write("**Gaussian-copula dependence between perils**")
        st.plotly_chart(go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                             colorscale="RdBu", text=corr.round(2).values, texttemplate="%{text}")),
                        width="stretch")
        fig = go.Figure(go.Histogram(x=P["sims_total"][P["sims_total"] > 0], nbinsx=40))
        fig.update_layout(title="Simulated total payouts (paying years)", xaxis_title="INR")
        st.plotly_chart(fig, width="stretch")
        st.write(f"VaR 99%: INR {P['var_99']:,.0f} | TVaR 99%: INR {P['tvar_99']:,.0f}")

    with a2:
        fig = go.Figure(go.Histogram(x=B, nbinsx=30))
        fig.add_vline(x=P["premium"], line_dash="dash", annotation_text="point estimate")
        fig.update_layout(title="Premium under parameter uncertainty", xaxis_title="INR")
        st.plotly_chart(fig, width="stretch")
        st.write("With only a few decades of data, the premium is an estimate with a range, not a single number.")
        st.subheader("Trend in seasonal rainfall (Mann-Kendall / Sen's slope)")
        st.write(f"tau = {T['tau']:.2f}, p = {T['p_value']:.2f}, slope = {T['pct_per_decade']:+.1f}% per decade -> "
                 + ("**significant trend: stationary pricing is questionable.**" if T["significant_5pct"]
                    else "no significant trend at 5%."))
        ser = R["idx"]["deficit"]
        fig = go.Figure(go.Scatter(x=ser.index, y=ser.values, mode="lines+markers"))
        fig.update_layout(xaxis_title="Year", yaxis_title="Jun-Sep rainfall (mm)")
        st.plotly_chart(fig, width="stretch")

    with a3:
        bt = R["backtest"]
        st.write("Each year is priced using **only earlier data**, then compared with what actually happened.")
        fig = go.Figure()
        fig.add_scatter(x=bt.index, y=bt["cum_premium"], name="Cumulative premium")
        fig.add_scatter(x=bt.index, y=bt["cum_payout"], name="Cumulative payout")
        fig.update_layout(yaxis_title="INR")
        st.plotly_chart(fig, width="stretch")
        st.metric("Out-of-sample cumulative loss ratio", f"{bt['cum_loss_ratio'].iloc[-1]:.0%}")
        st.dataframe(bt.round(0), width="stretch")

    with a4:
        st.write("Payouts vs. a **soil-water-bucket yield-loss proxy** (illustrative FAO-33 style model, not calibrated to a real crop).")
        ry = R["rel_yield"]
        fig = go.Figure()
        fig.add_scatter(x=R["loss"].index, y=R["loss"].values, name="Modelled farm loss", line=dict(width=3))
        fig.add_scatter(x=R["idx"].index, y=R["df_contract"].payout(R["idx"]["deficit"].values), name="Deficit-only payout")
        fig.add_scatter(x=R["idx"].index, y=R["mp"].payout_df(R["idx"]).values, name="Multi-peril payout")
        fig.update_layout(yaxis_title="INR", xaxis_title="Year")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame({"Deficit-only": R["metrics_deficit"], "Multi-peril": R["metrics_multi"]}).round(2))
        st.caption("Lower RMSE and fewer under-compensated years mean lower basis risk. Calibrate with real district yields before relying on this.")

    with a5:
        st.write("**Spatial basis risk:** P(contract at row-site pays | farm at column-site has a drought year)")
        H = R["spatial_hit"]
        st.plotly_chart(go.Figure(go.Heatmap(z=H.values, x=H.columns, y=H.index, zmin=0, zmax=1, colorscale="Viridis",
                                             text=H.round(2).values, texttemplate="%{text}")), width="stretch")
        st.write("**Correlation of seasonal rainfall between sites**")
        C = R["spatial_corr"]
        st.plotly_chart(go.Figure(go.Heatmap(z=C.values, x=C.columns, y=C.index, zmin=-1, zmax=1, colorscale="RdBu",
                                             text=C.round(2).values, texttemplate="%{text}")), width="stretch")
        st.subheader("Contract optimiser (deficit contract)")
        g = R["grid"]
        piv = g.pivot(index="exit_pct", columns="strike_pct", values="rmse_inr")
        st.plotly_chart(go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale="Viridis_r",
                                             colorbar=dict(title="RMSE"))).update_layout(
            xaxis_title="Strike percentile", yaxis_title="Exit percentile", title="Basis-risk RMSE (INR)"),
            width="stretch")
        if R["best"] is None:
            st.info("No strike/exit pair meets the premium cap. Raise the cap in the sidebar.")
        else:
            b = R["best"]
            st.success(f"Lowest basis risk within cap: strike P{int(b.strike_pct)}, exit P{int(b.exit_pct)} -> "
                       f"premium {b.premium_rate_pct:.1f}%, RMSE INR {b.rmse_inr:,.0f}.")
        st.caption("Optimised in-sample on a short record, so treat it as a starting point and validate out-of-sample.")
    st.stop()

# ---- KPIs ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Premium", f"INR {p['premium']:,.0f}", f"{p['premium_rate_pct']:.1f}% of sum insured")
c2.metric("Expected loss", f"INR {p['expected_loss_mc']:,.0f}")
c3.metric("P(payout)", f"{p['prob_payout']:.1%}")
c4.metric("Loss ratio", f"{p['loss_ratio']:.0%}")

tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Payout & risk", "Trigger validation", "Season monitor"])

with tab1:
    x = np.linspace(0, idx.max() * 1.4, 400)
    fig = go.Figure()
    fig.add_histogram(x=idx.values, histnorm="probability density", name="Observed", opacity=0.5, nbinsx=12)
    for f in fits:
        fig.add_scatter(x=x, y=f.frozen().pdf(x), name=f"{f.name} (AIC {f.aic:.1f})",
                        line=dict(width=4 if f.name == dist_choice else 1.5))
    fig.add_vline(x=contract.strike, line_dash="dash", annotation_text="strike")
    fig.add_vline(x=contract.exit, line_dash="dash", annotation_text="exit")
    fig.update_layout(xaxis_title="Seasonal rainfall Jun-Sep (mm)", yaxis_title="Density")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame([{"dist": f.name, "AIC": round(f.aic, 1), "KS p-value": round(f.ks_p, 3)} for f in fits]),
                 hide_index=True)

with tab2:
    xs = np.linspace(0, idx.max() * 1.2, 300)
    fig = go.Figure(go.Scatter(x=xs, y=contract.payout(xs), name="Payout"))
    fig.update_layout(xaxis_title="Seasonal rainfall (mm)", yaxis_title="Payout (INR)", title="Payout function")
    st.plotly_chart(fig, width="stretch")
    fig2 = go.Figure(go.Histogram(x=p["sims"][p["sims"] > 0], nbinsx=40))
    fig2.update_layout(title="Simulated payouts (paying years only)", xaxis_title="INR")
    st.plotly_chart(fig2, width="stretch")
    st.write(f"**VaR 99%:** INR {p['var_99']:,.0f} | **TVaR 99%:** INR {p['tvar_99']:,.0f} | "
             f"**Burn cost (historical):** INR {p['burn_cost']:,.0f}")

with tab3:
    st.write(f"Benchmark: SPI <= -1 counts as a drought year. **Hit rate {v['hit_rate']:.0%}**, "
             f"**false-alarm rate {v['false_alarm_rate']:.0%}**.")
    st.write(f"Droughts missed (basis risk for farmers): **{v['fn']}** | Payouts in non-drought years: **{v['fp']}**")
    t = v["table"].round(2)
    st.dataframe(t, width="stretch")

with tab4:
    year = st.selectbox("Season", sorted(idx.index, reverse=True))
    as_of = st.date_input("As of", pd.Timestamp(f"{year}-08-31"))
    st.json({k: (str(val) if val is not None else None) for k, val in
             pr.monitor_season(daily, year, contract, as_of=str(as_of)).items()})
