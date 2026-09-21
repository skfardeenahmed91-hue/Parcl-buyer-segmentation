"""
Parcl Buyer Intelligence Dashboard
==================================
Interactive companion to the research paper "Machine Learning based Buyer Segmentation and
Investment Profiling for Real Estate Market Intelligence".

Run:  streamlit run app.py
Data: data/processed/*  (created by  python src/pipeline.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

st.set_page_config(page_title="Parcl Buyer Intelligence", page_icon=":office:", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {background:#F3F6F9; border:1px solid #E1E7ED; padding:12px 14px; border-radius:10px;}
      div[data-testid="stMetricLabel"] p {font-size:0.80rem; color:#5B6B7A;}
      h1, h2, h3 {letter-spacing:-0.01em;}
      .seg-card {border-radius:10px; border:1px solid #E1E7ED; padding:14px 16px; height:100%;}
      .seg-card .tag {font-size:0.74rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;}
      .seg-card .name {font-size:1.05rem; font-weight:700; margin:2px 0 6px 0; color:#1B2A38;}
      .seg-card .big {font-size:1.6rem; font-weight:700; color:#1B2A38; line-height:1.1;}
      .seg-card .sub {font-size:0.82rem; color:#5B6B7A;}
      .small-note {font-size:0.82rem; color:#5B6B7A;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def show_chart(fig, key=None):
    """st.plotly_chart across Streamlit versions (the width argument changed in 1.4x-1.5x)."""
    fig.update_layout(font=dict(family="Arial, sans-serif", size=12), margin=dict(l=10, r=10, t=50, b=10), title_font=dict(size=15))
    try:
        st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def show_table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def money(x):
    return f"${x:,.0f}"


def money_m(x):
    return f"${x / 1e6:,.2f}M"


def pct(x, d=1):
    return f"{x:.{d}%}"


@st.cache_data(show_spinner=False)
def load_data():
    clients = pd.read_csv(PROC / "clients_segmented.csv", parse_dates=["first_purchase", "last_purchase"])
    tx = pd.read_csv(PROC / "transactions.csv", parse_dates=["month"])
    meta = json.load(open(PROC / "segments.json"))
    prof = pd.read_csv(PROC / "segment_profiles.csv", index_col=0)
    ms = pd.read_csv(PROC / "model_selection.csv")
    hyp = pd.read_csv(PROC / "hypothesis_tests.csv")
    res = json.load(open(PROC / "results.json"))
    imp = pd.read_csv(PROC / "feature_importance.csv")
    return clients, tx, meta, prof, ms, hyp, res, imp


if not (PROC / "clients_segmented.csv").exists():
    st.error("Processed data not found. Run `python src/pipeline.py` from the project folder first.")
    st.stop()

clients, tx, meta, prof, ms, hyp, res, imp = load_data()
SEG_IDS = list(meta.keys())
LABELS = [meta[s]["label"] for s in SEG_IDS]
COLOR = {meta[s]["label"]: meta[s]["color"] for s in SEG_IDS}
NAME_OF = {meta[s]["label"]: meta[s]["name"] for s in SEG_IDS}

# Segment playbook: wording is tied to the four archetypes the pipeline can produce
PLAYBOOK = {
    "Domestic Cash Home Buyers": dict(
        who="US-resident buyers purchasing for personal use without a loan.",
        actions=["Sell on speed and certainty: quick closings, transparent pricing and move-in-ready inventory.",
                 "Use them as the seed group for a referral programme. Only about one client in ten arrives by referral today, and satisfaction sits near 3 out of 5, so pair any incentive with a service fix.",
                 "Offer multi-unit bundles across towers, and office units, which make up about 15% of purchases today. These buyers have no lender to satisfy, so they can decide quickly."]),
    "Domestic Investors": dict(
        who="US-resident buyers purchasing as an investment. Financing is mixed: about 38% use a loan.",
        actions=["Lead with investment content: rental yield, portfolio bundles across towers, availability alerts for multi-unit blocks.",
                 "Run two message tracks: leveraged investors (lender co-marketing) and cash investors (bulk or portfolio pricing).",
                 "Keep a named account list of portfolio buyers (5+ units). They are about 6% of clients but hold roughly 9% of revenue, so each is worth about one and a half average clients."]),
    "Domestic Financed Home Buyers": dict(
        who="US-resident buyers purchasing for personal use with a loan.",
        actions=["Put mortgage support at first touch: lender partnerships, pre-approval, payment-based price calculators.",
                 "Treat the financing experience as the service priority. Satisfaction is about 3 out of 5 in every segment (no significant difference between them), so the aim is to lift it for this group rather than to fix a known gap.",
                 "Time-to-close and loan-approval turnaround are the operational KPIs to monitor for this group."]),
    "International Buyers": dict(
        who="Clients resident outside the United States, with mixed purposes and financing.",
        actions=["Build a cross-border service kit: multi-currency pricing, remote or proxy closing, local-language material.",
                 "Line up cross-border financing partners; about 37% of these clients apply for a loan.",
                 "Do not lead with portfolio offers: only 2% of international clients buy 5+ units, against 7% of domestic clients. This is also the most heterogeneous segment, so message by country rather than as one block."]),
}
DEFAULT_PLAY = dict(who="Segment defined by the clustering model.", actions=["Review the segment profile before choosing actions."])


# ------------------------------------------------------------------------------------------
# sidebar filters
# ------------------------------------------------------------------------------------------
st.sidebar.markdown("## Parcl Buyer Intelligence")
st.sidebar.caption("Machine-learning buyer segmentation and investment profiling")

ALL_COUNTRIES = sorted(clients.country.unique(), key=lambda c: -int((clients.country == c).sum()))
DEFAULTS = {"f_country": [], "f_region": [], "f_purpose": [], "f_type": [], "f_segment": []}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = list(v)


st.sidebar.button("Reset all filters", on_click=reset_filters)
st.sidebar.markdown("**Filters** (leave empty to include everything)")
sel_country = st.sidebar.multiselect("Country", ALL_COUNTRIES, key="f_country")
region_pool = clients[clients.country.isin(sel_country)] if sel_country else clients
region_opts = sorted(region_pool.region.unique())
st.session_state["f_region"] = [r for r in st.session_state["f_region"] if r in region_opts]     # drop stale picks
sel_region = st.sidebar.multiselect("Region / state", region_opts, key="f_region")
sel_purpose = st.sidebar.multiselect("Acquisition purpose", sorted(clients.acquisition_purpose.unique()), key="f_purpose")
sel_type = st.sidebar.multiselect("Client type", sorted(clients.client_type.unique()), key="f_type")
with st.sidebar.expander("More filters"):
    sel_segment = st.multiselect("Segment", LABELS, key="f_segment")

mask = pd.Series(True, index=clients.index)
if sel_country: mask &= clients.country.isin(sel_country)
if sel_region: mask &= clients.region.isin(sel_region)
if sel_purpose: mask &= clients.acquisition_purpose.isin(sel_purpose)
if sel_type: mask &= clients.client_type.isin(sel_type)
if sel_segment: mask &= clients.segment_label.isin(sel_segment)
fdf = clients[mask].copy()
ftx = tx[tx.client_id.isin(fdf.client_id) & tx.is_sold].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Clients in view", f"{len(fdf):,} of {len(clients):,}")
st.sidebar.caption(f"Data: {res['log']['date_min'][:7]} to {res['log']['date_max'][:7]} | {res['sold_units']:,} units sold | "
                   f"{res['n_clients']:,} clients")

# ------------------------------------------------------------------------------------------
# header + KPIs
# ------------------------------------------------------------------------------------------
st.title("Buyer Segmentation and Investment Profiling")
st.caption("Parcl Co. Limited x Unified Mentor. Segments come from K-Means clustering (validated with Ward hierarchical clustering) on client attributes and purchase behaviour.")

if fdf.empty:
    st.warning("No clients match the current filters. Widen the selection or press **Reset all filters**.")
    st.stop()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Clients", f"{len(fdf):,}")
k2.metric("Revenue (sold units)", money_m(fdf.total_spend.sum()))
k3.metric("Avg spend per client", money_m(fdf.total_spend.mean()))
k4.metric("Investment purpose", pct(fdf.investment_flag.mean()))
k5.metric("Loan applied", pct(fdf.loan_flag.mean()))
k6.metric("Avg satisfaction", f"{fdf.satisfaction_score.mean():.2f} / 5")

tab_over, tab_beh, tab_geo, tab_ins, tab_model, tab_data = st.tabs(
    ["Segmentation Overview", "Investor Behavior", "Geographic Analysis", "Segment Insights", "Model & Methodology", "Data Explorer"])

# ------------------------------------------------------------------------------------------
# 1. SEGMENTATION OVERVIEW
# ------------------------------------------------------------------------------------------
with tab_over:
    seg = (fdf.groupby("segment_label").agg(clients=("client_id", "count"), revenue=("total_spend", "sum"))
           .reindex(LABELS).fillna(0))
    seg["client_share"] = seg.clients / seg.clients.sum()
    seg["revenue_share"] = seg.revenue / seg.revenue.sum() if seg.revenue.sum() else 0

    cols = st.columns(len(SEG_IDS))
    for c, s in zip(cols, SEG_IDS):
        lab = meta[s]["label"]
        c.markdown(
            f"""<div class="seg-card" style="border-top:5px solid {meta[s]['color']}">
                  <div class="tag" style="color:{meta[s]['color']}">{s}</div>
                  <div class="name">{meta[s]['name']}</div>
                  <div class="big">{int(seg.loc[lab, 'clients']):,}</div>
                  <div class="sub">clients | {pct(seg.loc[lab, 'client_share'])} of view</div>
                  <div class="sub" style="margin-top:6px">{pct(seg.loc[lab, 'revenue_share'])} of revenue</div>
                </div>""", unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns([1, 1.15])
    with c1:
        d = seg.reset_index()
        d = d[d.clients > 0]
        fig = px.pie(d, names="segment_label", values="clients", hole=0.58, color="segment_label", color_discrete_map=COLOR,
                     title="Cluster distribution (share of clients)")
        fig.update_traces(textinfo="percent", textfont_size=13, hovertemplate="%{label}<br>%{value:,} clients (%{percent})<extra></extra>")
        fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", font=dict(size=11)))
        show_chart(fig, key="ov_donut")
    with c2:
        d = seg.reset_index().melt(id_vars="segment_label", value_vars=["client_share", "revenue_share"], var_name="measure", value_name="share")
        d["measure"] = d.measure.map({"client_share": "Share of clients", "revenue_share": "Share of revenue"})
        d["segment"] = d.segment_label.str.split(" - ").str[0]
        fig = px.bar(d, x="segment", y="share", color="measure", barmode="group", title="Clients vs revenue by segment",
                     color_discrete_sequence=["#1F4E79", "#E07B39"], text=d.share.map(lambda v: f"{v:.0%}"))
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None, legend_title=None, legend=dict(orientation="h", y=-0.15))
        fig.update_traces(textposition="outside")
        show_chart(fig, key="ov_bar")

    st.markdown("##### Segments in two dimensions")
    rng = np.random.default_rng(1)
    pts = fdf.copy()
    pts["pc1j"] = pts.pc1 + rng.normal(0, 0.012, len(pts))
    pts["pc2j"] = pts.pc2 + rng.normal(0, 0.012, len(pts))
    fig = px.scatter(pts, x="pc1j", y="pc2j", color="segment_label", color_discrete_map=COLOR, opacity=0.6,
                     hover_data={"client_id": True, "country": True, "region": True, "acquisition_purpose": True, "loan_applied": True,
                                 "pc1j": False, "pc2j": False, "segment_label": False},
                     labels={"pc1j": "Principal component 1", "pc2j": "Principal component 2"}, title=None)
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(legend_title=None, height=430, legend=dict(orientation="h", y=-0.2))
    show_chart(fig, key="ov_pca")
    st.markdown(
        f"<p class='small-note'>The two components capture only {res['pca_var'][0]:.0%} and {res['pca_var'][1]:.0%} of the variance. Each small island is one "
        f"combination of categorical attributes (region, referral channel, purpose, financing). Segments are unions of islands and are "
        f"<b>not cleanly separated blobs</b>: the mean silhouette is {res['silhouette']:.2f}, which indicates weak cluster structure.</p>",
        unsafe_allow_html=True)
    st.info(
        f"**How to read these segments.** A depth-3 decision tree reproduces the segment of every client with {res['tree_acc']:.0%} accuracy using only three questions: "
        "*Does the client live in the US? Is the purchase an investment? Was a loan applied for?* Age, satisfaction, units bought and spend differ by only a few percent across segments (all effect sizes below 0.005). "
        "The segments describe **how clients buy** (geography, intent, financing), not **how much they buy**.")

# ------------------------------------------------------------------------------------------
# 2. INVESTOR BEHAVIOR
# ------------------------------------------------------------------------------------------
with tab_beh:
    st.subheader("Investment patterns by segment")
    c1, c2 = st.columns(2)
    with c1:
        ct = pd.crosstab(fdf.segment_label, fdf.acquisition_purpose, normalize="index").reindex(LABELS).fillna(0).reset_index()
        d = ct.melt(id_vars="segment_label", var_name="Purpose", value_name="share")
        d["segment"] = d.segment_label.str.split(" - ").str[0]
        fig = px.bar(d, x="segment", y="share", color="Purpose", title="Acquisition purpose", color_discrete_map={"Investment": "#1F4E79", "Personal Use": "#C9D1D9"},
                     text=d.share.map(lambda v: f"{v:.0%}" if v > 0.04 else ""))
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None, legend_title=None, legend=dict(orientation="h", y=-0.15))
        show_chart(fig, key="bh_purpose")
    with c2:
        ct = pd.crosstab(fdf.segment_label, fdf.loan_applied, normalize="index").reindex(LABELS).fillna(0).reset_index()
        d = ct.melt(id_vars="segment_label", var_name="Loan applied", value_name="share")
        d["segment"] = d.segment_label.str.split(" - ").str[0]
        fig = px.bar(d, x="segment", y="share", color="Loan applied", title="Financing (loan applied)", color_discrete_map={"Yes": "#E07B39", "No": "#C9D1D9"},
                     text=d.share.map(lambda v: f"{v:.0%}" if v > 0.04 else ""))
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None, legend_title=None, legend=dict(orientation="h", y=-0.15))
        show_chart(fig, key="bh_loan")

    st.markdown("##### Compare segments on a purchase metric")
    metrics = {
        "Total spend per client (USD)": ("total_spend", "mean", money), "Units bought per client": ("n_units", "mean", lambda v: f"{v:.2f}"),
        "Average unit price (USD)": ("avg_price", "mean", money), "Average price per sq ft (USD)": ("avg_ppsf", "mean", lambda v: f"${v:,.0f}"),
        "Office share of purchases": ("share_office", "mean", lambda v: f"{v:.1%}"), "Share buying 5+ units": ("portfolio_buyer", "mean", lambda v: f"{v:.1%}"),
        "Satisfaction (1-5)": ("satisfaction_score", "mean", lambda v: f"{v:.2f}"), "Client age (years)": ("age", "mean", lambda v: f"{v:.1f}")}
    mc1, mc2 = st.columns([1, 2.2])
    with mc1:
        m_label = st.selectbox("Metric", list(metrics), key="beh_metric")
        show_box = st.checkbox("Show distribution (box plot)", value=False, key="beh_box")
        st.markdown("<p class='small-note'>Differences between segments on these metrics are small. The bar labels make that visible; treat gaps of a few percent as noise unless the sample is large.</p>",
                    unsafe_allow_html=True)
    with mc2:
        col, agg, fmt = metrics[m_label]
        if show_box:
            fig = px.box(fdf, x="segment_id", y=col, color="segment_label", color_discrete_map=COLOR, points=False, category_orders={"segment_id": SEG_IDS})
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=m_label, title=f"{m_label}: distribution")
        else:
            g = fdf.groupby("segment_label")[col].agg(["mean", "count"]).reindex(LABELS).dropna().reset_index()
            g["segment"] = g.segment_label.str.split(" - ").str[0]
            fig = px.bar(g, x="segment", y="mean", color="segment_label", color_discrete_map=COLOR, text=g["mean"].map(fmt),
                         hover_data={"count": True, "segment": False, "segment_label": False}, title=f"{m_label}: segment average")
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
            fig.update_traces(textposition="outside")
        show_chart(fig, key="bh_metric")

    c1, c2 = st.columns(2)
    with c1:
        if len(ftx):
            ut = pd.crosstab(ftx.segment_label, ftx.unit_category, normalize="index").reindex(LABELS).fillna(0).reset_index()
            d = ut.melt(id_vars="segment_label", var_name="Unit type", value_name="share")
            d["segment"] = d.segment_label.str.split(" - ").str[0]
            fig = px.bar(d, x="segment", y="share", color="Unit type", title="Unit type purchased", color_discrete_map={"Apartment": "#2A9D8F", "Office": "#8E5EA2"},
                         text=d.share.map(lambda v: f"{v:.0%}"))
            fig.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None, legend_title=None, legend=dict(orientation="h", y=-0.15))
            show_chart(fig, key="bh_unit")
    with c2:
        rc = pd.crosstab(fdf.segment_label, fdf.referral_channel, normalize="index").reindex(LABELS).fillna(0).reset_index()
        d = rc.melt(id_vars="segment_label", var_name="Channel", value_name="share")
        d["segment"] = d.segment_label.str.split(" - ").str[0]
        fig = px.bar(d, x="segment", y="share", color="Channel", title="How clients found Parcl (referral channel)",
                     color_discrete_map={"Website": "#1F4E79", "Agency": "#E07B39", "Client": "#2A9D8F"}, text=d.share.map(lambda v: f"{v:.0%}"))
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None, legend_title=None, legend=dict(orientation="h", y=-0.15))
        show_chart(fig, key="bh_ref")

    st.markdown("##### Units sold per month by segment")
    if len(ftx):
        mm = ftx.groupby(["month", "segment_label"]).size().reset_index(name="units")
        fig = px.line(mm, x="month", y="units", color="segment_label", color_discrete_map=COLOR, markers=True)
        fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title="Units sold", legend=dict(orientation="h", y=-0.2), height=380)
        show_chart(fig, key="bh_month")
        st.markdown("<p class='small-note'>Sales volume fell by roughly a third from 2024 to 2025 in every segment at once, so the decline is a market-wide effect and not a shift in buyer mix.</p>",
                    unsafe_allow_html=True)

    st.markdown("##### Investors vs personal-use buyers")
    cmp = fdf.groupby("acquisition_purpose").agg(Clients=("client_id", "count"), **{"Avg units": ("n_units", "mean"), "Avg total spend": ("total_spend", "mean"),
          "Avg unit price": ("avg_price", "mean"), "Loan share": ("loan_flag", "mean"), "Corporate share": ("client_type", lambda x: (x == "Corporate").mean()),
          "Avg satisfaction": ("satisfaction_score", "mean"), "Avg age": ("age", "mean")}).round(3)
    cmp["Avg total spend"] = cmp["Avg total spend"].map(money); cmp["Avg unit price"] = cmp["Avg unit price"].map(money)
    cmp["Loan share"] = cmp["Loan share"].map(pct); cmp["Corporate share"] = cmp["Corporate share"].map(pct)
    show_table(cmp)

# ------------------------------------------------------------------------------------------
# 3. GEOGRAPHIC ANALYSIS
# ------------------------------------------------------------------------------------------
with tab_geo:
    st.subheader("Where each segment lives")
    gc1, gc2, gc3, gc4 = st.columns([1, 1.3, 1.3, 1.1])
    scope = gc1.selectbox("Map area", ["world", "usa", "europe", "north america"], format_func=str.title, key="geo_scope")
    mode = gc2.selectbox("Colour regions by", ["Dominant segment", "Share of a chosen segment", "Investment share", "Loan share", "Average spend per client", "Average satisfaction"], key="geo_mode")
    seg_pick = gc3.selectbox("Segment (for share view)", LABELS, key="geo_seg", disabled=(mode != "Share of a chosen segment"))
    min_n = gc4.slider("Min. clients per region", 1, 40, 5, key="geo_min")

    g = fdf.groupby(["country", "region", "lat", "lon"]).agg(clients=("client_id", "count"), inv=("investment_flag", "mean"), loan=("loan_flag", "mean"),
                                                             spend=("total_spend", "mean"), sat=("satisfaction_score", "mean")).reset_index()
    seg_mix = pd.crosstab(fdf.region, fdf.segment_label, normalize="index").reindex(columns=LABELS).fillna(0)
    g = g.merge(seg_mix, left_on="region", right_index=True, how="left")
    g["dominant"] = g[LABELS].idxmax(axis=1)
    g = g[g.clients >= min_n]
    if g.empty:
        st.info("No region has that many clients under the current filters. Lower the minimum.")
    else:
        hover = {"clients": True, "lat": False, "lon": False, "country": True}
        common = dict(lat="lat", lon="lon", size="clients", size_max=42, hover_name="region", scope=scope, projection="natural earth" if scope in ("world",) else None)
        common = {k: v for k, v in common.items() if v is not None}
        if mode == "Dominant segment":
            fig = px.scatter_geo(g, color="dominant", color_discrete_map=COLOR, hover_data=hover, **common)
            fig.update_layout(legend_title="Dominant segment")
        else:
            if mode == "Share of a chosen segment":
                g["value"] = g[seg_pick]; ttl, fmt, scale = f"Share of {seg_pick}", ".0%", "Blues"
            elif mode == "Investment share":
                g["value"], ttl, fmt, scale = g.inv, "Investment share", ".0%", "Blues"
            elif mode == "Loan share":
                g["value"], ttl, fmt, scale = g.loan, "Loan share", ".0%", "Oranges"
            elif mode == "Average spend per client":
                g["value"], ttl, fmt, scale = g.spend, "Avg spend (USD)", ",.0f", "Teal"
            else:
                g["value"], ttl, fmt, scale = g.sat, "Avg satisfaction", ".2f", "Purples"
            fig = px.scatter_geo(g, color="value", color_continuous_scale=scale, hover_data=hover, **common)
            fig.update_layout(coloraxis_colorbar=dict(title=ttl, tickformat=fmt))
        fig.update_geos(showcountries=True, countrycolor="#B8C2CC", showland=True, landcolor="#F3F6F9", showocean=True, oceancolor="#EAF2F8", showsubunits=True, subunitcolor="#D5DCE3")
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig, key="geo_map")
        st.caption("Bubble size = number of clients in the region. Map tiles load from the Plotly CDN, so an internet connection is needed for the base map. "
                   "Shares from regions with few clients are noisy; raise the minimum-clients slider to suppress them.")

    c1, c2 = st.columns([1.15, 1])
    with c1:
        topn = st.slider("Regions shown in the heatmap (largest by clients)", 5, 30, 15, key="geo_topn")
        top_regions = fdf.region.value_counts().head(topn).index
        hm = pd.crosstab(fdf.region, fdf.segment_label, normalize="index").reindex(columns=LABELS).fillna(0).loc[top_regions]
        hm.columns = [c.split(" - ")[0] for c in hm.columns]
        fig = px.imshow(hm.values, x=list(hm.columns), y=list(hm.index), color_continuous_scale="Blues", zmin=0, zmax=1, aspect="auto",
                        text_auto=".0%", title="Segment mix within each region")
        fig.update_layout(coloraxis_showscale=False, height=max(320, 26 * topn + 90), xaxis_title=None, yaxis_title=None)
        fig.update_yaxes(autorange="reversed")
        show_chart(fig, key="geo_heat")
    with c2:
        ctry = fdf.groupby("country").agg(Clients=("client_id", "count"), Investment=("investment_flag", "mean"), Loan=("loan_flag", "mean"),
                                          AvgSpend=("total_spend", "mean"), Satisfaction=("satisfaction_score", "mean")).sort_values("Clients", ascending=False)
        ctry["Investment"] = ctry.Investment.map(pct, na_action="ignore"); ctry["Loan"] = ctry.Loan.map(pct, na_action="ignore")
        ctry["AvgSpend"] = ctry.AvgSpend.map(money); ctry["Satisfaction"] = ctry.Satisfaction.round(2)
        st.markdown("**Country summary**")
        show_table(ctry, height=min(420, 36 * (len(ctry) + 1) + 4))
        st.markdown("<p class='small-note'>Differences between countries in investment and loan share are not statistically significant (chi-square p = 0.56 and 0.43), and most country samples are small. Treat single-country gaps as noise (see the confidence intervals in the paper, Figure 10).</p>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------
# 4. SEGMENT INSIGHTS
# ------------------------------------------------------------------------------------------
with tab_ins:
    st.subheader("Segment insights panel")
    choice = st.selectbox("Segment", ["All clients in view"] + LABELS, key="ins_seg")
    sub = fdf if choice == "All clients in view" else fdf[fdf.segment_label == choice]
    if sub.empty:
        st.info("No clients from this segment under the current filters.")
    else:
        if choice != "All clients in view":
            nm = NAME_OF[choice]; play = PLAYBOOK.get(nm, DEFAULT_PLAY)
            sid = choice.split(" - ")[0]
            coh = float(prof.loc[sid, "mean_silhouette"])
            st.markdown(f"<div style='border-left:6px solid {COLOR[choice]}; padding:6px 14px; background:#F8FAFC; border-radius:4px'>"
                        f"<b>{choice}</b><br>{play['who']}<br><span class='small-note'>Cluster cohesion (mean silhouette): {coh:.2f} "
                        f"({'relatively cohesive for this model' if coh >= 0.2 else 'loosely defined'}; all four segments sit below the 0.25 structure threshold) | agreement with Ward clustering: {prof.loc[sid, 'ward_agreement']:.0%}</span></div>", unsafe_allow_html=True)
            st.write("")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Clients", f"{len(sub):,}"); m2.metric("Median spend", money_m(sub.total_spend.median()))
        m3.metric("Avg units", f"{sub.n_units.mean():.2f}"); m4.metric("Avg unit price", money(sub.avg_price.mean())); m5.metric("Avg age", f"{sub.age.mean():.1f}")

        st.markdown("##### Descriptive statistics")
        num = {"Age (years)": "age", "Satisfaction (1-5)": "satisfaction_score", "Units bought": "n_units", "Total spend (USD)": "total_spend",
               "Avg unit price (USD)": "avg_price", "Price per sq ft (USD)": "avg_ppsf", "Office share": "share_office", "Active months": "active_months"}
        desc = sub[list(num.values())].describe().T
        desc.index = list(num.keys())
        desc = desc.rename(columns={"count": "n", "50%": "median", "25%": "p25", "75%": "p75"})[["n", "mean", "std", "min", "p25", "median", "p75", "max"]]
        show_table(desc.round(2))

        c1, c2 = st.columns(2)
        with c1:
            comp_rows = []
            rest = fdf[fdf.segment_label != choice] if choice != "All clients in view" else fdf
            for nm_, col_ in [("Investment purpose", "investment_flag"), ("Loan applied", "loan_flag"), ("US resident", "is_usa"), ("Corporate client", None), ("Buys 5+ units", "portfolio_buyer")]:
                a = sub[col_].mean() if col_ else (sub.client_type == "Corporate").mean()
                b = rest[col_].mean() if (col_ and len(rest)) else ((rest.client_type == "Corporate").mean() if len(rest) else np.nan)
                comp_rows.append(dict(Attribute=nm_, Segment=a, Others=b))
            comp = pd.DataFrame(comp_rows)
            fig = go.Figure()
            fig.add_bar(y=comp.Attribute, x=comp.Segment, orientation="h", name="This segment" if choice != "All clients in view" else "All in view", marker_color=COLOR.get(choice, "#1F4E79"),
                        text=comp.Segment.map(lambda v: f"{v:.0%}"), textposition="outside")
            if choice != "All clients in view" and len(rest):
                fig.add_bar(y=comp.Attribute, x=comp.Others, orientation="h", name="All other clients", marker_color="#C9D1D9", text=comp.Others.map(lambda v: f"{v:.0%}"), textposition="outside")
            fig.update_layout(barmode="group", title="Profile vs the rest", xaxis_tickformat=".0%", xaxis_range=[0, 1.15], legend=dict(orientation="h", y=-0.15), yaxis=dict(autorange="reversed"), height=380)
            show_chart(fig, key="ins_profile")
        with c2:
            fig = px.histogram(sub, x="age", nbins=20, color_discrete_sequence=[COLOR.get(choice, "#1F4E79")], title="Age distribution")
            fig.update_layout(yaxis_title="Clients", xaxis_title="Age (years)", bargap=0.05, height=380)
            show_chart(fig, key="ins_age")

        c1, c2 = st.columns(2)
        with c1:
            top_c = sub.region.value_counts().head(8).rename_axis("Region").reset_index(name="Clients")
            fig = px.bar(top_c.iloc[::-1], x="Clients", y="Region", orientation="h", title="Top regions", color_discrete_sequence=[COLOR.get(choice, "#1F4E79")])
            fig.update_layout(yaxis_title=None, height=340)
            show_chart(fig, key="ins_regions")
        with c2:
            sat = sub.satisfaction_score.value_counts(normalize=True).reindex([1, 2, 3, 4, 5]).fillna(0).reset_index()
            sat.columns = ["Score", "Share"]
            fig = px.bar(sat, x="Score", y="Share", title="Satisfaction score distribution", color_discrete_sequence=["#2A9D8F"], text=sat.Share.map(lambda v: f"{v:.0%}"))
            fig.update_layout(yaxis_tickformat=".0%", yaxis_title=None, height=340)
            show_chart(fig, key="ins_sat")

        if choice != "All clients in view":
            st.markdown("##### Recommended actions")
            for a in play["actions"]:
                st.markdown(f"- {a}")
        st.download_button("Download this segment as CSV", sub.drop(columns=["lat", "lon", "iso3"], errors="ignore").to_csv(index=False).encode("utf-8"),
                           file_name=f"segment_{choice.split(' - ')[0] if choice != 'All clients in view' else 'all'}.csv", mime="text/csv")

    st.markdown("##### All segments side by side (whole client base, unfiltered)")
    tbl = prof.copy()
    show = pd.DataFrame({"Segment": tbl.segment_name, "Clients": tbl.clients, "Client share": tbl.client_share.map(pct), "Revenue share": tbl.revenue_share.map(pct),
                         "Investment": tbl.pct_investment.map(pct), "Loan": tbl.pct_loan.map(pct), "US resident": tbl.pct_usa.map(pct), "Avg age": tbl.age_mean.round(1),
                         "Satisfaction": tbl.satisfaction.round(2), "Avg units": tbl.units_mean.round(2), "Avg spend": tbl.spend_mean.map(money), "Cohesion (silhouette)": tbl.mean_silhouette.round(2)})
    show_table(show)

# ------------------------------------------------------------------------------------------
# 5. MODEL & METHODOLOGY
# ------------------------------------------------------------------------------------------
with tab_model:
    st.subheader("How the segments were built, and how far to trust them")
    st.markdown("<p class='small-note'>This tab describes the model fitted once on all 2,000 clients; sidebar filters do not change it.</p>", unsafe_allow_html=True)
    a, b, c, d_ = st.columns(4)
    a.metric("Clusters (k)", res["best_k"]); b.metric("Silhouette", f"{res['silhouette']:.3f}", help="0.26-0.50 = weak structure; 0.25 or below = no substantial structure (Kaufman and Rousseeuw, 1990)")
    c.metric("Bootstrap stability (ARI)", f"{res['stability']:.2f}", help="Agreement between the full solution and re-fits on random 80% subsamples; 1 = identical")
    d_.metric("K-Means vs Ward (ARI)", f"{res['ward_ari']:.2f}", help="Agreement between the two clustering algorithms; 1 = identical")
    st.markdown(
        f"""
**Pipeline.** (1) Clean: strip whitespace, normalise labels, parse mixed date formats, drop duplicates. (2) Engineer purchase-behaviour features from `properties.csv`
(units bought, average unit price, office share). (3) Encode: one-hot for client type, purpose, loan, referral channel, country and a 13-level region group; label-encoded copies of these fields are exported for other tools but are not used in the distance calculation.
(4) Scale numeric features to [0, 1] with MinMaxScaler (age, satisfaction, log units, log average price, office share). (5) Fit K-Means and Ward hierarchical clustering for k = 2 to 10.
(6) Choose k with a pre-declared rule that ranks k = 3 to 8 on silhouette, Davies-Bouldin, Calinski-Harabasz, bootstrap stability and K-Means/Ward agreement.

**Reading the quality scores.** The silhouette of {res['silhouette']:.2f} is below the 0.26 threshold usually taken to indicate even weak structure. The segments are stable and reproducible ({res['stability']:.0%} bootstrap agreement) and the two algorithms largely agree,
but they are separated by three categorical attributes rather than by natural gaps in the data. Elbow analysis alone suggests k = {res['elbow_k']}; the extra cluster only splits the largest segment by referral channel (website-only vs agency/client), so k = {res['best_k']} was kept.
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=ms.k, y=ms.inertia, mode="lines+markers", line=dict(color="#1F4E79"), name="Inertia")
        fig.add_vline(x=res["best_k"], line_dash="dash", line_color="#E07B39", annotation_text=f"chosen k = {res['best_k']}")
        fig.update_layout(title="Elbow method (within-cluster SSE)", xaxis_title="k", yaxis_title="Inertia", xaxis=dict(dtick=1))
        show_chart(fig, key="md_elbow")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=ms.k, y=ms.silhouette_kmeans, mode="lines+markers", name="K-Means", line=dict(color="#1F4E79"))
        fig.add_scatter(x=ms.k, y=ms.silhouette_ward, mode="lines+markers", name="Ward", line=dict(color="#2A9D8F"))
        fig.add_hrect(y0=0, y1=0.25, fillcolor="#DDDDDD", opacity=0.3, line_width=0, annotation_text="no substantial structure", annotation_position="bottom right")
        fig.add_vline(x=res["best_k"], line_dash="dash", line_color="#E07B39")
        fig.update_layout(title="Silhouette score", xaxis_title="k", yaxis_title="Mean silhouette", yaxis_range=[0, 0.4], xaxis=dict(dtick=1), legend=dict(orientation="h", y=-0.2))
        show_chart(fig, key="md_sil")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=ms.k, y=ms.stability_ari, mode="lines+markers", error_y=dict(type="data", array=ms.stability_sd, visible=True), line=dict(color="#1F4E79"))
        fig.add_vline(x=res["best_k"], line_dash="dash", line_color="#E07B39")
        fig.update_layout(title="Bootstrap stability (ARI, 30 resamples)", xaxis_title="k", yaxis_title="ARI", yaxis_range=[0, 1.1], xaxis=dict(dtick=1))
        show_chart(fig, key="md_stab")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=ms.k, y=ms.kmeans_vs_ward_ari, mode="lines+markers", line=dict(color="#1F4E79"))
        fig.add_vline(x=res["best_k"], line_dash="dash", line_color="#E07B39")
        fig.update_layout(title="K-Means vs Ward hierarchical agreement (ARI)", xaxis_title="k", yaxis_title="ARI", yaxis_range=[-0.05, 1], xaxis=dict(dtick=1))
        show_chart(fig, key="md_ward")
    st.markdown("**All candidate solutions**")
    tbl = ms.rename(columns={"silhouette_kmeans": "Silhouette (K-Means)", "silhouette_ward": "Silhouette (Ward)", "davies_bouldin": "Davies-Bouldin", "calinski_harabasz": "Calinski-Harabasz",
                             "stability_ari": "Stability ARI", "kmeans_vs_ward_ari": "K-Means vs Ward ARI", "smallest_cluster": "Smallest cluster", "mean_rank": "Mean rank (k = 3-8)"})
    show_table(tbl.drop(columns=["stability_sd"]).round(3), hide_index=True)

    if (FIG / "fig06_dendrogram.png").exists():
        st.markdown("**Hierarchical validation: Ward dendrogram**")
        st.image(str(FIG / "fig06_dendrogram.png"), caption="Cutting the Ward tree at four clusters reproduces the K-Means segments for most clients.")

    st.markdown("#### Testing the four buyer personas suggested in the project brief")
    h = hyp.copy()
    h["Verdict"] = np.where((h.p_value < 0.05) & (h.effect.abs() >= 0.10), "Supported", "Not supported")
    h["p_value"] = h.p_value.round(3); h["effect"] = h.effect.round(3)
    h = h.rename(columns={"persona": "Persona", "claim": "Hypothesis", "test": "Test", "observed": "What the data show", "p_value": "p-value", "effect": "Effect size", "effect_name": "Effect measure"})
    show_table(h, hide_index=True)
    st.markdown("<p class='small-note'>None of the four personas is visible in the data: every effect is negligible (|effect| below 0.05). Data on income, prior ownership and units per transaction "
                "would be needed to identify these personas properly. See Section 6 of the paper.</p>", unsafe_allow_html=True)

    with st.expander("Data quality notes"):
        lg = res["log"]
        st.markdown(f"""
- Raw files: {lg['raw_clients']:,} clients and {lg['raw_properties']:,} property listings ({lg['sold_listings']:,} sold, {lg['available_listings']:,} available).
- Missing values in client fields: {lg['missing_client_cells']}. Duplicate client IDs: {lg['dup_client_ids']}. Orphan client references in sales: {lg['orphan_client_refs']}.
- Dates of birth mix `M/D/YYYY` ({lg['dob_slash_format']:,} rows) and `MM-DD-YYYY` ({lg['dob_dash_format']:,} rows). The dash convention was inferred from `properties.csv`. If it were day-first instead, ages would change by {lg['dob_ambiguity_mean_abs_age_diff']:.2f} years on average (maximum {lg['dob_ambiguity_max_abs_age_diff']:.2f}), which is immaterial.
- The source labels corporate clients as "Company"; it is shown as "Corporate" here to match the brief. "Home" purchases are shown as "Personal Use".
- For corporate clients the gender and date of birth fields presumably describe a contact person; treat their age with caution.
- Names and dates of birth are dropped from all processed files.
""")

# ------------------------------------------------------------------------------------------
# 6. DATA EXPLORER
# ------------------------------------------------------------------------------------------
with tab_data:
    st.subheader("Client-level data (current filters)")
    cols = ["client_id", "segment_label", "client_type", "gender", "age", "country", "region", "acquisition_purpose", "loan_applied", "referral_channel",
            "satisfaction_score", "n_units", "total_spend", "avg_price", "avg_ppsf", "share_office", "first_purchase", "last_purchase"]
    view = fdf[cols].copy()
    view["age"] = view.age.round(1); view["avg_price"] = view.avg_price.round(0); view["avg_ppsf"] = view.avg_ppsf.round(1); view["share_office"] = view.share_office.round(2)
    show_table(view, hide_index=True, height=520)
    st.download_button("Download filtered clients (CSV)", view.to_csv(index=False).encode("utf-8"), file_name="clients_filtered.csv", mime="text/csv")
