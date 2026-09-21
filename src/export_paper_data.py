"""
Collects every number the research paper cites into data/processed/paper_data.json,
so the Word document is generated from the analysis outputs and nothing is typed by hand.
Run after pipeline.py, make_figures.py and design_comparison.py.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC, FIG = ROOT / "data" / "processed", ROOT / "figures"

df = pd.read_csv(PROC / "clients_segmented.csv")
tx = pd.read_csv(PROC / "transactions.csv", parse_dates=["month"])
res = json.load(open(PROC / "results.json"))
meta = json.load(open(PROC / "segments.json"))
prof = pd.read_csv(PROC / "segment_profiles.csv", index_col=0)
ms = pd.read_csv(PROC / "model_selection.csv")
hyp = pd.read_csv(PROC / "hypothesis_tests.csv")
diff = pd.read_csv(PROC / "segment_differences.csv")
imp = pd.read_csv(PROC / "feature_importance.csv")
design = pd.read_csv(PROC / "design_comparison.csv")
mon = pd.read_csv(PROC / "monthly_market.csv", parse_dates=["month"])
tow = pd.read_csv(PROC / "tower_summary.csv")
sold = tx[tx.is_sold]
seg_ids = list(meta.keys())

out = {"res": {k: v for k, v in res.items() if k not in ("segment_meta",)}, "meta": meta}

# ---- descriptives
out["desc"] = dict(
    age_mean=df.age.mean(), age_sd=df.age.std(), age_min=df.age.min(), age_max=df.age.max(),
    age_corp=df[df.client_type == "Corporate"].age.mean(), age_ind=df[df.client_type == "Individual"].age.mean(),
    n_corp=int((df.client_type == "Corporate").sum()), n_ind=int((df.client_type == "Individual").sum()),
    corp_age_p=float(stats.mannwhitneyu(df[df.client_type == "Corporate"].age, df[df.client_type == "Individual"].age)[1]),
    sat_mean=df.satisfaction_score.mean(), sat_low_share=float((df.satisfaction_score <= 2).mean()),
    sat_dist={int(k): float(v) for k, v in df.satisfaction_score.value_counts(normalize=True).sort_index().items()},
    country_counts={k: int(v) for k, v in df.country.value_counts().items()}, usa_share=float(df.is_usa.mean()),
    n_regions=int(df.region.nunique()), california=int((df.region == "California").sum()),
    top_regions={k: int(v) for k, v in df.region.value_counts().head(6).items()},
    male_share=float((df.gender == "Male").mean()), inv_share=float(df.investment_flag.mean()), loan_share=float(df.loan_flag.mean()),
    ref={k: float(v) for k, v in df.referral_channel.value_counts(normalize=True).items()},
    units_dist={int(k): int(v) for k, v in df.n_units.value_counts().sort_index().items()}, units_mean=df.n_units.mean(),
    share_3_4=float(df.n_units.isin([3, 4]).mean()), units_max=int(df.n_units.max()),
    spend_mean=df.total_spend.mean(), spend_median=df.total_spend.median(), spend_min=df.total_spend.min(), spend_max=df.total_spend.max(),
    avg_price_mean=df.avg_price.mean(), price_mean=sold.price.mean(), price_min=sold.price.min(), price_max=sold.price.max(),
    area_mean=sold.floor_area_sqft.mean(), office_share_sold=float((sold.unit_category == "Office").mean()), office_share_client=df.share_office.mean(),
    tower_price_min=tow.avg_price.min(), tower_price_max=tow.avg_price.max(), tower_ppsf_min=tow.ppsf.min(), tower_ppsf_max=tow.ppsf.max(),
    tower_st_min=tow.sell_through.min(), tower_st_max=tow.sell_through.max(),
    ppsf_month_min=mon.ppsf.min(), ppsf_month_max=mon.ppsf.max(),
    dec24_listed=int(mon.loc[mon.month == "2024-12-01", "listed"].iloc[0]), jan25_listed=int(mon.loc[mon.month == "2025-01-01", "listed"].iloc[0]),
    listed_drop=float(res["listings_2025_per_month"] / res["listings_2024_per_month"] - 1),
    sold_drop=float(res["sold_2025_per_month"] / res["sold_2024_per_month"] - 1),
    rev_2024=float(sold[sold.month.dt.year == 2024].price.sum()), rev_2025=float(sold[sold.month.dt.year == 2025].price.sum()),
)
# US vs international behaviour
usa, intl = df[df.is_usa == 1], df[df.is_usa == 0]
ct = pd.crosstab(df.is_usa, df.portfolio_buyer)
out["geo"] = dict(
    intl_5plus=int(intl.portfolio_buyer.sum()), intl_n=len(intl), usa_5plus=int(usa.portfolio_buyer.sum()), usa_n=len(usa),
    fisher_p=float(stats.fisher_exact(ct)[1]),
    units_usa=usa.n_units.mean(), units_intl=intl.n_units.mean(), spend_usa=usa.total_spend.mean(), spend_intl=intl.total_spend.mean(),
    spend_p=float(stats.mannwhitneyu(usa.total_spend, intl.total_spend)[1]), units_p=float(stats.mannwhitneyu(usa.n_units, intl.n_units)[1]),
    chi_inv_country_p=float(stats.chi2_contingency(pd.crosstab(df.country, df.investment_flag))[1]),
    chi_loan_country_p=float(stats.chi2_contingency(pd.crosstab(df.country, df.loan_flag))[1]),
    us_region_seg_p=float(stats.chi2_contingency(pd.crosstab(usa.region, usa.segment_id))[1]),
    us_states=int(usa.region.nunique()),
)
# k=5: what the fifth cluster adds
from sklearn.cluster import KMeans
X = pd.read_csv(PROC / "model_matrix.csv").drop(columns="client_id").values
k5 = KMeans(5, n_init=50, random_state=42).fit_predict(X)
s1 = df.segment_id == "S1"
cross = pd.crosstab(df.referral_channel[s1], k5[s1])
out["k5"] = dict(split={int(c): int((k5[s1] == c).sum()) for c in np.unique(k5[s1])},
                 referral_by_cluster={int(c): {k: int(v) for k, v in cross[c].items()} for c in cross.columns})

# ---- tables
overall = dict(
    clients=len(df), client_share=1.0, revenue_share=1.0, pct_usa=df.is_usa.mean(), pct_investment=df.investment_flag.mean(), pct_loan=df.loan_flag.mean(),
    pct_corporate=(df.client_type == "Corporate").mean(), age_mean=df.age.mean(), satisfaction=df.satisfaction_score.mean(), units_mean=df.n_units.mean(),
    avg_price=df.avg_price.mean(), spend_mean=df.total_spend.mean(), office_share=df.share_office.mean(), pct_portfolio=df.portfolio_buyer.mean(),
    pct_ref_website=(df.referral_channel == "Website").mean(), pct_ref_agency=(df.referral_channel == "Agency").mean(), pct_ref_client=(df.referral_channel == "Client").mean(),
    mean_silhouette=df.silhouette.mean(), ward_agreement=df.ward_agrees.mean())
out["profile"] = {s: prof.loc[s].to_dict() for s in seg_ids}
out["profile_overall"] = overall
out["model_selection"] = ms.to_dict("records")
out["hypotheses"] = hyp.to_dict("records")
out["seg_diff"] = diff.to_dict("records")
out["importance"] = imp.to_dict("records")
out["design"] = design.to_dict("records")
out["silhouette_by_segment"] = {s: float(v) for s, v in df.groupby("segment_id").silhouette.mean().items()}


# ---- investors vs personal-use buyers (the "investment profile")
def _grp(g):
    return dict(n=int(len(g)), units=g.n_units.mean(), spend=g.total_spend.mean(), spend_median=g.total_spend.median(), avg_price=g.avg_price.mean(),
                ppsf=g.avg_ppsf.mean(), office=g.share_office.mean(), loan=g.loan_flag.mean(), corp=(g.client_type == "Corporate").mean(),
                sat=g.satisfaction_score.mean(), age=g.age.mean(), portfolio=g.portfolio_buyer.mean(), usa=g.is_usa.mean())
inv, per = df[df.investment_flag == 1], df[df.investment_flag == 0]
out["inv_vs_personal"] = dict(investment=_grp(inv), personal=_grp(per),
    p_spend=float(stats.mannwhitneyu(inv.total_spend, per.total_spend)[1]), p_units=float(stats.mannwhitneyu(inv.n_units, per.n_units)[1]),
    p_price=float(stats.mannwhitneyu(inv.avg_price, per.avg_price)[1]), p_office=float(stats.mannwhitneyu(inv.share_office, per.share_office)[1]),
    p_loan=float(stats.chi2_contingency(pd.crosstab(df.investment_flag, df.loan_flag), correction=False)[1]))

# ---- software versions (for the reproducibility note)
import sys, sklearn, scipy, matplotlib, plotly
try:
    import streamlit
    st_v = streamlit.__version__
except Exception:
    st_v = "n/a"
out["versions"] = dict(python=sys.version.split()[0], sklearn=sklearn.__version__, scipy=scipy.__version__, pandas=pd.__version__, numpy=np.__version__,
                       matplotlib=matplotlib.__version__, plotly=plotly.__version__, streamlit=st_v)


# ---- change in units sold 2024 -> 2025, per segment
_s = sold[sold.segment_id.notna()].copy(); _s["yr"] = _s.month.dt.year
_yy = _s.groupby(["segment_id", "yr"]).size().unstack()
out["seg_change"] = {k: float(v) for k, v in (_yy[2025] / _yy[2024] - 1).items()}

# ---- figure sizes (pixels) so that the builder can keep aspect ratios
out["images"] = {}
for f in sorted(FIG.glob("fig*.png")):
    with Image.open(f) as im:
        out["images"][f.stem] = dict(path=str(f), w=im.width, h=im.height)

def _clean(o):
    """Recursively convert numpy types and NaN/inf to JSON-safe values (NaN -> null)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    return o


json.dump(_clean(out), open(PROC / "paper_data.json", "w"), indent=1, default=str, allow_nan=False)
print("written", PROC / "paper_data.json")
print("k5 split of S1:", out["k5"])
print("geo:", json.dumps(out["geo"], indent=1))
print("corp age p:", out["desc"]["corp_age_p"], "| volume step Dec-24 -> Jan-25:", out["desc"]["dec24_listed"], "->", out["desc"]["jan25_listed"])
