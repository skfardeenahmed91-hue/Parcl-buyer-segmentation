"""
Generates every figure used in the research paper (figures/*.png).
Run after pipeline.py:   python src/make_figures.py
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import stats
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC, FIG = ROOT / "data" / "processed", ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "axes.grid": True, "grid.color": "#E6E9ED", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight",
})
NAVY, ORANGE, TEAL, GREY, LIGHT = "#1F4E79", "#E07B39", "#2A9D8F", "#8A96A3", "#C9D1D9"

df = pd.read_csv(PROC / "clients_segmented.csv", parse_dates=["first_purchase", "last_purchase"])
ms = pd.read_csv(PROC / "model_selection.csv")
prof = pd.read_csv(PROC / "segment_profiles.csv", index_col=0)
meta = json.load(open(PROC / "segments.json"))
res = json.load(open(PROC / "results.json"))
hyp = pd.read_csv(PROC / "hypothesis_tests.csv")
imp = pd.read_csv(PROC / "feature_importance.csv")
mon = pd.read_csv(PROC / "monthly_market.csv", parse_dates=["month"])
tow = pd.read_csv(PROC / "tower_summary.csv")
tx = pd.read_csv(PROC / "transactions.csv", parse_dates=["month"])
K = res["best_k"]
seg_ids = list(meta.keys())
SEG_COL = {s: meta[s]["color"] for s in seg_ids}
short = {s: f"{s} {meta[s]['name'].replace('Domestic ', 'Dom. ').replace('International', 'Intl.')}" for s in seg_ids}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("saved", name)


def panel(ax, letter):
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")


# ----------------------------------------------------------------------------- Fig 1: client base
def fig_client_profile():
    fig, axs = plt.subplots(2, 3, figsize=(9.2, 5.4))
    ax = axs[0, 0]
    for t, c in [("Individual", NAVY), ("Corporate", ORANGE)]:
        ax.hist(df[df.client_type == t].age, bins=np.arange(20, 100, 5), color=c, alpha=.85 if t == "Individual" else .9, label=f"{t} (n={int((df.client_type == t).sum())})")
    ax.set_ylim(0, 235); ax.set_xlim(20, 100); ax.set_title("Age at end of 2025"); ax.set_xlabel("Age (years)"); ax.set_ylabel("Clients"); ax.legend(loc="upper left", ncol=1); panel(ax, "a")
    ax = axs[0, 1]
    c = df.country.value_counts()
    ax.barh(c.index[::-1], c.values[::-1], color=[NAVY if i == "USA" else GREY for i in c.index[::-1]])
    for y, v in enumerate(c.values[::-1]):
        ax.text(v + 15, y, f"{v / len(df):.0%}" if v / len(df) >= .05 else f"{v / len(df):.1%}", va="center", fontsize=7.5)
    ax.set_title("Country of residence"); ax.set_xlabel("Clients"); ax.set_xlim(0, c.max() * 1.15); panel(ax, "b")
    ax = axs[0, 2]
    r = df.satisfaction_score.value_counts().sort_index()
    ax.bar(r.index, r.values, color=TEAL, width=.7)
    ax.axhline(len(df) / 5, color="#333", ls="--", lw=.9); ax.text(0.55, len(df) / 5 + 12, "uniform benchmark (400)", ha="left", fontsize=7.5); ax.set_ylim(0, 470)
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_title("Satisfaction score (1-5)"); ax.set_xlabel("Score"); ax.set_ylabel("Clients"); panel(ax, "c")
    for ax, col, order, title, letter in [(axs[1, 0], "acquisition_purpose", ["Personal Use", "Investment"], "Acquisition purpose", "d"),
                                          (axs[1, 1], "loan_applied", ["No", "Yes"], "Loan applied", "e"),
                                          (axs[1, 2], "referral_channel", ["Website", "Agency", "Client"], "Referral channel", "f")]:
        v = df[col].value_counts().reindex(order)
        ax.bar(v.index, v.values, color=[NAVY, ORANGE, TEAL][:len(v)], width=.6)
        for i, x in enumerate(v.values):
            ax.text(i, x + 15, f"{x / len(df):.1%}", ha="center", fontsize=8)
        ax.set_title(title); ax.set_ylabel("Clients"); ax.set_ylim(0, v.max() * 1.15); panel(ax, letter)
    fig.tight_layout(h_pad=1.8, w_pad=1.6)
    save(fig, "fig01_client_profile")


# ----------------------------------------------------------------------------- Fig 2: purchase behaviour
def fig_purchase_behaviour():
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.1))
    ax = axs[0]
    u = df.n_units.value_counts().sort_index()
    ax.bar(u.index, u.values, color=[NAVY if i < 5 else ORANGE for i in u.index], width=.75)
    ax.set_xticks(range(3, 14)); ax.set_yscale("log"); ax.set_title("Units purchased per client"); ax.set_xlabel("Units"); ax.set_ylabel("Clients (log scale)")
    ax.text(0.98, 0.95, f"{res['portfolio_clients']} clients ({res['portfolio_clients'] / len(df):.0%})\nbuy 5+ units", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=ORANGE)
    panel(ax, "a")
    ax = axs[1]
    ax.hist(df.total_spend / 1e6, bins=40, color=NAVY)
    ax.axvline(df.total_spend.median() / 1e6, color=ORANGE, lw=1.6)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2); ax.text(df.total_spend.median() / 1e6 + .3, ax.get_ylim()[1] * .9, f"median ${df.total_spend.median() / 1e6:.2f}M", color=ORANGE, fontsize=8)
    ax.set_title("Total spend per client"); ax.set_xlabel("USD millions"); ax.set_ylabel("Clients"); panel(ax, "b")
    ax = axs[2]
    sold = tx[tx.is_sold]
    for cat, col in [("Apartment", NAVY), ("Office", ORANGE)]:
        ax.hist(sold[sold.unit_category == cat].price_per_sqft, bins=np.arange(200, 440, 8), color=col, alpha=.75, density=True, label=cat)
    ax.set_title("Price per sq ft by unit type"); ax.set_xlabel("USD per sq ft"); ax.set_ylabel("Density"); ax.legend(); panel(ax, "c")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig02_purchase_behaviour")


# ----------------------------------------------------------------------------- Fig 3: market context
def fig_market():
    fig, axs = plt.subplots(1, 3, figsize=(9.4, 3.2))
    ax = axs[0]
    ax.bar(mon.month, mon.listed, width=22, color=LIGHT, label="Listed")
    ax.bar(mon.month, mon.sold, width=22, color=NAVY, label="Sold")
    ax.set_ylim(0, 640); ax.set_title("Monthly listings and sales"); ax.set_ylabel("Units"); ax.legend(loc="upper right", ncol=2)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%y")); ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    panel(ax, "a")
    ax = axs[1]
    ax.plot(mon.month, mon.ppsf, color=TEAL, marker="o", ms=3.5, lw=1.6)
    ax.axhline(res["ppsf_mean"], color="#333", ls="--", lw=.8)
    ax.set_ylim(250, 350); ax.set_title("Average sale price per sq ft"); ax.set_ylabel("USD per sq ft")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%y")); ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.text(0.03, 0.06, f"Spearman rho vs. time = {res['ppsf_time_rho']:.3f} (p = {res['ppsf_time_p']:.2f})", transform=ax.transAxes, fontsize=7.5)
    panel(ax, "b")
    ax = axs[2]
    t = tow.sort_values("tower_number")
    ax.bar(t.tower_number, t.avg_price / 1e3, color=[ORANGE if v > 355e3 else NAVY for v in t.avg_price])
    ax.set_ylim(0, 470); ax.text(0.03, 0.95, "orange = average above $355k", transform=ax.transAxes, fontsize=7.5, color=ORANGE, fontweight="bold", va="top"); ax.set_title("Average list price by tower"); ax.set_xlabel("Tower"); ax.set_ylabel("USD thousands"); ax.set_xticks([1, 5, 10, 15, 20]); panel(ax, "c")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig03_market_context")


# ----------------------------------------------------------------------------- Fig 4: association matrix
def _cramers_v_bc(x, y):
    ct = pd.crosstab(x, y); n = ct.values.sum(); chi2 = stats.chi2_contingency(ct, correction=False)[0]
    phi2 = chi2 / n; r, k = ct.shape
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1)); rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2c / max(min(kc - 1, rc - 1), 1e-9))


def _eta(cat, num):
    g = [num[cat == c] for c in cat.unique()]; grand = num.mean()
    ssb = sum(len(x) * (x.mean() - grand) ** 2 for x in g); sst = ((num - grand) ** 2).sum()
    return np.sqrt(ssb / sst)


def fig_association():
    d = df.copy(); d["country_group"] = np.where(d.is_usa == 1, "USA", "Intl")
    cats = {"Client type": "client_type", "Gender": "gender", "Country (US/Intl)": "country_group", "Purpose": "acquisition_purpose",
            "Loan applied": "loan_applied", "Referral channel": "referral_channel"}
    nums = {"Age": "age", "Satisfaction": "satisfaction_score", "Units bought": "n_units", "Total spend": "total_spend",
            "Avg unit price": "avg_price", "Office share": "share_office"}
    names = list(cats) + list(nums); n = len(names); M = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j: M[i, j] = np.nan; continue
            ai, bj = cats.get(a), cats.get(b)
            if ai and bj: M[i, j] = _cramers_v_bc(d[ai], d[bj])
            elif not ai and not bj: M[i, j] = abs(stats.spearmanr(d[nums[a]], d[nums[b]])[0])
            else: M[i, j] = _eta(d[ai or bj], d[nums[b] if ai else nums[a]])
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names); ax.grid(False)
    for i in range(n):
        for j in range(n):
            if i != j: ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.8, color="white" if M[i, j] > .55 else "#333")
    ax.axhline(5.5, color="#333", lw=.8); ax.axvline(5.5, color="#333", lw=.8)
    cb = fig.colorbar(im, fraction=.046, pad=.03); cb.set_label("Association strength (0 = none, 1 = perfect)")
    ax.set_title("Pairwise association between client attributes", pad=10)
    save(fig, "fig04_association")
    off = M[~np.isnan(M)]
    return float(np.median(off)), float(np.nanmax(M[:6, 6:])), M, names


# ----------------------------------------------------------------------------- Fig 5: model selection
def fig_model_selection():
    fig, axs = plt.subplots(2, 3, figsize=(9.4, 5.6)); axs = axs.ravel()
    k = ms.k
    def mark(ax):
        ax.axvspan(K - .4, K + .4, color=ORANGE, alpha=.15, lw=0)
        ax.set_xticks(k); ax.set_xlabel("Number of clusters (k)")
    ax = axs[0]; ax.plot(k, ms.inertia, marker="o", color=NAVY); mark(ax)
    ax.annotate(f"elbow rule: k={res['elbow_k']}", (res["elbow_k"], ms.loc[ms.k == res["elbow_k"], "inertia"].iloc[0]), xytext=(res["elbow_k"] + .7, ms.inertia.max() * .9),
                arrowprops=dict(arrowstyle="->", color="#555"), fontsize=8)
    ax.set_title("Elbow: within-cluster SSE"); ax.set_ylabel("Inertia"); panel(ax, "a")
    ax = axs[1]; ax.plot(k, ms.silhouette_kmeans, marker="o", color=NAVY, label="K-Means"); ax.plot(k, ms.silhouette_ward, marker="s", color=TEAL, label="Ward"); mark(ax)
    ax.axhspan(0, 0.25, color="#DDDDDD", alpha=.35, lw=0); ax.text(10.4, 0.012, "no substantial structure (<= 0.25)", ha="right", fontsize=7)
    ax.set_ylim(0, .35); ax.set_title("Silhouette score"); ax.set_ylabel("Mean silhouette"); ax.legend(loc="upper left"); panel(ax, "b")
    ax = axs[2]; ax.plot(k, ms.davies_bouldin, marker="o", color=NAVY); mark(ax); ax.set_title("Davies-Bouldin (lower is better)"); ax.set_ylabel("Index"); panel(ax, "c")
    ax = axs[3]; ax.plot(k, ms.calinski_harabasz, marker="o", color=NAVY); mark(ax); ax.set_title("Calinski-Harabasz (higher is better)"); ax.set_ylabel("Index"); panel(ax, "d")
    ax = axs[4]; ax.errorbar(k, ms.stability_ari, yerr=ms.stability_sd, marker="o", color=NAVY, capsize=2.5, lw=1.3, elinewidth=.8); mark(ax)
    ax.set_ylim(0, 1.05); ax.set_title("Bootstrap stability (ARI)"); ax.set_ylabel("Adjusted Rand index"); panel(ax, "e")
    ax = axs[5]; ax.plot(k, ms.kmeans_vs_ward_ari, marker="o", color=NAVY); mark(ax); ax.set_ylim(-.05, 1); ax.set_title("K-Means vs Ward agreement"); ax.set_ylabel("Adjusted Rand index"); panel(ax, "f")
    fig.tight_layout(h_pad=1.8, w_pad=1.4)
    save(fig, "fig05_model_selection")


# ----------------------------------------------------------------------------- Fig 6: dendrogram
def fig_dendrogram():
    Z = np.load(PROC / "ward_linkage.npy")
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    dendrogram(Z, truncate_mode="lastp", p=14, ax=ax, color_threshold=0, above_threshold_color=NAVY, leaf_rotation=0, leaf_font_size=8, show_contracted=False)
    cut = (Z[-(K - 1), 2] + Z[-K, 2]) / 2
    ax.axhline(cut, color=ORANGE, ls="--", lw=1.4); ax.text(ax.get_xlim()[1] * .99, cut * 1.03, f"cut at {K} clusters", color=ORANGE, ha="right", fontsize=8.5)
    ax.set_title("Ward hierarchical clustering (last 14 merges shown; leaf labels give cluster size)"); ax.set_ylabel("Ward distance"); ax.set_xlabel("")
    ax.grid(False)
    save(fig, "fig06_dendrogram")


# ----------------------------------------------------------------------------- Fig 7: PCA + importance
def fig_pca_importance():
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    rng = np.random.default_rng(1)
    for s in seg_ids:
        m = df.segment_id == s
        ax.scatter(df.loc[m, "pc1"] + rng.normal(0, .015, m.sum()), df.loc[m, "pc2"] + rng.normal(0, .015, m.sum()), s=9, alpha=.55, color=SEG_COL[s], label=short[s], lw=0)
    ax.set_xlabel(f"PC1 ({res['pca_var'][0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({res['pca_var'][1]:.0%})")
    ax.set_title("Segments in principal-component space"); ax.legend(markerscale=2.2, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2); panel(ax, "a")
    ax = axs[1]
    top = imp.head(7).iloc[::-1]
    lab = {"is_usa": "US resident", "investment_purpose": "Investment purpose", "loan_applied": "Loan applied", "corporate": "Corporate client", "age": "Age",
           "satisfaction": "Satisfaction", "units": "Units bought", "avg_price": "Avg unit price", "office_share": "Office share", "california": "California",
           "ref_website": "Referral: website", "ref_agency": "Referral: agency", "ref_client": "Referral: client"}
    ax.barh([lab[f] for f in top.feature], top.importance, color=[NAVY if v > .01 else LIGHT for v in top.importance])
    ax.set_title("What defines a segment?\n(permutation importance)"); ax.set_xlabel("Drop in accuracy when shuffled"); panel(ax, "b")
    fig.tight_layout(w_pad=2)
    save(fig, "fig07_pca_importance")


# ----------------------------------------------------------------------------- Fig 8: profile heatmap
def fig_profile_heatmap():
    top_cols = [("pct_usa", "US resident"), ("pct_investment", "Investment purpose"), ("pct_loan", "Loan applied"), ("pct_corporate", "Corporate client")]
    bot_cols = [("age_mean", "Mean age (years)", "{:.1f}", "age"), ("satisfaction", "Satisfaction (1-5)", "{:.2f}", "satisfaction_score"), ("units_mean", "Units per client", "{:.2f}", "n_units"),
                ("avg_price", "Avg unit price", "${:,.0f}", "avg_price"), ("spend_mean", "Total spend per client", "${:,.0f}", "total_spend"), ("pct_portfolio", "Buys 5+ units", "{:.1%}", "portfolio_buyer")]
    overall = {"age": df.age.mean(), "satisfaction_score": df.satisfaction_score.mean(), "n_units": df.n_units.mean(), "avg_price": df.avg_price.mean(),
               "total_spend": df.total_spend.mean(), "portfolio_buyer": df.portfolio_buyer.mean()}
    fig = plt.figure(figsize=(9.0, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 6], width_ratios=[40, 1.2], hspace=.32, wspace=.04)
    labels = [f"{s}\n{meta[s]['name']}\n(n={int(prof.loc[s, 'clients'])})".replace("Domestic ", "Domestic\n") for s in seg_ids]
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0]); cax1 = fig.add_subplot(gs[0, 1]); cax2 = fig.add_subplot(gs[1, 1])
    A = prof.loc[seg_ids, [c[0] for c in top_cols]].values.T
    im1 = ax1.imshow(A, cmap="Blues", vmin=0, vmax=1, aspect="auto"); ax1.grid(False)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]): ax1.text(j, i, f"{A[i, j]:.0%}", ha="center", va="center", fontsize=9, color="white" if A[i, j] > .55 else "#222")
    ax1.set_yticks(range(len(top_cols))); ax1.set_yticklabels([c[1] for c in top_cols]); ax1.set_xticks(range(len(seg_ids))); ax1.set_xticklabels(labels, fontsize=8); ax1.xaxis.tick_top()
    ax1.set_title("What defines a segment (share of clients)", fontsize=9.5, loc="left", pad=44)
    fig.colorbar(im1, cax=cax1).set_label("share", fontsize=7.5)
    B = np.array([[prof.loc[s, c[0]] for s in seg_ids] for c in bot_cols]); D = np.array([[prof.loc[s, c[0]] / overall[c[3]] - 1 for s in seg_ids] for c in bot_cols])
    im2 = ax2.imshow(D, cmap="RdBu_r", vmin=-.10, vmax=.10, aspect="auto"); ax2.grid(False)
    for i, c in enumerate(bot_cols):
        for j in range(len(seg_ids)): ax2.text(j, i, c[2].format(B[i, j]), ha="center", va="center", fontsize=8.5, color="white" if abs(D[i, j]) > .07 else "#222")
    ax2.set_yticks(range(len(bot_cols))); ax2.set_yticklabels([c[1] for c in bot_cols]); ax2.set_xticks([])
    ax2.set_title("Demographics and behaviour (colour = % difference from the all-client average, capped at \u00b110%)", fontsize=9.5, loc="left")
    cb2 = fig.colorbar(im2, cax=cax2); cb2.set_ticks([-.10, -.05, 0, .05, .10]); cb2.set_ticklabels(["-10%", "-5%", "0%", "+5%", "+10%"]); cb2.set_label("vs average", fontsize=7.5)
    save(fig, "fig08_profile_heatmap")


# ----------------------------------------------------------------------------- Fig 9: composition
def fig_composition():
    fig, axs = plt.subplots(1, 4, figsize=(9.6, 3.3), sharey=True)
    def stack(ax, col, cats, colors, title, letter):
        ct = pd.crosstab(df.segment_id, df[col], normalize="index").reindex(seg_ids)[cats]
        bottom = np.zeros(len(seg_ids))
        for c, colr in zip(cats, colors):
            ax.bar(range(len(seg_ids)), ct[c].values, bottom=bottom, color=colr, width=.7, label=c)
            for i, v in enumerate(ct[c].values):
                if v > .07: ax.text(i, bottom[i] + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=7.5, color="#222" if colr == LIGHT else "white")
            bottom += ct[c].values
        ax.set_xticks(range(len(seg_ids))); ax.set_xticklabels(seg_ids); ax.set_title(title); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=1, fontsize=7.5); ax.grid(False); panel(ax, letter)
    d = df.copy(); d["geo"] = np.where(d.is_usa == 1, "United States", "International")
    stack(axs[0], "acquisition_purpose", ["Investment", "Personal Use"], [NAVY, LIGHT], "Purpose", "a")
    stack(axs[1], "loan_applied", ["Yes", "No"], [ORANGE, LIGHT], "Loan applied", "b")
    df["geo"] = d.geo
    stack(axs[2], "geo", ["United States", "International"], [TEAL, LIGHT], "Residence", "c")
    stack(axs[3], "referral_channel", ["Website", "Agency", "Client"], [NAVY, ORANGE, TEAL], "Referral channel", "d")
    axs[0].set_ylabel("Share of segment"); axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig09_composition")


# ----------------------------------------------------------------------------- Fig 10: geography
def fig_geography():
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    top = df.region.value_counts().head(14).index
    ct = pd.crosstab(df.region, df.segment_id).loc[top][seg_ids].iloc[::-1]
    left = np.zeros(len(ct))
    for s in seg_ids:
        ax.barh(ct.index, ct[s].values, left=left, color=SEG_COL[s], label=short[s], height=.72); left += ct[s].values
    for i, v in enumerate(left): ax.text(v + 6, i, int(v), va="center", fontsize=7.5)
    ax.set_title("Clients by region (14 largest) and segment"); ax.set_xlabel("Clients"); ax.legend(loc="lower right", fontsize=7.5); ax.set_xlim(0, left.max() * 1.1); panel(ax, "a")
    ax = axs[1]
    cty = df.groupby("country").agg(n=("client_id", "size"), inv=("investment_flag", "mean"), loan=("loan_flag", "mean")).sort_values("n", ascending=False)
    x = np.arange(len(cty)); w = .38
    def wilson(p, n, z=1.96):
        den = 1 + z ** 2 / n; c = (p + z ** 2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
        return np.array([p - (c - h), (c + h) - p])
    ci_i = np.array([wilson(p, n) for p, n in zip(cty.inv, cty.n)]).T; ci_l = np.array([wilson(p, n) for p, n in zip(cty.loan, cty.n)]).T
    ax.bar(x - w / 2, cty.inv, w, color=NAVY, label="Investment share", yerr=ci_i, error_kw=dict(lw=.8, capsize=1.5, ecolor="#333"))
    ax.bar(x + w / 2, cty.loan, w, color=ORANGE, label="Loan share", yerr=ci_l, error_kw=dict(lw=.8, capsize=1.5, ecolor="#333"))
    ax.axhline(df.investment_flag.mean(), color=NAVY, ls=":", lw=1); ax.axhline(df.loan_flag.mean(), color=ORANGE, ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cty.index, cty.n)], rotation=90, fontsize=7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}")); ax.set_title("Investment and loan share by country (95% CI)"); ax.legend(loc="upper right", ncol=2, fontsize=7.5); ax.set_ylim(0, .85); panel(ax, "b")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig10_geography")


# ----------------------------------------------------------------------------- Fig 11: hypothesis tests (bootstrap CIs)
def fig_hypotheses():
    rng = np.random.default_rng(7); B = 1000; rows = []
    n = len(df)
    def boot(fn):
        vals = []
        for _ in range(B):
            s = df.iloc[rng.integers(0, n, n)].reset_index(drop=True); vals.append(fn(s))
        return np.percentile(vals, [2.5, 97.5])
    def v_intl_inv(s):
        ct = pd.crosstab(s.is_usa, s.investment_flag).values; chi2 = stats.chi2_contingency(ct, correction=False)[0]; return np.sqrt(chi2 / ct.sum())
    def r_age_loan(s):
        a, b = s[s.loan_flag == 1].age, s[s.loan_flag == 0].age; u = stats.mannwhitneyu(a, b)[0]; return 1 - 2 * u / (len(a) * len(b))
    def r_corp_units(s):
        a, b = s[s.client_type == "Corporate"].n_units, s[s.client_type == "Individual"].n_units; u = stats.mannwhitneyu(a, b)[0]; return 1 - 2 * u / (len(a) * len(b))
    def rho_sat(s): return stats.spearmanr(s.satisfaction_score, s.total_spend)[0]
    fns = [v_intl_inv, r_age_loan, r_corp_units, rho_sat]
    labels = ["C1 Global Investors\nintl. buyers invest more\n(Cramer's V)", "C2 First-Time Buyers\nyounger buyers use loans more\n(rank-biserial r)",
              "C3 Corporate Buyers\ncorporates buy more units\n(rank-biserial r)", "C4 Luxury Investors\nsatisfaction rises with spend\n(Spearman rho)"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.axvspan(-.1, .1, color="#DDE3EA", alpha=.7, lw=0); ax.axvline(0, color="#333", lw=.8)
    for i, (fn, e) in enumerate(zip(fns, hyp.effect)):
        lo, hi = boot(fn); y = len(fns) - 1 - i
        ax.plot([lo, hi], [y, y], color=NAVY, lw=2); ax.plot(e, y, "o", color=ORANGE, ms=8, zorder=3)
        ax.text(.36, y, f"p = {hyp.p_value.iloc[i]:.2f}", va="center", fontsize=8.5)
    ax.set_ylim(-.6, len(fns) - .4); ax.set_yticks(range(len(fns))); ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlim(-.3, .45); ax.set_xlabel("Effect size with 95% bootstrap interval (shaded band = negligible, |effect| < 0.10)")
    ax.set_title("None of the four hypothesised personas is supported by the data"); ax.grid(axis="y", visible=False)
    save(fig, "fig11_hypotheses")


# ----------------------------------------------------------------------------- Fig 12: segment behaviour over time + k=5 check
def fig_extra():
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.3))
    ax = axs[0]
    sold = tx[tx.is_sold & tx.segment_id.notna()]
    m = sold.groupby(["month", "segment_id"]).size().unstack().reindex(columns=seg_ids)
    share = m.div(m.sum(axis=1), axis=0)
    for s in seg_ids:
        ax.plot(share.index, share[s], color=SEG_COL[s], lw=1.6, label=short[s])
    ax.set_ylim(0, .5); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}")); ax.set_title("Segment share of monthly unit sales"); ax.legend(fontsize=7, ncol=2, loc="upper center")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%y")); ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10])); panel(ax, "a")
    ax = axs[1]
    X = pd.read_csv(PROC / "model_matrix.csv").drop(columns="client_id").values
    k5 = KMeans(5, n_init=50, random_state=42).fit_predict(X)
    ct = pd.crosstab(df.segment_id, k5).reindex(seg_ids)
    im = ax.imshow(ct.values, cmap="Blues", aspect="auto"); ax.grid(False)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"k5-{i + 1}" for i in range(5)]); ax.set_yticks(range(len(seg_ids))); ax.set_yticklabels(seg_ids)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            if ct.values[i, j]: ax.text(j, i, int(ct.values[i, j]), ha="center", va="center", fontsize=8.5, color="white" if ct.values[i, j] > 350 else "#222")
    ax.set_title("What k = 5 adds: the k = 4 segments cross-tabulated\nagainst the k = 5 solution", fontsize=9.5); ax.set_xlabel("k = 5 cluster"); ax.set_ylabel("k = 4 segment"); panel(ax, "b")
    fig.tight_layout(w_pad=1.8)
    save(fig, "fig12_time_and_k5")
    web = df.loc[df.segment_id == "S1"].referral_channel
    return dict(k5_cross=ct.to_dict(), s1_split=[int((k5[df.segment_id == "S1"] == c).sum()) for c in range(5)])


if __name__ == "__main__":
    fig_client_profile(); fig_purchase_behaviour(); fig_market()
    med_assoc, max_cn, M, names = fig_association()
    fig_model_selection(); fig_dendrogram(); fig_pca_importance(); fig_profile_heatmap(); fig_composition(); fig_geography(); fig_hypotheses()
    extra = fig_extra()
    res.update(assoc_median=med_assoc, assoc_max_cat_num=max_cn, k5_s1_split=extra["s1_split"])
    # largest association excluding the country-group<->... none: record strongest pair for the paper
    idx = np.dstack(np.unravel_index(np.argsort(-np.nan_to_num(M).ravel()), M.shape))[0]
    pairs = []
    for i, j in idx:
        if i < j and len(pairs) < 6: pairs.append((names[i], names[j], float(M[i, j])))
    res["top_assoc_pairs"] = pairs
    json.dump(res, open(PROC / "results.json", "w"), indent=2, default=str)
    print("median off-diagonal association:", round(med_assoc, 3)); print("top pairs:", pairs); print("k5 S1 split:", extra["s1_split"])
