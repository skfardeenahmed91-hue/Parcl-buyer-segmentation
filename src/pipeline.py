"""
Buyer Segmentation & Investment Profiling - reproducible analysis pipeline
==========================================================================
Parcl Co. Limited x Unified Mentor

Run from the project root:   python src/pipeline.py

Stages
  1. Load + clean            (data/raw  -> tidy client and transaction tables)
  2. Feature engineering     (purchase behaviour joined from properties.csv)
  3. Encoding + scaling      (one-hot, label encoding, MinMax)
  4. Model selection         (K-Means + Ward, elbow / silhouette / DB / CH / stability)
  5. Final segmentation      (K-Means, validated against Ward hierarchical)
  6. Interpretation          (profiles, tests, surrogate tree, PRD hypothesis tests)
  7. Export                  (data/processed/*; figures come from src/make_figures.py)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
SEED = 42
AS_OF = pd.Timestamp("2025-12-31")      # end of the transaction window; age is measured at this date
K_RANGE = range(2, 11)                   # range searched for the elbow / silhouette curves
K_CANDIDATES = range(3, 9)               # k values a marketing team can act on (3-8 segments)
N_BOOT, BOOT_FRAC = 30, 0.80             # bootstrap stability: 30 subsamples of 80 %

# Approximate region centroids (lat, lon) for the dashboard's geographic module
REGION_COORDS = {
    "Arizona": (34.05, -111.09), "California": (36.78, -119.42), "Colorado": (39.55, -105.78), "Florida": (27.66, -81.52),
    "Georgia": (32.16, -82.90), "Kansas": (39.01, -98.48), "Nevada": (38.80, -116.42), "New York": (43.00, -75.00),
    "Ohio": (40.42, -82.91), "Oregon": (43.80, -120.55), "Texas": (31.97, -99.90), "Utah": (39.32, -111.09),
    "Virginia": (37.43, -78.66), "Washington": (47.75, -120.74), "Wyoming": (43.08, -107.29),
    "England": (52.36, -1.17), "Northern Ireland": (54.79, -6.49), "Scotland": (56.49, -4.20), "Wales": (52.13, -3.78),
    "Alberta": (53.93, -116.58), "British Columbia": (53.73, -127.65), "Manitoba": (53.76, -98.81),
    "Ontario": (51.25, -85.32), "Quebec": (52.94, -73.55),
    "Bavaria": (48.79, 11.50), "Berlin": (52.52, 13.40), "Hamburg": (53.55, 9.99), "Hesse": (50.65, 9.16),
    "North Rhine-Westphalia": (51.43, 7.66),
    "Brittany": (48.20, -2.93), "Ile-de-France": (48.85, 2.35), "Normandy": (49.18, -0.37), "Occitanie": (43.89, 3.28),
    "Provence": (43.94, 6.06),
    "Brussels": (50.85, 4.35), "Flanders": (51.05, 4.48), "Wallonia": (50.41, 4.44),
    "Baja California": (30.84, -115.28), "Jalisco": (20.66, -103.35), "Mexico City": (19.43, -99.13),
    "Nuevo Leon": (25.59, -99.99), "Puebla": (19.04, -98.21),
    "New South Wales": (-31.25, 146.92), "Queensland": (-20.92, 142.70), "South Australia": (-30.00, 136.21),
    "Victoria": (-36.85, 144.28), "Western Australia": (-25.04, 117.79),
    "Krasnodar Krai": (45.04, 38.98), "Moscow Oblast": (55.34, 38.29), "Novosibirsk": (55.03, 82.92),
    "Saint Petersburg": (59.93, 30.34), "Tatarstan": (55.18, 50.73),
    "Capital Region": (55.68, 12.57), "Central Denmark": (56.26, 9.50), "North Denmark": (57.05, 9.92),
    "Southern Denmark": (55.30, 9.00), "Zealand": (55.45, 11.79),
}
COUNTRY_ISO3 = {"USA": "USA", "UK": "GBR", "Canada": "CAN", "Germany": "DEU", "France": "FRA", "Belgium": "BEL",
                "Mexico": "MEX", "Australia": "AUS", "Russia": "RUS", "Denmark": "DNK"}


# --------------------------------------------------------------------------------------
# 1. LOAD + CLEAN
# --------------------------------------------------------------------------------------
def parse_mixed_dates(s: pd.Series) -> pd.Series:
    """Parse a column that mixes 'M/D/YYYY' (slash) and 'MM-DD-YYYY' (dash) strings.

    Slash dates are unambiguous (the day part reaches 31). Dash dates have both parts <= 12 in
    clients.csv, so the convention is inferred from properties.csv, where the same dash format only
    yields a sensible 24-month calendar when read as MM-DD-YYYY (see paper, Section 2.3).
    """
    s = s.astype(str).str.strip()
    slash = pd.to_datetime(s.where(s.str.contains("/")), format="%m/%d/%Y", errors="coerce")
    dash = pd.to_datetime(s.where(s.str.contains("-")), format="%m-%d-%Y", errors="coerce")
    return slash.fillna(dash)


def load_and_clean():
    log = {}
    clients = pd.read_csv(RAW / "clients.csv", dtype=str)
    props = pd.read_csv(RAW / "properties.csv", dtype=str)
    log["raw_clients"], log["raw_properties"] = len(clients), len(props)

    # ---- generic hygiene: strip whitespace, collapse internal spaces
    for df in (clients, props):
        for col in df.columns:
            df[col] = df[col].str.strip().str.replace(r"\s+", " ", regex=True)

    # ---- missing values (none in the supplied files, but the rules are kept so reruns on new data are safe)
    cat_cols = ["client_type", "gender", "country", "region", "acquisition_purpose", "loan_applied", "referral_channel"]
    log["missing_client_cells"] = int(clients[cat_cols + ["date_of_birth", "satisfaction_score"]].isna().sum().sum())
    for col in cat_cols:
        clients[col] = clients[col].fillna("Unknown")

    # ---- duplicates
    log["dup_client_ids"] = int(clients.client_id.duplicated().sum())
    log["dup_client_rows_ignoring_id"] = int(clients.drop(columns="client_id").duplicated().sum())
    clients = clients.drop_duplicates(subset="client_id")
    clients = clients.drop_duplicates(subset=[c for c in clients.columns if c != "client_id"])

    # ---- label normalisation (aligned with the PRD vocabulary)
    clients["client_type"] = clients.client_type.str.title().replace({"Company": "Corporate"})
    clients["acquisition_purpose"] = clients.acquisition_purpose.str.title().replace({"Home": "Personal Use"})
    clients["gender"] = clients.gender.str.upper().replace({"M": "Male", "F": "Female"})
    clients["loan_applied"] = clients.loan_applied.str.title()
    clients["referral_channel"] = clients.referral_channel.str.title()
    clients["satisfaction_score"] = pd.to_numeric(clients.satisfaction_score, errors="coerce")
    clients["satisfaction_score"] = clients.satisfaction_score.fillna(clients.satisfaction_score.median())

    # ---- date of birth -> age in years at the end of the observation window
    dob = parse_mixed_dates(clients.date_of_birth)
    log["dob_unparsed"] = int(dob.isna().sum())
    is_dash = clients.date_of_birth.str.contains("-")
    log["dob_slash_format"], log["dob_dash_format"] = int((~is_dash).sum()), int(is_dash.sum())
    clients["age"] = (AS_OF - dob).dt.days / 365.25
    clients["age"] = clients.age.fillna(clients.age.median())
    # sensitivity check: how much would ages move if dash dates were DD-MM-YYYY instead?
    alt = pd.to_datetime(clients.date_of_birth.where(is_dash), format="%d-%m-%Y", errors="coerce")
    alt_age = (AS_OF - dob.where(~is_dash, alt)).dt.days / 365.25
    diff = (clients.age - alt_age).abs()
    log["dob_ambiguity_mean_abs_age_diff"] = float(diff.mean())
    log["dob_ambiguity_max_abs_age_diff"] = float(diff.max())
    clients["age_band"] = pd.cut(clients.age, [0, 34.99, 44.99, 54.99, 64.99, 200],
                                 labels=["<35", "35-44", "45-54", "55-64", "65+"]).astype(str)
    # DOB and names are not needed downstream; dropping them keeps the exported tables free of direct identifiers
    clients = clients.drop(columns=["date_of_birth", "first_name", "last_name"])

    # ---- properties
    props["price"] = props.sale_price.str.replace(r"[$,]", "", regex=True).astype(float)
    props["floor_area_sqft"] = props.floor_area_sqft.astype(float)
    props["date"] = pd.to_datetime(props.transaction_date, format="%m-%d-%Y")
    props["month"] = props.date.dt.to_period("M").dt.to_timestamp()
    props["tower_number"] = props.tower_number.astype(int)
    props["price_per_sqft"] = props.price / props.floor_area_sqft
    props = props.rename(columns={"client_ref": "client_id"}).drop(columns=["sale_price", "transaction_date"])
    props["is_sold"] = props.listing_status.eq("Sold")
    log["dup_listing_ids"] = int(props.listing_id.duplicated().sum())
    log["sold_listings"], log["available_listings"] = int(props.is_sold.sum()), int((~props.is_sold).sum())
    log["orphan_client_refs"] = int((~props.loc[props.is_sold, "client_id"].isin(clients.client_id)).sum())
    log["sold_without_client"] = int((props.is_sold & props.client_id.isna()).sum())
    log["date_min"], log["date_max"] = str(props.date.min().date()), str(props.date.max().date())
    return clients.reset_index(drop=True), props, log


# --------------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------------------
def build_client_table(clients, props):
    sold = props[props.is_sold]
    g = sold.groupby("client_id").agg(
        n_units=("listing_id", "count"), total_spend=("price", "sum"), avg_price=("price", "mean"),
        avg_ppsf=("price_per_sqft", "mean"), avg_area=("floor_area_sqft", "mean"),
        share_office=("unit_category", lambda x: (x == "Office").mean()),
        n_towers=("tower_number", "nunique"), first_purchase=("date", "min"), last_purchase=("date", "max"))
    g["active_months"] = ((g.last_purchase - g.first_purchase).dt.days / 30.44).round(1)
    df = clients.merge(g, left_on="client_id", right_index=True, how="left")
    # collapse the 57 regions to 13 groups so that rare regions do not distort distances
    us_regions = df[df.country == "USA"].region.value_counts()
    keep = us_regions[us_regions >= 40].index
    df["region_group"] = np.where(df.region.isin(keep), df.region,
                                  np.where(df.country == "USA", "US - Other", "International"))
    df["is_usa"] = (df.country == "USA").astype(int)
    df["portfolio_buyer"] = (df.n_units >= 5).astype(int)   # >= 5 units: the heavy tail of the unit-count distribution
    df["inv"] = (df.acquisition_purpose == "Investment").astype(int)
    df["loan"] = (df.loan_applied == "Yes").astype(int)
    return df


# --------------------------------------------------------------------------------------
# 3. ENCODING + SCALING
# --------------------------------------------------------------------------------------
CAT_COLS = ["client_type", "acquisition_purpose", "loan_applied", "referral_channel", "country", "region_group"]
NUM_COLS = ["age", "satisfaction_score", "log_units", "log_avg_price", "share_office"]


def encode_and_scale(df):
    d = df.copy()
    d["log_units"], d["log_avg_price"] = np.log(d.n_units), np.log(d.avg_price)
    onehot = pd.get_dummies(d[CAT_COLS], dtype=float)                                                  # one-hot encoding
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(d[NUM_COLS]), columns=NUM_COLS, index=d.index)  # min-max scaling
    X = pd.concat([scaled, onehot], axis=1)
    # label (ordinal) codes: exported for BI / downstream tools; never used in distance calculations (the codes impose a false order)
    label_cols = ["client_type", "acquisition_purpose", "loan_applied", "referral_channel", "country", "region", "gender"]
    labels = pd.DataFrame({f"{c}_code": LabelEncoder().fit_transform(d[c]) for c in label_cols}, index=d.index)
    return X, labels


# --------------------------------------------------------------------------------------
# 4. MODEL SELECTION
# --------------------------------------------------------------------------------------
def bootstrap_stability(X, labels, k, rng):
    """Mean adjusted Rand index between the full-data solution and K-Means refits on random 80 % subsamples."""
    aris = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(X), int(BOOT_FRAC * len(X)), replace=False)
        lb = KMeans(k, n_init=10, random_state=int(rng.integers(1_000_000))).fit(X[idx]).labels_
        aris.append(adjusted_rand_score(labels[idx], lb))
    return float(np.mean(aris)), float(np.std(aris))


def elbow_point(ks, inertia):
    """Kneedle-style elbow: the k lying farthest below the chord that joins the two ends of the inertia curve."""
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(inertia) - inertia[-1]) / (inertia[0] - inertia[-1])
    return int(ks[int(np.argmax((1 - x) - y))])


def model_selection(X):
    Xv, rng, rows = X.values, np.random.default_rng(SEED), []
    for k in K_RANGE:
        km = KMeans(k, n_init=30, random_state=SEED).fit(Xv)
        ward = AgglomerativeClustering(k, linkage="ward").fit_predict(Xv)
        stab, stab_sd = bootstrap_stability(Xv, km.labels_, k, rng)
        rows.append(dict(
            k=k, inertia=km.inertia_, silhouette_kmeans=silhouette_score(Xv, km.labels_),
            silhouette_ward=silhouette_score(Xv, ward), davies_bouldin=davies_bouldin_score(Xv, km.labels_),
            calinski_harabasz=calinski_harabasz_score(Xv, km.labels_), stability_ari=stab, stability_sd=stab_sd,
            kmeans_vs_ward_ari=adjusted_rand_score(km.labels_, ward), smallest_cluster=int(np.bincount(km.labels_).min())))
    ms = pd.DataFrame(rows)
    # Pre-declared rule: rank each candidate k on five criteria and take the lowest mean rank (ties -> smaller k)
    cand = ms[ms.k.isin(K_CANDIDATES)].copy()
    ranks = pd.DataFrame({
        "silhouette": cand.silhouette_kmeans.rank(ascending=False), "davies_bouldin": cand.davies_bouldin.rank(ascending=True),
        "calinski_harabasz": cand.calinski_harabasz.rank(ascending=False), "stability": cand.stability_ari.rank(ascending=False),
        "ward_agreement": cand.kmeans_vs_ward_ari.rank(ascending=False)})
    cand["mean_rank"] = ranks.mean(axis=1).values
    ms = ms.merge(cand[["k", "mean_rank"]], on="k", how="left")
    best_k = int(cand.sort_values(["mean_rank", "k"]).iloc[0].k)
    return ms, best_k, elbow_point(list(ms.k), list(ms.inertia))


# --------------------------------------------------------------------------------------
# 5/6. FINAL MODEL + INTERPRETATION
# --------------------------------------------------------------------------------------
PALETTE = ["#1F4E79", "#E07B39", "#2A9D8F", "#8E5EA2", "#C0392B", "#7F8C8D", "#B8A000", "#3E7CB1"]


def name_segments(d):
    """Name every cluster from its measured profile so that names can never drift from the data."""
    prof = d.groupby("segment_raw").agg(usa=("is_usa", "mean"), inv=("inv", "mean"), loan=("loan", "mean"))
    names = {}
    for s, r in prof.iterrows():
        if r.usa < 0.5:
            names[s] = "International Buyers"
        elif r.inv > 0.5:
            names[s] = "Domestic Investors"
        elif r.loan > 0.5:
            names[s] = "Domestic Financed Home Buyers"
        else:
            names[s] = "Domestic Cash Home Buyers"
    # if k is changed and two clusters earn the same name, disambiguate with A/B suffixes
    counts, seen = pd.Series(list(names.values())).value_counts(), {}
    for s in sorted(names):
        if counts[names[s]] > 1:
            seen[names[s]] = seen.get(names[s], 0) + 1
            names[s] = f"{names[s]} ({chr(64 + seen[names[s]])})"
    return names


def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
    return float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))), float(p)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    clients, props, log = load_and_clean()
    df = build_client_table(clients, props)
    X, labels = encode_and_scale(df)
    Xv = X.values

    ms, best_k, elbow_k = model_selection(X)
    ms.to_csv(OUT / "model_selection.csv", index=False)

    km = KMeans(best_k, n_init=50, random_state=SEED).fit(Xv)
    ward = AgglomerativeClustering(best_k, linkage="ward").fit_predict(Xv)
    df["segment_raw"] = km.labels_
    names = name_segments(df)
    order = df.groupby("segment_raw").size().sort_values(ascending=False).index            # S1 = largest
    sid = {raw: f"S{i + 1}" for i, raw in enumerate(order)}
    df["segment_id"] = df.segment_raw.map(sid)
    df["segment_name"] = df.segment_raw.map(names)
    df["segment_label"] = df.segment_id + " - " + df.segment_name
    # Ward assignment, relabelled to the K-Means segment it overlaps most (used only for validation displays)
    xt = pd.crosstab(ward, df.segment_raw)
    ward_map = {w: xt.loc[w].idxmax() for w in xt.index}
    df["ward_segment_id"] = pd.Series(ward).map(ward_map).map(sid).values
    df["ward_agrees"] = (df.ward_segment_id == df.segment_id).astype(int)
    df["silhouette"] = silhouette_samples(Xv, km.labels_)

    pca = PCA(n_components=2, random_state=SEED).fit(Xv)
    df[["pc1", "pc2"]] = pca.transform(Xv)
    df = pd.concat([df, labels], axis=1)
    df["lat"] = df.region.map(lambda r: REGION_COORDS[r][0])
    df["lon"] = df.region.map(lambda r: REGION_COORDS[r][1])
    df["iso3"] = df.country.map(COUNTRY_ISO3)

    seg_order = [sid[r] for r in order]
    seg_names = {sid[r]: names[r] for r in order}

    # ---------- segment profiles
    prof = df.groupby("segment_id").agg(
        clients=("client_id", "size"), age_mean=("age", "mean"), age_median=("age", "median"),
        satisfaction=("satisfaction_score", "mean"), units_mean=("n_units", "mean"),
        spend_mean=("total_spend", "mean"), spend_median=("total_spend", "median"),
        avg_price=("avg_price", "mean"), avg_ppsf=("avg_ppsf", "mean"), office_share=("share_office", "mean"),
        pct_corporate=("client_type", lambda x: (x == "Corporate").mean()), pct_investment=("inv", "mean"),
        pct_loan=("loan", "mean"), pct_usa=("is_usa", "mean"), pct_california=("region", lambda x: (x == "California").mean()),
        pct_portfolio=("portfolio_buyer", "mean"), total_revenue=("total_spend", "sum"),
        mean_silhouette=("silhouette", "mean"), ward_agreement=("ward_agrees", "mean")).loc[seg_order]
    prof.insert(0, "segment_name", [seg_names[s] for s in prof.index])
    prof["client_share"] = prof.clients / prof.clients.sum()
    prof["revenue_share"] = prof.total_revenue / prof.total_revenue.sum()
    for ch in ["Website", "Agency", "Client"]:
        prof[f"pct_ref_{ch.lower()}"] = df.groupby("segment_id").referral_channel.apply(lambda x, ch=ch: (x == ch).mean()).loc[seg_order]
    prof.to_csv(OUT / "segment_profiles.csv")

    # ---------- which variables actually differ between segments?
    diff_rows = []
    for v in ["age", "satisfaction_score", "n_units", "total_spend", "avg_price", "share_office"]:
        h, p = stats.kruskal(*[g[v].values for _, g in df.groupby("segment_id")])
        eps2 = float((h - best_k + 1) / (len(df) - best_k))         # epsilon-squared effect size
        diff_rows.append(dict(variable=v, kind="numeric", test="Kruskal-Wallis", p_value=p, effect_size=max(eps2, 0.0), effect_name="epsilon-squared"))
    for v in ["client_type", "acquisition_purpose", "loan_applied", "referral_channel", "country", "gender"]:
        cv, p = cramers_v(df[v], df.segment_id)
        diff_rows.append(dict(variable=v, kind="categorical", test="Chi-square", p_value=p, effect_size=cv, effect_name="Cramer's V"))
    seg_diff = pd.DataFrame(diff_rows)
    seg_diff.to_csv(OUT / "segment_differences.csv", index=False)

    # ---------- surrogate models: what defines a segment?
    feat = pd.DataFrame({
        "is_usa": df.is_usa, "loan_applied": df.loan, "investment_purpose": df.inv,
        "corporate": (df.client_type == "Corporate").astype(int), "age": df.age, "satisfaction": df.satisfaction_score,
        "units": df.n_units, "avg_price": df.avg_price, "office_share": df.share_office,
        "california": (df.region == "California").astype(int), "ref_website": (df.referral_channel == "Website").astype(int),
        "ref_agency": (df.referral_channel == "Agency").astype(int), "ref_client": (df.referral_channel == "Client").astype(int)})
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(feat, df.segment_id)
    tree_acc = float(tree.score(feat, df.segment_id))
    tree_cv = float(cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=SEED), feat, df.segment_id, cv=5).mean())
    rules = export_text(tree, feature_names=list(feat.columns))
    rf = RandomForestClassifier(300, random_state=SEED, n_jobs=-1).fit(feat, df.segment_id)
    pi = permutation_importance(rf, feat, df.segment_id, n_repeats=10, random_state=SEED, n_jobs=-1)
    importance = pd.DataFrame({"feature": feat.columns, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    importance.to_csv(OUT / "feature_importance.csv", index=False)

    # ---------- testing the PRD's four hypothesised personas
    hyp = []
    inv_intl, inv_us = df[df.is_usa == 0].inv.mean(), df[df.is_usa == 1].inv.mean()
    v, p = cramers_v(df.is_usa, df.inv)
    hyp.append(dict(persona="C1 Global Investors", claim="International buyers skew towards investment purchases",
                    test="Chi-square: country group x purpose",
                    observed=f"Investment share: international {inv_intl:.1%} vs domestic {inv_us:.1%}",
                    p_value=p, effect=v, effect_name="Cramer's V"))
    a_loan, a_noloan = df[df.loan == 1].age, df[df.loan == 0].age
    u, p = stats.mannwhitneyu(a_loan, a_noloan)
    under35 = df[df.age < 35]
    hyp.append(dict(persona="C2 First-Time Buyers", claim="Younger buyers are more loan-dependent",
                    test="Mann-Whitney: age by loan status",
                    observed=f"Mean age: loan {a_loan.mean():.1f} vs no loan {a_noloan.mean():.1f}; loan rate under 35: {under35.loan.mean():.1%} vs {df.loan.mean():.1%} overall",
                    p_value=p, effect=float(1 - 2 * u / (len(a_loan) * len(a_noloan))), effect_name="rank-biserial r"))
    corp, ind = df[df.client_type == "Corporate"].n_units, df[df.client_type == "Individual"].n_units
    u, p = stats.mannwhitneyu(corp, ind)
    hyp.append(dict(persona="C3 Corporate Buyers", claim="Corporate clients purchase more units",
                    test="Mann-Whitney: units by client type",
                    observed=f"Mean units: corporate {corp.mean():.2f} vs individual {ind.mean():.2f}; 5+ units: {(corp >= 5).mean():.1%} vs {(ind >= 5).mean():.1%}",
                    p_value=p, effect=float(1 - 2 * u / (len(corp) * len(ind))), effect_name="rank-biserial r"))
    rho, p = stats.spearmanr(df.satisfaction_score, df.total_spend)
    rho2, _ = stats.spearmanr(df.satisfaction_score, df.avg_price)
    hyp.append(dict(persona="C4 Luxury Investors", claim="High satisfaction goes with large investments",
                    test="Spearman: satisfaction vs total spend",
                    observed=f"rho = {rho:.3f} (total spend), {rho2:.3f} (average unit price)",
                    p_value=p, effect=float(rho), effect_name="Spearman rho"))
    hyp = pd.DataFrame(hyp)
    hyp.to_csv(OUT / "hypothesis_tests.csv", index=False)

    # ---------- transactions tagged with the buyer's segment (behaviour + market pages)
    tx = props.merge(df[["client_id", "segment_id", "segment_name", "segment_label", "country", "region",
                         "client_type", "acquisition_purpose", "loan_applied"]], on="client_id", how="left")
    tx.drop(columns=["date"]).to_csv(OUT / "transactions.csv", index=False)

    # ---------- market context + concentration
    sold = props[props.is_sold]
    spend = df.total_spend.sort_values(ascending=False)
    monthly = props.groupby("month").agg(listed=("listing_id", "count"), sold=("is_sold", "sum")).join(
        sold.groupby("month").agg(revenue=("price", "sum"), ppsf=("price_per_sqft", "mean")))
    monthly.to_csv(OUT / "monthly_market.csv")
    y24, y25 = monthly.loc["2024"], monthly.loc["2025"]
    tower = props.groupby("tower_number").agg(listings=("listing_id", "count"), sell_through=("is_sold", "mean"),
                                              avg_price=("price", "mean"), ppsf=("price_per_sqft", "mean"))
    tower.to_csv(OUT / "tower_summary.csv")
    rho_t, p_t = stats.spearmanr(sold.date.astype("int64"), sold.price_per_sqft)

    df_out = df.drop(columns=["segment_raw"]).rename(columns={"inv": "investment_flag", "loan": "loan_flag"})
    df_out.to_csv(OUT / "clients_segmented.csv", index=False)
    X.assign(client_id=df.client_id.values).to_csv(OUT / "model_matrix.csv", index=False)
    np.save(OUT / "ward_linkage.npy", linkage(Xv, method="ward"))

    seg_meta = {s: dict(name=seg_names[s], label=f"{s} - {seg_names[s]}", color=PALETTE[i]) for i, s in enumerate(seg_order)}
    json.dump(seg_meta, open(OUT / "segments.json", "w"), indent=2)

    row = ms.loc[ms.k == best_k].iloc[0]
    results = dict(
        log=log, best_k=best_k, elbow_k=elbow_k, n_features=int(X.shape[1]), n_clients=int(len(df)),
        silhouette=float(row.silhouette_kmeans), silhouette_ward=float(row.silhouette_ward),
        davies_bouldin=float(row.davies_bouldin), calinski_harabasz=float(row.calinski_harabasz),
        stability=float(row.stability_ari), ward_ari=float(row.kmeans_vs_ward_ari), ward_agreement_pct=float(df.ward_agrees.mean()),
        tree_acc=tree_acc, tree_cv=tree_cv, tree_rules=rules, pca_var=[float(v) for v in pca.explained_variance_ratio_],
        segment_meta=seg_meta,
        top10_share=float(spend.head(int(.1 * len(spend))).sum() / spend.sum()),
        top20_share=float(spend.head(int(.2 * len(spend))).sum() / spend.sum()),
        portfolio_clients=int(df.portfolio_buyer.sum()),
        portfolio_rev_share=float(df[df.portfolio_buyer == 1].total_spend.sum() / spend.sum()),
        total_revenue=float(sold.price.sum()), sold_units=int(len(sold)),
        listings_2024_per_month=float(y24.listed.mean()), listings_2025_per_month=float(y25.listed.mean()),
        sold_2024_per_month=float(y24.sold.mean()), sold_2025_per_month=float(y25.sold.mean()),
        sell_through=float(props.is_sold.mean()), ppsf_mean=float(sold.price_per_sqft.mean()), ppsf_sd=float(sold.price_per_sqft.std()),
        ppsf_office=float(sold[sold.unit_category == "Office"].price_per_sqft.mean()),
        ppsf_apt=float(sold[sold.unit_category == "Apartment"].price_per_sqft.mean()),
        ppsf_time_rho=float(rho_t), ppsf_time_p=float(p_t),
        area_price_r=float(np.corrcoef(sold.floor_area_sqft, sold.price)[0, 1]))
    json.dump(results, open(OUT / "results.json", "w"), indent=2, default=str)
    return df, X, ms, prof, seg_diff, hyp, importance, results


if __name__ == "__main__":
    df, X, ms, prof, seg_diff, hyp, importance, res = run()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\nDATA LOG:", json.dumps(res["log"], indent=1))
    print("\nMODEL SELECTION\n", ms.round(3).to_string(index=False))
    print(f"\nSelected k = {res['best_k']}   (elbow suggests {res['elbow_k']})   model-matrix columns = {res['n_features']}")
    print("\nSEGMENT PROFILES\n", prof.round(3).T.to_string())
    print("\nWHICH VARIABLES DIFFER ACROSS SEGMENTS\n", seg_diff.round(4).to_string(index=False))
    print("\nHYPOTHESES\n", hyp.round(4).to_string(index=False))
    print("\nIMPORTANCE\n", importance.round(4).head(8).to_string(index=False))
    print(f"\nSurrogate tree accuracy {res['tree_acc']:.3f}  (5-fold CV {res['tree_cv']:.3f})\n{res['tree_rules']}")
