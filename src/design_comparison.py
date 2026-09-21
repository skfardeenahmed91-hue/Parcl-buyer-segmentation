"""
Compares alternative feature-scaling designs for the clustering step (paper, Appendix C).
Run after pipeline.py:   python src/design_comparison.py
Writes data/processed/design_comparison.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pipeline as P

warnings.filterwarnings("ignore")
SEED, N_BOOT = 42, 30

clients, props, _ = P.load_and_clean()
df = P.build_client_table(clients, props)
df["log_units"], df["log_avg_price"] = np.log(df.n_units), np.log(df.avg_price)
OH = pd.get_dummies(df[P.CAT_COLS], dtype=float)


def block_equal_variance(col):
    """One-hot block rescaled so that the whole block has total variance 1 (same as one standardised numeric)."""
    B = pd.get_dummies(df[col], dtype=float)
    if B.shape[1] == 2:
        B = B.iloc[:, [1]]
    return ((B - B.mean()) / np.sqrt(B.var(ddof=0).sum())).values


NUM5, NUM2 = P.NUM_COLS, ["age", "satisfaction_score"]
DESIGNS = {
    "A. PRD-literal: age + satisfaction (MinMax) + one-hot": np.hstack([MinMaxScaler().fit_transform(df[NUM2]), OH.values]),
    "B. Chosen: 5 numeric features (MinMax) + one-hot": np.hstack([MinMaxScaler().fit_transform(df[NUM5]), OH.values]),
    "C. 5 numeric features (Standard) + one-hot": np.hstack([StandardScaler().fit_transform(df[NUM5]), OH.values]),
    "D. Everything standardised, one-hot included": StandardScaler().fit_transform(np.hstack([df[NUM5].values, OH.values])),
    "E. Equal-variance blocks": np.hstack([StandardScaler().fit_transform(df[NUM5])] + [block_equal_variance(c) for c in P.CAT_COLS]),
}


def stability(X, labels, k, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for b in range(N_BOOT):
        idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        out.append(adjusted_rand_score(labels[idx], KMeans(k, n_init=10, random_state=b).fit(X[idx]).labels_))
    return float(np.mean(out))


rows = []
for name, X in DESIGNS.items():
    sil, small = {}, {}
    for k in range(2, 11):
        lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(X)
        sil[k], small[k] = silhouette_score(X, lab), int(np.bincount(lab).min())
    kb = max(sil, key=sil.get)
    lab4 = KMeans(4, n_init=30, random_state=SEED).fit_predict(X)
    ward4 = AgglomerativeClustering(4, linkage="ward").fit_predict(X)
    rows.append(dict(design=name, silhouette_k4=sil[4], stability_k4=stability(X, lab4, 4), ward_ari_k4=adjusted_rand_score(lab4, ward4),
                     smallest_k4=small[4], best_k=kb, silhouette_best=sil[kb], smallest_best=small[kb]))
    print(name, "done")

out = pd.DataFrame(rows)
out.to_csv(P.OUT / "design_comparison.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string(index=False))
