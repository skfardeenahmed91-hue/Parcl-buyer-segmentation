# Buyer Segmentation and Investment Profiling

Machine-learning buyer segmentation for **Parcl Co. Limited** (project with Unified Mentor).
It clusters 2,000 clients using their profile and purchase behaviour, validates the result, tests the buyer personas suggested in the project brief, and serves everything through an interactive Streamlit dashboard.

## What is in this folder

| Item | Purpose |
|---|---|
| `paper/Buyer_Segmentation_Research_Paper.docx` | The research paper: EDA, method, results, insights, recommendations, limitations (19 pages) |
| `app.py` | The Streamlit dashboard (the four modules in the brief, plus Model & Methodology and Data Explorer tabs) |
| `src/pipeline.py` | The full analysis: cleaning, features, encoding, scaling, K-Means and Ward clustering, validation, exports |
| `src/make_figures.py` | Regenerates the 12 figures used in the paper |
| `src/design_comparison.py` | Compares five feature-scaling designs (paper, Appendix C) |
| `src/export_paper_data.py`, `paper/build_paper.js` | Build the Word document from the analysis outputs |
| `data/raw/` | The two source files (contain names and dates of birth: keep private) |
| `data/processed/` | Cleaned, segmented outputs used by the dashboard (no direct identifiers) |

## Quick start (dashboard)

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501. The Geographic Analysis map loads its base layer from the Plotly CDN, so it needs an internet connection; every other view works offline.

## Re-running the analysis

```bash
python src/pipeline.py             # writes data/processed/*
python src/make_figures.py         # writes figures/*.png
python src/design_comparison.py    # writes data/processed/design_comparison.csv
python src/export_paper_data.py    # writes data/processed/paper_data.json
npm install docx && node paper/build_paper.js   # rebuilds the Word paper
```

All random choices use seed 42, so reruns give identical results.

## Results at a glance

Four segments were selected by a rule fixed in advance (mean rank over five criteria for k = 3 to 8):

| ID | Segment | Clients | Share | US resident | Investment | Loan |
|---|---|---|---|---|---|---|
| S1 | Domestic Cash Home Buyers | 681 | 34.1% | 100% | 0% | 0% |
| S2 | Domestic Investors | 470 | 23.5% | 100% | 100% | 38% |
| S3 | International Buyers | 462 | 23.1% | 0% | 31% | 37% |
| S4 | Domestic Financed Home Buyers | 387 | 19.4% | 100% | 0% | 100% |

* **Quality is modest and reported as such.** Silhouette 0.16 (below the 0.26 that marks even weak structure), bootstrap stability 0.91, agreement with Ward hierarchical clustering ARI 0.79.
* **The segments are defined by three attributes.** A depth-3 decision tree reproduces every client's segment from *US residence*, *investment purpose* and *loan applied* (100% accuracy). Age, satisfaction, units bought and spend barely differ between segments.
* **Client attributes are almost independent** (median pairwise association 0.01), which is why cluster separation is weak. Confirm with the data owner how the register was compiled.
* **None of the four personas in the brief is supported** (Global Investors, First-Time Buyers, Corporate Buyers, Luxury Investors). Every effect size is below 0.05. The paper lists the extra data (income, prior ownership, ...) needed to test them properly.

## Notes

* Labels were normalised to the brief's vocabulary: source "Company" is shown as "Corporate" and "Home" as "Personal Use".
* Dates of birth mix two formats; the hyphenated format was read as month-day-year (inferred from `properties.csv`). Recomputing under the other reading changes ages by 0.14 years on average, so it does not matter.
* Before publishing this project (for example on GitHub or Streamlit Community Cloud), keep `data/raw/` out of the repository. `.gitignore` already excludes it and the dashboard does not need it.
* Tested with Python 3.12 on two library stacks (pandas 2.2 / scikit-learn 1.5 / Streamlit 1.40, and pandas 3.0 / scikit-learn 1.8 / Streamlit 1.64) with identical results.
* Fill in `[Author name]` on the first page of the paper before submitting it.
