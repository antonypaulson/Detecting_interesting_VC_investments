"""What is in ``data/`` and what is not.

## Shipped (used to reproduce the 2019 scoring)

| File | Rows | Role |
| --- | --- | --- |
| `analytics.csv` | 68,830 | Intermediate app table after EDA cleaning (no daily-growth columns). |
| `frame_scoring_model.csv` | 68,830 | Scoring-ready table: ranks, subscription flag, age, daily growth, max growth rate. |

`python -m vc_investments` and `TCG_scoring_model.ipynb` read `frame_scoring_model.csv`.

The scoring CSV was exported with an unnamed index column. pandas uses that extra field as the index; do not shift the headers by hand unless you also drop the leading index values.

## Not shipped (too large / originally only on Google Drive)

The data-prep notebook was written against two pickle files from a Colab Drive folder `TCG/`:

- `app_info_df.pkl` — App Store listing attributes (~68k apps).
- `app_rank_df.pkl` — daily rank panel for the prior three years.

Those files are **not** in git. They were never committed (Drive paths only). This project does not re-scrape the App Store.

If you still have the original extracts, put them here:

```text
data/raw/app_info_df.pkl
data/raw/app_rank_df.pkl
```

Then run `TCG_Data_preparation.ipynb`. It writes `data/final_analysis_frame.csv`, which `EDA_and_Machine_Learning_TCG.ipynb` prefers when present.

Without the pickles you can still:

1. Score apps with the shipped CSVs.
2. Run the consistency models on `frame_scoring_model.csv` (the EDA notebook falls back to that file). The original `is_editor_choice` column is not in the shipped extract, so that one feature is omitted.
