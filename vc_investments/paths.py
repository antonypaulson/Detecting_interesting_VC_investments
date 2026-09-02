"""Repository paths. Notebooks and scripts should load data from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs"

SCORING_CSV = DATA_DIR / "frame_scoring_model.csv"
ANALYTICS_CSV = DATA_DIR / "analytics.csv"
RAW_MERGE_CSV = DATA_DIR / "final_analysis_frame.csv"

# Original Colab pickle extracts (not shipped; see data/README.md).
RAW_APP_INFO = RAW_DIR / "app_info_df.pkl"
RAW_APP_RANK = RAW_DIR / "app_rank_df.pkl"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
