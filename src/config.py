from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLOTS_DIR = PROJECT_ROOT / "resources" / "plots"
DATA_DIR = PROJECT_ROOT / "resources" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"