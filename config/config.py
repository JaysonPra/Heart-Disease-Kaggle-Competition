from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
EXPERIMENTATION_CONFIG_DIR = ROOT_DIR / "config" / "experimentation"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"