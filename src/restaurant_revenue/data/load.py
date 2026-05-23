import shutil
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "raw"


def download_dataset() -> None:
    import kagglehub

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download("gokulstark/restaurant-revenue-prediction")
    for f in Path(path).glob("*.csv"):
        shutil.copy(f, DATA_DIR / f.name)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (DATA_DIR / "train.csv").exists():
        download_dataset()

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test
