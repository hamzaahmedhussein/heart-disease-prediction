import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from src.config import (
    COLUMN_NAMES,
    DATA_DOWNLOAD_URL,
    DATA_RAW_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def download_dataset(dest: Optional[Path] = None) -> Path:
    dest = dest or DATA_RAW_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("Dataset already exists at %s", dest)
        return dest

    logger.info("Downloading dataset from %s", DATA_DOWNLOAD_URL)
    df = pd.read_csv(
        DATA_DOWNLOAD_URL, header=None, names=COLUMN_NAMES, na_values="?"
    )
    df.to_csv(dest, index=False)
    logger.info("Saved raw dataset to %s (%d rows)", dest, len(df))
    return dest


def load_and_clean(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or DATA_RAW_PATH
    if not path.exists():
        path = download_dataset(path)

    df = pd.read_csv(path)
    logger.info("Loaded dataset: %d rows x %d cols", *df.shape)

    for col in ("ca", "thal"):
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("Filled %d missing values in '%s' with mode (%.1f)", n_missing, col, mode_val)

    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    logger.info("Target distribution:\n%s", df["target"].value_counts().to_string())

    return df


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    X = df.drop("target", axis=1)
    y = df["target"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Split: %d train / %d test (stratified)", len(X_train), len(X_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    cw = dict(enumerate(weights))
    logger.info("Class weights: %s", cw)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "class_weights": cw,
        "feature_names": feature_names,
    }
