import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


_cfg = load_config()

DATA_RAW_PATH = PROJECT_ROOT / _cfg["data"]["raw_path"]
DATA_DOWNLOAD_URL = _cfg["data"]["download_url"]
COLUMN_NAMES: list[str] = _cfg["data"]["column_names"]
TEST_SIZE: float = _cfg["data"]["test_size"]
RANDOM_STATE: int = _cfg["data"]["random_state"]

HIDDEN_LAYERS: list[int] = _cfg["model"]["architecture"]["hidden_layers"]
DROPOUT_RATE: float = _cfg["model"]["architecture"]["dropout_rate"]
ACTIVATION: str = _cfg["model"]["architecture"]["activation"]
OUTPUT_ACTIVATION: str = _cfg["model"]["architecture"]["output_activation"]
MAX_NORM_VALUE: float = _cfg["model"]["architecture"]["kernel_constraint_max_norm"]
INITIALIZER: str = _cfg["model"]["architecture"]["initializer"]

EPOCHS: int = _cfg["model"]["training"]["epochs"]
BATCH_SIZE: int = _cfg["model"]["training"]["batch_size"]
EARLY_STOPPING_PATIENCE: int = _cfg["model"]["training"]["early_stopping_patience"]
VALIDATION_SPLIT: float = _cfg["model"]["training"]["validation_split"]

OPTIMIZER_NAME: str = _cfg["model"]["optimizer"]["name"]
INITIAL_LR: float = _cfg["model"]["optimizer"]["initial_learning_rate"]
LR_DECAY_STEPS: int = _cfg["model"]["optimizer"]["lr_decay_steps"]
LR_DECAY_RATE: float = _cfg["model"]["optimizer"]["lr_decay_rate"]

MC_DROPOUT_SAMPLES: int = _cfg["model"]["inference"]["mc_dropout_samples"]
UNCERTAINTY_THRESHOLD: float = _cfg["model"]["inference"]["uncertainty_threshold"]

MODEL_SAVE_PATH = PROJECT_ROOT / _cfg["paths"]["model_save"]
SCALER_SAVE_PATH = PROJECT_ROOT / _cfg["paths"]["scaler_save"]
RESULTS_DIR = PROJECT_ROOT / _cfg["paths"]["results_dir"]

API_HOST: str = _cfg["api"]["host"]
API_PORT: int = _cfg["api"]["port"]
