
import logging
import os
from pathlib import Path

import joblib
from tensorflow.keras.callbacks import EarlyStopping

from src.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    MODEL_SAVE_PATH,
    RESULTS_DIR,
    SCALER_SAVE_PATH,
    VALIDATION_SPLIT,
    PROJECT_ROOT,
)
from src.data_loader import download_dataset, load_and_clean, split_and_scale
from src.evaluate import full_evaluation
from src.model import LRTracker, build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train() -> dict:

    logger.info("=" * 60)
    logger.info("STAGE 1 - Data Loading & Preprocessing")
    logger.info("=" * 60)

    download_dataset()
    df = load_and_clean()
    data = split_and_scale(df)

    logger.info("=" * 60)
    logger.info("STAGE 2 - Model Construction")
    logger.info("=" * 60)

    input_dim = data["X_train_scaled"].shape[1]
    model = build_model(input_dim)
    model.summary(print_fn=logger.info)

    logger.info("=" * 60)
    logger.info("STAGE 3 - Training")
    logger.info("=" * 60)

    lr_tracker = LRTracker()
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        data["X_train_scaled"],
        data["y_train"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=data["class_weights"],
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop, lr_tracker],
        verbose=1,
    )

    logger.info("=" * 60)
    logger.info("STAGE 4 - Saving Artifacts")
    logger.info("=" * 60)

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_SAVE_PATH))
    logger.info("Model saved -> %s", MODEL_SAVE_PATH)

    joblib.dump(data["scaler"], str(SCALER_SAVE_PATH))
    logger.info("Scaler saved -> %s", SCALER_SAVE_PATH)

    logger.info("=" * 60)
    logger.info("STAGE 5 - Evaluation")
    logger.info("=" * 60)

    metrics = full_evaluation(
        model=model,
        X_test=data["X_test_scaled"],
        y_test=data["y_test"],
        feature_names=data["feature_names"],
        history=history,
        lr_tracker=lr_tracker,
        results_dir=RESULTS_DIR,
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)

    return {"model": model, "history": history, "data": data, "metrics": metrics}


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    train()
