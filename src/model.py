import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model

from src.config import (
    ACTIVATION,
    DROPOUT_RATE,
    HIDDEN_LAYERS,
    INITIAL_LR,
    LR_DECAY_RATE,
    LR_DECAY_STEPS,
    MAX_NORM_VALUE,
    MC_DROPOUT_SAMPLES,
    OUTPUT_ACTIVATION,
)

logger = logging.getLogger(__name__)


class MCDropout(Dropout):

    def call(self, inputs):
        return super().call(inputs, training=True)


class LRTracker(Callback):

    def on_train_begin(self, logs=None):
        self.lrs: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        step = tf.cast(self.model.optimizer.iterations, tf.float32)
        current_lr = float(lr(step).numpy() if callable(lr) else lr.numpy())
        self.lrs.append(current_lr)


def build_model(
    input_dim: int,
    hidden_layers: Optional[list[int]] = None,
    dropout_rate: float = DROPOUT_RATE,
    activation: str = ACTIVATION,
    max_norm_value: float = MAX_NORM_VALUE,
    initial_lr: float = INITIAL_LR,
    lr_decay_steps: int = LR_DECAY_STEPS,
    lr_decay_rate: float = LR_DECAY_RATE,
) -> Model:

    layers = hidden_layers or HIDDEN_LAYERS
    init = tf.keras.initializers.HeNormal()

    inputs = Input(shape=(input_dim,), name="patient_features")
    x = inputs

    for i, units in enumerate(layers):
        x = Dense(
            units,
            kernel_initializer=init,
            kernel_constraint=max_norm(max_norm_value),
            name=f"dense_{i}",
        )(x)
        x = BatchNormalization(name=f"bn_{i}")(x)
        x = Activation(activation, name=f"{activation}_{i}")(x)
        x = MCDropout(dropout_rate, name=f"mc_dropout_{i}")(x)

    outputs = Dense(1, activation=OUTPUT_ACTIVATION, name="prediction")(x)
    model = Model(inputs=inputs, outputs=outputs, name="HeartDisease_MLP")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=lr_decay_steps,
        decay_rate=lr_decay_rate,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Built model with %d layers, input_dim=%d", len(model.layers), input_dim)
    return model


def mc_predict(
    model: Model,
    X: np.ndarray,
    n_samples: int = MC_DROPOUT_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:

    X_repeated = np.repeat(X, n_samples, axis=0)
    all_preds = model.predict(X_repeated, verbose=0, batch_size=256)
    all_preds = all_preds.reshape(len(X), n_samples, -1)

    mean_preds = all_preds.mean(axis=1)
    std_preds = all_preds.std(axis=1)
    return mean_preds, std_preds
