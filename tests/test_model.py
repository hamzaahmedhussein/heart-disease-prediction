import numpy as np
import pytest
import tensorflow as tf

from src.model import MCDropout, build_model, mc_predict


class TestMCDropout:

    def test_dropout_active_during_inference(self):
        layer = MCDropout(rate=0.5)
        x = tf.ones((100, 10))

        outputs = [layer(x).numpy().sum() for _ in range(5)]
        assert len(set(outputs)) > 1, "MCDropout should produce stochastic outputs"

    def test_standard_dropout_inactive_during_inference(self):
        layer = tf.keras.layers.Dropout(rate=0.5)
        x = tf.ones((100, 10))

        outputs = [layer(x, training=False).numpy().sum() for _ in range(5)]
        assert len(set(outputs)) == 1, "Standard Dropout should be deterministic at inference"


class TestBuildModel:

    @pytest.fixture
    def model(self):
        return build_model(input_dim=13)

    def test_model_is_keras_model(self, model):
        assert isinstance(model, tf.keras.Model)

    def test_input_shape(self, model):
        assert model.input_shape == (None, 13)

    def test_output_shape(self, model):
        assert model.output_shape == (None, 1)

    def test_output_is_probability(self, model):
        dummy = np.random.randn(5, 13).astype(np.float32)
        preds = model.predict(dummy, verbose=0)
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_model_has_mc_dropout_layers(self, model):
        mc_layers = [l for l in model.layers if isinstance(l, MCDropout)]
        assert len(mc_layers) >= 1, "Model should contain at least one MCDropout layer"

    def test_model_has_batch_norm(self, model):
        bn_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
        assert len(bn_layers) >= 1, "Model should contain BatchNormalization"

    def test_custom_hidden_layers(self):
        model = build_model(input_dim=5, hidden_layers=[32, 16, 8])
        dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
        assert len(dense_layers) == 4


class TestMCPredict:

    @pytest.fixture
    def model_and_data(self):
        model = build_model(input_dim=13)
        X = np.random.randn(3, 13).astype(np.float32)
        return model, X

    def test_returns_mean_and_std(self, model_and_data):
        model, X = model_and_data
        mean, std = mc_predict(model, X, n_samples=10)
        assert mean.shape == (3, 1)
        assert std.shape == (3, 1)

    def test_mean_is_probability(self, model_and_data):
        model, X = model_and_data
        mean, _ = mc_predict(model, X, n_samples=10)
        assert mean.min() >= 0.0
        assert mean.max() <= 1.0

    def test_std_is_non_negative(self, model_and_data):
        model, X = model_and_data
        _, std = mc_predict(model, X, n_samples=10)
        assert (std >= 0).all()
