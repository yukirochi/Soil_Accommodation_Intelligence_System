"""
Tests for the ferti fertilization classifier.

Dataset: soil_fertilization.csv (880 rows, 841 after filtering Output=2 rows)
Features: N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B
Target:   Output (0 = no fertilization needed, 1 = fertilization needed)

Run: pytest test_ferti.py -v
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib

from soil_fertility_model import ferti

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

CSV_PATH = "soil_fertilization.csv"
FEATURE_COLS = ["N", "P", "K", "pH", "EC", "OC", "S", "Zn", "Fe", "Cu", "Mn", "B"]

# One representative sample from each class (real values from dataset)
SAMPLE_CLASS_0 = pd.DataFrame([{
    "N": 138, "P": 8.6, "K": 560, "pH": 7.46, "EC": 0.62,
    "OC": 0.70, "S": 5.9, "Zn": 0.24, "Fe": 0.31, "Cu": 0.77,
    "Mn": 8.71, "B": 0.11
}])

SAMPLE_CLASS_1 = pd.DataFrame([{
    "N": 270, "P": 9.9, "K": 444, "pH": 7.63, "EC": 0.40,
    "OC": 0.86, "S": 11.8, "Zn": 0.25, "Fe": 0.76, "Cu": 1.69,
    "Mn": 2.43, "B": 2.26
}])

MULTI_SAMPLE = pd.DataFrame([
    {"N": 138, "P": 8.6,  "K": 560, "pH": 7.46, "EC": 0.62, "OC": 0.70, "S":  5.9, "Zn": 0.24, "Fe": 0.31, "Cu": 0.77, "Mn": 8.71, "B": 0.11},
    {"N": 270, "P": 9.9,  "K": 444, "pH": 7.63, "EC": 0.40, "OC": 0.86, "S": 11.8, "Zn": 0.25, "Fe": 0.76, "Cu": 1.69, "Mn": 2.43, "B": 2.26},
    {"N": 163, "P": 9.6,  "K": 718, "pH": 7.59, "EC": 0.51, "OC": 1.11, "S": 14.3, "Zn": 0.30, "Fe": 0.86, "Cu": 1.57, "Mn": 2.70, "B": 2.03},
    {"N": 213, "P": 7.5,  "K": 338, "pH": 7.62, "EC": 0.75, "OC": 1.06, "S": 25.4, "Zn": 0.30, "Fe": 0.86, "Cu": 1.54, "Mn": 2.89, "B": 2.29},
    {"N":  50, "P": 3.0,  "K":  80, "pH": 6.50, "EC": 0.20, "OC": 0.50, "S":  2.0, "Zn": 0.10, "Fe": 0.25, "Cu": 0.15, "Mn": 0.50, "B": 0.10},
])


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """One trained ferti instance reused across the module."""
    pkl = str(tmp_path_factory.mktemp("model") / "ferti_model.pkl")
    return ferti(save_path=pkl, data_path=CSV_PATH)


@pytest.fixture()
def fresh_model(tmp_path):
    """Fresh ferti instance per test, isolated save path."""
    pkl = str(tmp_path / "ferti_model.pkl")
    return ferti(save_path=pkl, data_path=CSV_PATH)


# ─────────────────────────────────────────────────────────────
# 1. Initialisation
# ─────────────────────────────────────────────────────────────

class TestInit:
    def test_instantiation_succeeds(self, model):
        assert model is not None

    def test_model_attribute_exists(self, model):
        assert hasattr(model, "model")
        assert model.model is not None

    def test_scaler_attribute_exists(self, model):
        assert hasattr(model, "scaler")
        assert model.scaler is not None

    def test_scaler_is_fitted(self, model):
        # A fitted StandardScaler has mean_ set
        assert hasattr(model.scaler, "mean_"), "Scaler was not fitted"
        assert len(model.scaler.mean_) == len(FEATURE_COLS)

    def test_pkl_file_is_created_on_first_init(self, fresh_model):
        assert os.path.exists(fresh_model.save_path), "pkl file not created after first init"


# ─────────────────────────────────────────────────────────────
# 2. Data filtering (Output=2 rows must be excluded)
# ─────────────────────────────────────────────────────────────

class TestDataFiltering:
    def test_raw_csv_has_output_2_rows(self):
        df = pd.read_csv(CSV_PATH)
        assert (df["Output"] == 2).sum() == 39, "Expected 39 Output=2 rows in raw CSV"

    def test_model_trained_on_binary_labels_only(self, model):
        # RandomForest should have exactly 2 classes
        classes = model.model.classes_
        assert set(classes) == {0, 1}, f"Model classes should be {{0,1}}, got {classes}"


# ─────────────────────────────────────────────────────────────
# 3. Predict — output shape & value constraints
# ─────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_single_row_returns_one_value(self, model):
        result = model.predict(SAMPLE_CLASS_0)
        assert len(result) == 1

    def test_predict_multi_row_shape(self, model):
        result = model.predict(MULTI_SAMPLE)
        assert len(result) == len(MULTI_SAMPLE)

    def test_predict_returns_only_binary_values(self, model):
        result = model.predict(MULTI_SAMPLE)
        assert set(result).issubset({0, 1}), f"Non-binary predictions: {set(result)}"

    def test_predict_returns_numpy_array(self, model):
        result = model.predict(SAMPLE_CLASS_0)
        assert isinstance(result, np.ndarray)

    def test_predict_accepts_numpy_input(self, model):
        arr = SAMPLE_CLASS_0.values  # numpy array, no column names
        result = model.predict(arr)
        assert len(result) == 1
        assert result[0] in {0, 1}

    def test_predict_wrong_feature_count_raises(self, model):
        bad_input = pd.DataFrame([{"N": 100, "P": 5.0}])  # only 2 of 12 features
        with pytest.raises(Exception):
            model.predict(bad_input)


# ─────────────────────────────────────────────────────────────
# 4. Save / Load
# ─────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_model_creates_file(self, fresh_model, tmp_path):
        custom_pkl = str(tmp_path / "manual_save.pkl")
        fresh_model.save_path = custom_pkl
        fresh_model._save()
        assert os.path.exists(custom_pkl)

    def test_saved_pkl_contains_model_and_scaler_keys(self, fresh_model):
        fresh_model._save()
        state = joblib.load(fresh_model.save_path)
        assert "model" in state
        assert "scaler" in state

    def test_load_model_restores_scaler(self, fresh_model, tmp_path):
        original_mean = fresh_model.scaler.mean_.copy()
        fresh_model._save()

        # Wipe in-memory scaler
        fresh_model.scaler = None
        fresh_model._load()
        assert fresh_model.scaler is not None
        np.testing.assert_array_almost_equal(fresh_model.scaler.mean_, original_mean)

    def test_predict_consistent_after_save_load(self, fresh_model):
        pred_before = fresh_model.predict(MULTI_SAMPLE)
        fresh_model._save()
        fresh_model._load()
        pred_after = fresh_model.predict(MULTI_SAMPLE)
        np.testing.assert_array_equal(pred_before, pred_after)


# ─────────────────────────────────────────────────────────────
# 5. Second-instance load (the real hot-path check)
# ─────────────────────────────────────────────────────────────

class TestHotLoadPath:
    def test_second_instance_loads_without_retraining(self, tmp_path):
        pkl = str(tmp_path / "shared.pkl")

        # First instance trains and saves
        m1 = ferti(save_path=pkl, data_path=CSV_PATH)
        pred1 = m1.predict(MULTI_SAMPLE)

        # Second instance should load, not retrain
        m2 = ferti(save_path=pkl, data_path=CSV_PATH)
        pred2 = m2.predict(MULTI_SAMPLE)

        np.testing.assert_array_equal(pred1, pred2)

    def test_second_instance_pkl_mtime_unchanged(self, tmp_path):
        """pkl file must NOT be overwritten on second instantiation."""
        pkl = str(tmp_path / "shared.pkl")

        ferti(save_path=pkl, data_path=CSV_PATH)
        mtime_after_first = os.path.getmtime(pkl)

        ferti(save_path=pkl, data_path=CSV_PATH)
        mtime_after_second = os.path.getmtime(pkl)

        assert mtime_after_first == mtime_after_second, (
            "pkl was overwritten on second instantiation — model is retraining unnecessarily"
        )


# ─────────────────────────────────────────────────────────────
# 6. Sanity / smoke tests
# ─────────────────────────────────────────────────────────────

class TestSanity:
    def test_prediction_is_not_all_one_class(self, model):
        """Model shouldn't collapse to always predicting the same class."""
        # Use a 50-row sample from the actual CSV for this check
        df = pd.read_csv(CSV_PATH)
        df = df[df["Output"].isin([0, 1])].head(50)
        X = df[FEATURE_COLS]
        preds = model.predict(X)
        assert len(set(preds)) > 1, "Model always predicts the same class — likely broken"

    def test_accuracy_above_threshold_on_full_data(self, model):
        """Minimum smoke-test: >75 % accuracy on the full (filtered) dataset."""
        df = pd.read_csv(CSV_PATH)
        df = df[df["Output"].isin([0, 1])]
        X = df[FEATURE_COLS]
        y = df["Output"].values
        preds = model.predict(X)
        accuracy = (preds == y).mean()
        assert accuracy > 0.75, f"Accuracy {accuracy:.2%} is below acceptable threshold"