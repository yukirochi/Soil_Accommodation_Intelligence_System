"""
Pytest suite for the `soily` class.

Assumes the class lives in a module importable as `soily`
(e.g. a file named soily.py on the path containing `class soily`).
Adjust the import below if your module/file is named differently.
"""
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from soil_moisture_model import soily

FEATURE_COLUMNS = [
    'hour_sin', 'hour_cos', 'soil_temp', 'rain_fall',
    'prev_5', 'prev_4', 'prev_3', 'prev_2', 'prev_1',
]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _make_dataframe(n=60, seed=42):
    """Small, deterministic, roughly-linear synthetic dataset with the
    columns soily expects."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n) % 24
    df = pd.DataFrame({
        'hour_sin': np.sin(2 * np.pi * hours / 24),
        'hour_cos': np.cos(2 * np.pi * hours / 24),
        'soil_temp': rng.normal(20, 3, n),
        'rain_fall': rng.uniform(0, 5, n),
        'prev_5': rng.uniform(10, 40, n),
        'prev_4': rng.uniform(10, 40, n),
        'prev_3': rng.uniform(10, 40, n),
        'prev_2': rng.uniform(10, 40, n),
        'prev_1': rng.uniform(10, 40, n),
    })
    # target is a noisy linear combination so the model has something
    # real (but easy) to learn
    df['future_moisture'] = (
        0.5 * df['prev_1'] + 0.3 * df['prev_2'] + 0.1 * df['rain_fall']
        + rng.normal(0, 0.5, n)
    )
    return df


@pytest.fixture
def sample_df():
    return _make_dataframe()


@pytest.fixture
def csv_path(tmp_path, sample_df):
    path = tmp_path / "soil_moist.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def save_path(tmp_path):
    # Just a path inside tmp_path; deliberately does not exist yet.
    return str(tmp_path / "soily.pkl")


@pytest.fixture
def fresh_model(csv_path, save_path):
    """A soily instance built from scratch (no pre-existing save file)."""
    return soily(data_path=csv_path, save_path=save_path)


# --------------------------------------------------------------------------
# __init__ / _initalize_base_model
# --------------------------------------------------------------------------

class TestInit:

    def test_creates_new_scaler_and_model_when_no_save_file(self, csv_path, save_path):
        assert not os.path.exists(save_path)
        s = soily(data_path=csv_path, save_path=save_path)

        assert isinstance(s.scaler, StandardScaler)
        assert isinstance(s.model, SGDRegressor)

    def test_does_not_write_save_file_on_first_init(self, csv_path, save_path):
        soily(data_path=csv_path, save_path=save_path)
        # init only trains in-memory; nothing should be persisted yet
        assert not os.path.exists(save_path)

    def test_scaler_is_fit_on_init(self, csv_path, save_path):
        s = soily(data_path=csv_path, save_path=save_path)
        # a fitted StandardScaler exposes mean_/scale_
        assert hasattr(s.scaler, 'mean_')
        assert s.scaler.mean_.shape == (len(FEATURE_COLUMNS),)

    def test_model_is_fit_on_init(self, csv_path, save_path):
        s = soily(data_path=csv_path, save_path=save_path)
        # a fitted SGDRegressor exposes coef_
        assert hasattr(s.model, 'coef_')
        assert s.model.coef_.shape == (len(FEATURE_COLUMNS),)

    def test_loads_existing_state_instead_of_reinitializing(self, csv_path, save_path, sample_df):
        # First run: train and persist a model.
        first = soily(data_path=csv_path, save_path=save_path)
        first.save_model()
        trained_coef = first.model.coef_.copy()

        # Second run: point at a data_path that doesn't exist. If the
        # class tried to rebuild the base model it would raise a
        # FileNotFoundError reading the csv, so success here proves the
        # load branch was taken.
        second = soily(data_path='/nonexistent/does_not_exist.csv', save_path=save_path)

        np.testing.assert_allclose(second.model.coef_, trained_coef)
        assert isinstance(second.scaler, StandardScaler)

    def test_init_raises_if_data_path_missing_and_no_save_file(self, save_path):
        with pytest.raises(FileNotFoundError):
            soily(data_path='/nonexistent/does_not_exist.csv', save_path=save_path)


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------

class TestPredict:

    def test_predict_returns_correct_shape(self, fresh_model, sample_df):
        X = sample_df[FEATURE_COLUMNS].iloc[:5]
        preds = fresh_model.predict(X)
        assert preds.shape == (5,)

    def test_predict_returns_numeric_array(self, fresh_model, sample_df):
        X = sample_df[FEATURE_COLUMNS].iloc[:3]
        preds = fresh_model.predict(X)
        assert np.issubdtype(preds.dtype, np.floating)
        assert np.all(np.isfinite(preds))

    def test_predict_uses_the_fitted_scaler(self, fresh_model, sample_df):
        # Predictions on scaled-away-from-training data shouldn't blow up
        # or be trivially zero/constant for varied inputs.
        X = sample_df[FEATURE_COLUMNS].iloc[:10]
        preds = fresh_model.predict(X)
        assert len(np.unique(preds)) > 1

    def test_predict_wrong_column_count_raises(self, fresh_model, sample_df):
        bad_X = sample_df[FEATURE_COLUMNS].iloc[:5, :-1]  # drop one column
        with pytest.raises(ValueError):
            fresh_model.predict(bad_X)


# --------------------------------------------------------------------------
# train / save_model / load_model
# --------------------------------------------------------------------------

class TestTrainSaveLoad:

    def test_train_updates_model_coefficients(self, fresh_model, sample_df):
        before = fresh_model.model.coef_.copy()
        fresh_model.train(sample_df.iloc[:10])
        after = fresh_model.model.coef_.copy()
        assert not np.allclose(before, after)

    def test_train_persists_to_save_path(self, fresh_model, sample_df, save_path):
        assert not os.path.exists(save_path)
        fresh_model.train(sample_df.iloc[:10])
        assert os.path.exists(save_path)

    def test_train_drops_target_column_before_fitting(self, fresh_model, sample_df, monkeypatch):
        seen = {}
        original_partial_fit = fresh_model.model.partial_fit

        def spy_partial_fit(X, y, *args, **kwargs):
            seen['n_features'] = X.shape[1]
            return original_partial_fit(X, y, *args, **kwargs)

        monkeypatch.setattr(fresh_model.model, 'partial_fit', spy_partial_fit)
        # train() also calls save_model(), which would try to pickle the
        # model -- and this spy closure isn't picklable. We only care
        # about the shape seen by partial_fit here, so stub out the
        # actual disk write for this test.
        monkeypatch.setattr(joblib, 'dump', lambda *args, **kwargs: None)

        fresh_model.train(sample_df.iloc[:10])
        assert seen['n_features'] == len(FEATURE_COLUMNS)

    def test_save_model_writes_expected_keys(self, fresh_model, save_path):
        fresh_model.save_model()
        state = joblib.load(save_path)
        assert set(state.keys()) == {'model', 'scaler'}
        assert isinstance(state['model'], SGDRegressor)
        assert isinstance(state['scaler'], StandardScaler)

    def test_load_model_restores_matching_state(self, fresh_model, save_path):
        fresh_model.save_model()
        original_coef = fresh_model.model.coef_.copy()
        original_mean = fresh_model.scaler.mean_.copy()

        # mutate in place, then reload and confirm restoration
        fresh_model.model = SGDRegressor()
        fresh_model.scaler = StandardScaler()
        fresh_model.load_model()

        np.testing.assert_allclose(fresh_model.model.coef_, original_coef)
        np.testing.assert_allclose(fresh_model.scaler.mean_, original_mean)

    def test_train_then_reload_round_trip(self, csv_path, save_path, sample_df):
        s = soily(data_path=csv_path, save_path=save_path)
        s.train(sample_df.iloc[:10])
        trained_coef = s.model.coef_.copy()

        reloaded = soily(data_path=csv_path, save_path=save_path)
        np.testing.assert_allclose(reloaded.model.coef_, trained_coef)


# --------------------------------------------------------------------------
# current_performance
# --------------------------------------------------------------------------

class TestCurrentPerformance:

    def test_returns_r2_and_rmse(self, fresh_model, sample_df):
        X_test = sample_df[FEATURE_COLUMNS].iloc[40:]
        y_test = sample_df['future_moisture'].iloc[40:]

        r2, rmse = fresh_model.current_performance(X_test, y_test)

        assert isinstance(r2, float)
        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_perfect_predictions_give_zero_rmse_and_r2_one(self, fresh_model, sample_df, monkeypatch):
        X_test = sample_df[FEATURE_COLUMNS].iloc[40:]
        y_test = sample_df['future_moisture'].iloc[40:]

        monkeypatch.setattr(
            fresh_model.model, 'predict', lambda X: y_test.to_numpy()
        )

        r2, rmse = fresh_model.current_performance(X_test, y_test)
        assert r2 == pytest.approx(1.0)
        assert rmse == pytest.approx(0.0, abs=1e-8)