"""
tests/test_input_manager.py
===========================
Pytest suite for InputManager.

Strategy
--------
* All SQLite and model imports are mocked so the tests run without a real
  database file or trained model artefacts.
* Fixtures build minimal but realistic DataFrames that match the expected
  input shapes documented in input_manager.py.
* Each test class groups related tests by method / concern.
"""

import sys
import types
import sqlite3
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub out heavyweight model packages before importing InputManager so the
# test file can be collected even when the model artefacts are absent.
# ---------------------------------------------------------------------------

def _make_model_stub(class_name: str) -> types.ModuleType:
    """Return a module stub that exports *class_name* as a MagicMock class."""
    mod = types.ModuleType(class_name)
    setattr(mod, class_name, MagicMock)
    return mod


# We register stub modules for all paths that InputManager imports lazily.
_STUB_MODULES = {
    "model": types.ModuleType("model"),
    "model.soil_moisture_model": types.ModuleType("model.soil_moisture_model"),
    "model.soil_moisture_model.soil_moisture_model": _make_model_stub("soily"),
    "model.soil_fertility_model": types.ModuleType("model.soil_fertility_model"),
    "model.soil_fertility_model.soil_fertility_model": _make_model_stub("ferti"),
}
for name, mod in _STUB_MODULES.items():
    sys.modules.setdefault(name, mod)

# Stub the Evaluation class so unit tests don't go through real evaluation logic
from unittest.mock import MagicMock as _MM
import types as _types
_eval_stub = _types.ModuleType("evaluation")
_eval_stub.Evaluation = _MM
sys.modules.setdefault("evaluation", _eval_stub)

from input_manager import InputManager  # noqa: E402  (must come after stubs)


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

DB_PATH = ":memory:"  # safe sentinel; real connections are always mocked


def _moisture_df(rows: int = 1, hour: int = 10) -> pd.DataFrame:
    """Minimal soil-moisture DataFrame matching the expected input columns."""
    return pd.DataFrame(
        {
            "hour": [hour] * rows,
            "soil_temp": [22.5] * rows,
            "rain_fall": [0.0] * rows,
            "soil_moisture": [35.0] * rows,
        }
    )


def _fertility_df() -> pd.DataFrame:
    """Minimal soil-fertility DataFrame with all 12 required columns."""
    cols = ["N", "P", "K", "pH", "EC", "OC", "S", "Zn", "Fe", "Cu", "Mn", "B"]
    return pd.DataFrame([{c: float(i + 1) for i, c in enumerate(cols)}])


def _last_5_rows() -> pd.DataFrame:
    """Simulate the 5 historical rows fetched from the DB for soil_moisture."""
    return pd.DataFrame(
        {
            "id": range(1, 6),
            "hour": [8, 9, 10, 11, 12],
            "soil_temp": [20.0] * 5,
            "rain_fall": [0.1] * 5,
            "soil_moisture": [30.0, 31.0, 32.0, 33.0, 34.0],
            "future_moisture": [30.5, 31.5, 32.5, 33.5, 34.5],
            # These may be present in a real DB row but are not required
        }
    )


@pytest.fixture
def manager():
    return InputManager(db_path=DB_PATH)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_db_path(self):
        mgr = InputManager()
        assert mgr.db_path == "database/permanent.db"

    def test_default_temp_db_path(self):
        mgr = InputManager()
        assert mgr.temp_db_path == "database/temporary.db"

    def test_custom_db_path(self, manager):
        assert manager.db_path == DB_PATH

    def test_custom_temp_db_path(self):
        mgr = InputManager(temp_db_path="/tmp/test.db")
        assert mgr.temp_db_path == "/tmp/test.db"


# ---------------------------------------------------------------------------
# TestGetPrediction
# ---------------------------------------------------------------------------

class TestGetPrediction:
    """Tests for InputManager.get_prediction()"""

    def test_returns_value_error_when_df_is_none(self, manager):
        result = manager.get_prediction("soil_moisture", None)
        assert isinstance(result, ValueError)

    def test_raises_for_unknown_model_name(self, manager):
        df = _moisture_df()
        with pytest.raises(ValueError, match="not recognized"):
            manager.get_prediction("unknown_model", df)

    @patch.object(InputManager, "_clean_data")
    def test_soil_moisture_calls_clean_data(self, mock_clean, manager):
        """get_prediction must delegate cleaning to _clean_data."""
        mock_clean.return_value = _moisture_df()
        # Stub the lazily-imported model
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [42.0]
        with patch.dict(
            "sys.modules",
            {
                "model.soil_moisture_model.soil_moisture_model": MagicMock(
                    soily=MagicMock(return_value=mock_model_instance)
                )
            },
        ):
            manager.get_prediction("soil_moisture", _moisture_df())
        # Use assert_called_once and inspect call_args to avoid ambiguous Series equality
        mock_clean.assert_called_once()
        call_name = mock_clean.call_args[0][0]
        assert call_name == "soil_moisture"

    @patch.object(InputManager, "_clean_data")
    def test_soil_fertility_calls_clean_data(self, mock_clean, manager):
        mock_clean.return_value = _fertility_df()
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = 1   # valid fertility label
        with patch.dict(
            "sys.modules",
            {
                "model.soil_fertility_model.soil_fertility_model": MagicMock(
                    ferti=MagicMock(return_value=mock_model_instance)
                )
            },
        ):
            manager.get_prediction("soil_fertility", _fertility_df())
        mock_clean.assert_called_once()
        call_name = mock_clean.call_args[0][0]
        assert call_name == "soil_fertility"

    @patch.object(InputManager, "_clean_data")
    def test_prediction_result_propagated(self, mock_clean, manager):
        """get_prediction returns the scalar result after evaluation unwraps it."""
        mock_clean.return_value = _moisture_df()
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [55.0]   # in-range moisture
        # Replace the stubbed evaluation with a minimal real-behaving stand-in
        class _FakeEval:
            def evaluate_result(self, name, result):
                return result[0]   # mirrors real Evaluation.evaluate_result for soil_moisture
        manager.evaluation = _FakeEval()
        with patch.dict(
            "sys.modules",
            {
                "model.soil_moisture_model.soil_moisture_model": MagicMock(
                    soily=MagicMock(return_value=mock_model_instance)
                )
            },
        ):
            result = manager.get_prediction("soil_moisture", _moisture_df())
        # evaluate_result unwraps result[0] and returns a scalar
        assert result == 55.0


# ---------------------------------------------------------------------------
# TestCleanDataSoilMoisture
# ---------------------------------------------------------------------------

class TestCleanDataSoilMoisture:
    """Tests for _clean_data() when name == 'soil_moisture'"""

    def _run_clean(self, manager, input_df=None, last_5=None):
        """Helper that patches sqlite3 + read_sql_query and calls _clean_data."""
        if input_df is None:
            input_df = _moisture_df()
        if last_5 is None:
            last_5 = _last_5_rows()

        with (
            patch("input_manager.sqlite3.connect") as mock_connect,
            patch("input_manager.pd.read_sql_query", return_value=last_5),
            patch.object(manager, "_store_to_permanent_db"),
            patch.object(manager, "_store_to_temporary_db"),
        ):
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            result = manager._clean_data("soil_moisture", input_df)

        return result

    def test_returns_dataframe(self, manager):
        result = self._run_clean(manager)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_expected_feature_columns(self, manager):
        expected_cols = {
            "hour_sin", "hour_cos", "soil_temp", "rain_fall",
            "prev_5", "prev_4", "prev_3", "prev_2", "prev_1",
        }
        result = self._run_clean(manager)
        assert expected_cols.issubset(set(result.columns))

    def test_output_is_single_row(self, manager):
        result = self._run_clean(manager)
        assert len(result) == 1

    def test_hour_sin_cos_encoding(self, manager):
        """Verify trigonometric hour encoding formula."""
        hour = 6
        result = self._run_clean(manager, input_df=_moisture_df(hour=hour))
        expected_sin = np.sin(2 * np.pi * hour / 24)
        expected_cos = np.cos(2 * np.pi * hour / 24)
        assert abs(float(result["hour_sin"].iloc[0]) - expected_sin) < 1e-5
        assert abs(float(result["hour_cos"].iloc[0]) - expected_cos) < 1e-5

    def test_dtypes_are_float32(self, manager):
        result = self._run_clean(manager)
        for col in result.columns:
            assert result[col].dtype == np.float32, f"{col} is not float32"

    def test_raises_value_error_for_unrecognized_name(self, manager):
        with pytest.raises(ValueError, match="Unrecognized clean_data target"):
            manager._clean_data("unknown_name", _moisture_df())

    def test_insufficient_history_returns_value_error(self, manager):
        """When history rows carry NaN prev_* values the null-check branch is hit."""
        # Build a history where future_moisture is NULL so prev_* shifts produce NaN.
        sparse_history = _last_5_rows().copy()
        sparse_history["future_moisture"] = np.nan
        result = self._run_clean(manager, last_5=sparse_history)
        # _clean_data returns a ValueError object (not raises) in this branch
        assert isinstance(result, ValueError)

    def test_commits_future_moisture_update(self, manager):
        with (
            patch("input_manager.sqlite3.connect") as mock_connect,
            patch("input_manager.pd.read_sql_query", return_value=_last_5_rows()),
            patch.object(manager, "_store_to_permanent_db"),
            patch.object(manager, "_store_to_temporary_db"),
        ):
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            manager._clean_data("soil_moisture", _moisture_df())

        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()


# ---------------------------------------------------------------------------
# TestCleanDataSoilFertilization
# ---------------------------------------------------------------------------

class TestCleanDataSoilFertilization:
    """Tests for _clean_data() when name == 'soil_fertilization'"""

    def _run_clean(self, manager, df=None):
        if df is None:
            df = _fertility_df()
        with patch("input_manager.sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            result = manager._clean_data("soil_fertilization", df)
        return result

    def test_returns_dataframe_with_correct_columns(self, manager):
        result = self._run_clean(manager)
        expected_cols = ["N", "P", "K", "pH", "EC", "OC", "S", "Zn", "Fe", "Cu", "Mn", "B"]
        assert list(result.columns) == expected_cols

    def test_returns_single_row(self, manager):
        result = self._run_clean(manager)
        assert len(result) == 1

    def test_dtypes_are_float32(self, manager):
        result = self._run_clean(manager)
        for col in result.columns:
            assert result[col].dtype == np.float32

    def test_raises_when_column_missing(self, manager):
        df = _fertility_df().drop(columns=["N", "Fe"])
        with pytest.raises(ValueError, match="soil_fertilization input missing columns"):
            self._run_clean(manager, df=df)

    def test_returns_none_when_row_has_nulls(self, manager):
        df = _fertility_df()
        df.loc[0, "N"] = np.nan
        result = self._run_clean(manager, df=df)
        assert result is None

    def test_inserts_row_into_db(self, manager):
        with patch("input_manager.sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            manager._clean_data("soil_fertilization", _fertility_df())

        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()
        mock_conn.close.assert_called()

    def test_closes_connection_on_missing_column_error(self, manager):
        """Connection should be closed even when a ValueError is raised early."""
        df = _fertility_df().drop(columns=["B"])
        with patch("input_manager.sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            with pytest.raises(ValueError):
                manager._clean_data("soil_fertilization", df)
        mock_conn.close.assert_called()


# ---------------------------------------------------------------------------
# TestStoreMoistureRow
# ---------------------------------------------------------------------------

class TestStoreMoistureRow:
    """Tests for the shared _store_moisture_row() helper."""

    def _make_latest(self):
        return pd.Series(
            {
                "hour_sin": 0.5,
                "hour_cos": 0.866,
                "soil_temp": 22.5,
                "rain_fall": 0.0,
                "prev_5": 30.0,
                "prev_4": 31.0,
                "prev_3": 32.0,
                "prev_2": 33.0,
                "prev_1": 34.0,
            }
        )

    def test_executes_update_and_insert(self, manager):
        with patch("input_manager.sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            manager._store_moisture_row(DB_PATH, self._make_latest(), 35.0)

        assert mock_conn.execute.call_count == 2
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_update_uses_correct_future_moisture_value(self, manager):
        future_val = 99.1
        with patch("input_manager.sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            manager._store_moisture_row(DB_PATH, self._make_latest(), future_val)

        first_call_args = mock_conn.execute.call_args_list[0][0]
        # The parameterised value should appear in the first execute() call
        assert (future_val,) in first_call_args or future_val in str(first_call_args)

    def test_insert_contains_all_feature_values(self, manager):
        latest = self._make_latest()
        with patch("input_manager.sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            manager._store_moisture_row(DB_PATH, latest, 35.0)

        second_call_args = mock_conn.execute.call_args_list[1][0]
        values_tuple = second_call_args[1]
        assert values_tuple[0] == pytest.approx(0.5)   # hour_sin
        assert values_tuple[2] == pytest.approx(22.5)  # soil_temp


# ---------------------------------------------------------------------------
# TestStoreToPermanentAndTemporaryDB
# ---------------------------------------------------------------------------

class TestStoreDelegation:
    """_store_to_permanent_db and _store_to_temporary_db must delegate correctly."""

    def _make_latest(self):
        return pd.Series({"hour_sin": 0.1, "hour_cos": 0.9, "soil_temp": 20.0,
                          "rain_fall": 0.0, "prev_5": 1.0, "prev_4": 2.0,
                          "prev_3": 3.0, "prev_2": 4.0, "prev_1": 5.0})

    def test_permanent_db_calls_store_moisture_row(self, manager):
        with patch.object(manager, "_store_moisture_row") as mock_store:
            manager._store_to_permanent_db(self._make_latest(), 10.0)
        mock_store.assert_called_once()
        called_db_path, _, called_future = mock_store.call_args[0]
        assert called_db_path == DB_PATH
        assert called_future == pytest.approx(10.0)

    def test_temporary_db_calls_store_moisture_row(self, manager):
        manager.temp_db_path = ":memory:"
        with patch.object(manager, "_store_moisture_row") as mock_store:
            manager._store_to_temporary_db(self._make_latest(), 10.0)
        mock_store.assert_called_once()
        called_db_path, _, called_future = mock_store.call_args[0]
        assert called_db_path == ":memory:"
        assert called_future == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TestIntegration — real SQLite, no DB mocking
# ---------------------------------------------------------------------------

_MOISTURE_SCHEMA = """
    CREATE TABLE soil_moisture (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hour_sin REAL, hour_cos REAL, soil_temp REAL, rain_fall REAL,
        prev_5 REAL, prev_4 REAL, prev_3 REAL, prev_2 REAL, prev_1 REAL,
        future_moisture REAL
    )
"""

_FERTILITY_SCHEMA = """
    CREATE TABLE soil_fertility (
        N REAL, P REAL, K REAL, pH REAL, EC REAL, OC REAL,
        S REAL, Zn REAL, Fe REAL, Cu REAL, Mn REAL, B REAL
    )
"""


def _seed_permanent_db(path: str, rows: int = 5) -> None:
    """Create schema and insert *rows* historical moisture rows with non-null future_moisture."""
    conn = sqlite3.connect(path)
    conn.execute(_MOISTURE_SCHEMA)
    conn.execute(_FERTILITY_SCHEMA)
    for i in range(rows):
        conn.execute(
            """INSERT INTO soil_moisture
               (hour_sin, hour_cos, soil_temp, rain_fall,
                prev_5, prev_4, prev_3, prev_2, prev_1, future_moisture)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (0.5, 0.866, 20.0 + i, 0.1, 25.0, 26.0, 27.0, 28.0, 29.0 + i, 30.0 + i),
        )
    conn.commit()
    conn.close()


def _create_empty_db(path: str) -> None:
    """Create schema-only DB (no rows) — mirrors temporary.db."""
    conn = sqlite3.connect(path)
    conn.execute(_MOISTURE_SCHEMA)
    conn.execute(_FERTILITY_SCHEMA)
    conn.commit()
    conn.close()


def _row_count(db_path: str, table: str) -> int:
    conn = sqlite3.connect(db_path)
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return count


class TestIntegration:
    """
    End-to-end tests against real SQLite files (tmp_path fixture).
    No DB layer is mocked — this is where we catch actual write failures.
    """

    @pytest.fixture
    def dbs(self, tmp_path):
        """Return (perm_path, temp_path) with seeded permanent and empty temporary DB."""
        perm = str(tmp_path / "permanent.db")
        temp = str(tmp_path / "temporary.db")
        _seed_permanent_db(perm, rows=5)
        _create_empty_db(temp)
        return perm, temp

    @pytest.fixture
    def real_manager(self, dbs):
        perm, temp = dbs
        return InputManager(db_path=perm, temp_db_path=temp)

    # ------------------------------------------------------------------
    # soil_moisture — actual DB write verification
    # ------------------------------------------------------------------

    def test_moisture_writes_one_row_to_permanent_db(self, real_manager, dbs):
        perm, _ = dbs
        real_manager._clean_data("soil_moisture", _moisture_df())
        assert _row_count(perm, "soil_moisture") == 6  # 5 seeded + 1 new

    def test_moisture_writes_one_row_to_temporary_db(self, real_manager, dbs):
        """THE core regression test: temp DB must receive a row after each cycle."""
        _, temp = dbs
        real_manager._clean_data("soil_moisture", _moisture_df())
        assert _row_count(temp, "soil_moisture") == 1

    def test_moisture_stored_feature_values_are_correct(self, real_manager, dbs):
        """Values written to temp DB must match the computed features."""
        _, temp = dbs
        real_manager._clean_data("soil_moisture", _moisture_df(hour=10))
        conn = sqlite3.connect(temp)
        row = conn.execute(
            "SELECT hour_sin, hour_cos, soil_temp, rain_fall FROM soil_moisture"
        ).fetchone()
        conn.close()
        expected_sin = np.sin(2 * np.pi * 10 / 24)
        expected_cos = np.cos(2 * np.pi * 10 / 24)
        assert row[0] == pytest.approx(expected_sin, abs=1e-4)
        assert row[1] == pytest.approx(expected_cos, abs=1e-4)
        assert row[2] == pytest.approx(22.5, abs=0.01)
        assert row[3] == pytest.approx(0.0, abs=0.01)

    def test_moisture_temp_db_accumulates_across_calls(self, real_manager, dbs):
        """
        With the IS NOT NULL query fix, every cycle produces a row.
        3 calls yield 3 rows in temp DB.
        """
        _, temp = dbs
        for _ in range(3):
            real_manager._clean_data("soil_moisture", _moisture_df())
        assert _row_count(temp, "soil_moisture") == 3

    def test_moisture_permanent_db_future_moisture_backfill(self, real_manager, dbs):
        """
        Call 1 inserts a row with future_moisture=NULL and backfills any prior NULL.
        Call 2 backfills call 1's NULL row, then inserts a new one.
        After 2 calls there is always exactly 1 NULL (the latest row).
        """
        perm, _ = dbs
        real_manager._clean_data("soil_moisture", _moisture_df())
        real_manager._clean_data("soil_moisture", _moisture_df())
        conn = sqlite3.connect(perm)
        null_count = conn.execute(
            "SELECT COUNT(*) FROM soil_moisture WHERE future_moisture IS NULL"
        ).fetchone()[0]
        conn.close()
        assert null_count == 1

    # ------------------------------------------------------------------
    # soil_fertilization — actual DB write verification
    # ------------------------------------------------------------------

    def test_fertility_writes_row_to_permanent_db(self, real_manager, dbs):
        perm, _ = dbs
        real_manager._clean_data("soil_fertilization", _fertility_df())
        assert _row_count(perm, "soil_fertility") == 1

    def test_fertility_stored_values_match_input(self, real_manager, dbs):
        perm, _ = dbs
        real_manager._clean_data("soil_fertilization", _fertility_df())
        conn = sqlite3.connect(perm)
        row = conn.execute("SELECT N, P, K FROM soil_fertility").fetchone()
        conn.close()
        assert row[0] == pytest.approx(1.0, abs=0.01)  # N
        assert row[1] == pytest.approx(2.0, abs=0.01)  # P
        assert row[2] == pytest.approx(3.0, abs=0.01)  # K

    # ------------------------------------------------------------------
    # get_prediction end-to-end (real DB + mocked model only)
    # ------------------------------------------------------------------

    def test_get_prediction_moisture_triggers_temp_db_write(self, real_manager, dbs):
        """Full pipeline: real DB ops + real evaluate_result, only model mocked."""
        _, temp = dbs
        import importlib.util, os
        _spec = importlib.util.spec_from_file_location(
            "_eval_real",
            os.path.join(os.path.dirname(__file__), "..", "evaluation.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        real_manager.evaluation = _mod.Evaluation()

        mock_model = MagicMock()
        mock_model.predict.return_value = [36.0]
        with patch.dict(
            "sys.modules",
            {
                "model.soil_moisture_model.soil_moisture_model": MagicMock(
                    soily=MagicMock(return_value=mock_model)
                )
            },
        ):
            result = real_manager.get_prediction("soil_moisture", _moisture_df())

        # evaluate_result unwraps [36.0] -> 36.0 and validates range
        assert isinstance(result, float)
        assert 0 <= result <= 100
        assert _row_count(temp, "soil_moisture") == 1, (
            "get_prediction must write one row to the temporary DB"
        )


# ---------------------------------------------------------------------------
# TestGetPredictionIntegration — real model, real DB, real evaluation
# ---------------------------------------------------------------------------

import importlib.util as _ilu
import os as _os

_MODEL_DIR = _os.path.join(
    _os.path.dirname(__file__), "..", "model", "soil_moisture_model"
)
_PKL_EXISTS = _os.path.exists(_os.path.join(_MODEL_DIR, "soily.pkl"))

full_integration = pytest.mark.skipif(
    not _PKL_EXISTS,
    reason="soily.pkl not found — skipping full integration tests"
)


class TestGetPredictionIntegration:
    """
    Tests get_prediction() with the real soily model, real SQLite databases,
    and real Evaluation class. These verify the complete pipeline end-to-end.
    """

    @staticmethod
    def _real_soily():
        spec = _ilu.spec_from_file_location(
            "_soily_real", _os.path.join(_MODEL_DIR, "soil_moisture_model.py")
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.soily

    @staticmethod
    def _real_evaluation():
        spec = _ilu.spec_from_file_location(
            "_eval_real",
            _os.path.join(_os.path.dirname(__file__), "..", "evaluation.py")
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.Evaluation()

    @pytest.fixture
    def real_dbs(self, tmp_path):
        perm = str(tmp_path / "permanent.db")
        temp = str(tmp_path / "temporary.db")
        _seed_permanent_db(perm, rows=5)
        _create_empty_db(temp)
        return perm, temp

    def _make_mgr(self, perm, temp):
        mgr = InputManager(db_path=perm, temp_db_path=temp)
        mgr.evaluation = self._real_evaluation()
        return mgr

    @full_integration
    def test_get_prediction_returns_float_in_valid_range(self, real_dbs):
        """get_prediction returns a float between 0 and 100."""
        perm, temp = real_dbs
        original_dir = _os.getcwd()
        _os.chdir(_MODEL_DIR)
        try:
            with patch.dict("sys.modules", {
                "model.soil_moisture_model.soil_moisture_model":
                    type("_mod", (), {"soily": self._real_soily()})()
            }):
                mgr = self._make_mgr(perm, temp)
                result = mgr.get_prediction("soil_moisture", _moisture_df())
        finally:
            _os.chdir(original_dir)

        assert isinstance(result, float), f"Expected float, got {type(result)}: {result}"
        assert 0 <= result <= 100, f"Result {result} is out of valid range 0-100"

    @full_integration
    def test_get_prediction_writes_to_temp_db(self, real_dbs):
        """After get_prediction, exactly one row is inserted into the temporary DB."""
        perm, temp = real_dbs
        original_dir = _os.getcwd()
        _os.chdir(_MODEL_DIR)
        try:
            with patch.dict("sys.modules", {
                "model.soil_moisture_model.soil_moisture_model":
                    type("_mod", (), {"soily": self._real_soily()})()
            }):
                mgr = self._make_mgr(perm, temp)
                mgr.get_prediction("soil_moisture", _moisture_df())
        finally:
            _os.chdir(original_dir)

        assert _row_count(temp, "soil_moisture") == 1

    @full_integration
    def test_get_prediction_raises_on_out_of_range_result(self, real_dbs):
        """Evaluation must raise ValueError if predicted value is outside 0-100."""
        perm, temp = real_dbs
        mock_model = MagicMock()
        mock_model.predict.return_value = [-999.0]  # intentionally invalid
        with patch.dict("sys.modules", {
            "model.soil_moisture_model.soil_moisture_model": MagicMock(
                soily=MagicMock(return_value=mock_model)
            )
        }):
            mgr = self._make_mgr(perm, temp)
            with pytest.raises(ValueError, match="out of expected range"):
                mgr.get_prediction("soil_moisture", _moisture_df())

    @full_integration
    def test_three_consecutive_predictions_all_succeed(self, real_dbs):
        """Three consecutive calls all return valid floats and accumulate 3 temp rows."""
        perm, temp = real_dbs
        results = []
        original_dir = _os.getcwd()
        _os.chdir(_MODEL_DIR)
        try:
            with patch.dict("sys.modules", {
                "model.soil_moisture_model.soil_moisture_model":
                    type("_mod", (), {"soily": self._real_soily()})()
            }):
                mgr = self._make_mgr(perm, temp)
                for reading in [
                    {'hour': 8,  'soil_temp': 21.0, 'rain_fall': 0.0, 'soil_moisture': 34.5},
                    {'hour': 9,  'soil_temp': 21.8, 'rain_fall': 0.1, 'soil_moisture': 34.2},
                    {'hour': 10, 'soil_temp': 22.5, 'rain_fall': 0.0, 'soil_moisture': 33.9},
                ]:
                    results.append(mgr.get_prediction("soil_moisture", pd.DataFrame([reading])))
        finally:
            _os.chdir(original_dir)

        assert len(results) == 3, "All 3 cycles should produce a result"
        for r in results:
            assert isinstance(r, float), f"Expected float, got {type(r)}: {r}"
            assert 0 <= r <= 100, f"Result {r} out of range"
        assert _row_count(temp, "soil_moisture") == 3, "All 3 cycles should write to temp DB"

