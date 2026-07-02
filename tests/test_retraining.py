"""
tests/test_retraining.py
========================
Pytest suite for the Retraining class.

Strategy
--------
* Unit tests mock soily and sqlite3.connect so they run without model
  artefacts or a real database.
* Integration tests use real SQLite temp files (pytest tmp_path) and a
  real soily model loaded from soily.pkl to verify end-to-end behaviour.
"""

import sys
import os
import types
import sqlite3
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub model imports so the file can be collected without artefacts
# ---------------------------------------------------------------------------

def _make_stub(class_name: str) -> types.ModuleType:
    mod = types.ModuleType(class_name)
    setattr(mod, class_name, MagicMock)
    return mod

for _name, _mod in {
    "model": types.ModuleType("model"),
    "model.soil_moisture_model": types.ModuleType("model.soil_moisture_model"),
    "model.soil_moisture_model.soil_moisture_model": _make_stub("soily"),
}.items():
    sys.modules.setdefault(_name, _mod)

from retraining import Retraining  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
    CREATE TABLE soil_moisture (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hour_sin REAL, hour_cos REAL, soil_temp REAL, rain_fall REAL,
        prev_5 REAL, prev_4 REAL, prev_3 REAL, prev_2 REAL, prev_1 REAL,
        future_moisture REAL
    )
"""

_FEATURE_COLS = [
    "hour_sin", "hour_cos", "soil_temp", "rain_fall",
    "prev_5", "prev_4", "prev_3", "prev_2", "prev_1",
]


def _seed_db(path: str, rows: int = 5, include_null: bool = True) -> None:
    """Create schema and insert *rows* completed rows plus optionally one NULL row."""
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    for i in range(rows):
        conn.execute(
            f"INSERT INTO soil_moisture ({', '.join(_FEATURE_COLS)}, future_moisture) "
            f"VALUES ({', '.join(['?']*10)})",
            (0.5, 0.866, 20.0 + i, 0.1, 25.0, 26.0, 27.0, 28.0, 29.0 + i, 30.0 + i),
        )
    if include_null:
        conn.execute(
            f"INSERT INTO soil_moisture ({', '.join(_FEATURE_COLS)}, future_moisture) "
            f"VALUES ({', '.join(['?']*10)})",
            (0.5, 0.866, 25.0, 0.0, 25.0, 26.0, 27.0, 28.0, 30.0, None),
        )
    conn.commit()
    conn.close()


def _row_count(path: str, where: str = "") -> int:
    conn = sqlite3.connect(path)
    clause = f"WHERE {where}" if where else ""
    n = conn.execute(f"SELECT COUNT(*) FROM soil_moisture {clause}").fetchone()[0]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_paths(self):
        r = Retraining()
        assert r.temp_db == "database/temporary.db"
        assert r.table_name == "soil_moisture"

    def test_custom_temp_db_path(self):
        r = Retraining(temp_db_path="/tmp/test.db")
        assert r.temp_db == "/tmp/test.db"

    def test_custom_table_name(self):
        r = Retraining(table_name="other_table")
        assert r.table_name == "other_table"


# ---------------------------------------------------------------------------
# TestTrain — unit tests (mocked soily + sqlite3)
# ---------------------------------------------------------------------------

class TestTrain:

    def _make_df(self, rows: int = 5) -> pd.DataFrame:
        data = {col: [float(i)] * rows for i, col in enumerate(_FEATURE_COLS)}
        data["future_moisture"] = [30.0 + i for i in range(rows)]
        data["id"] = list(range(1, rows + 1))  # mimic real DB output which includes id
        return pd.DataFrame(data)

    @patch("retraining.sqlite3.connect")
    @patch("retraining.pd.read_sql_query")
    @patch("retraining.soily")
    def test_train_calls_model_train(self, mock_soily_cls, mock_read_sql, mock_connect):
        mock_read_sql.return_value = self._make_df()
        mock_model = MagicMock()
        mock_model.train.return_value = True
        mock_soily_cls.return_value = mock_model

        r = Retraining()
        result = r.train()

        mock_model.train.assert_called_once()
        assert result is True

    @patch("retraining.sqlite3.connect")
    @patch("retraining.pd.read_sql_query")
    @patch("retraining.soily")
    def test_train_returns_false_when_df_is_empty(self, mock_soily_cls, mock_read_sql, mock_connect):
        mock_read_sql.return_value = pd.DataFrame()
        mock_model = MagicMock()
        mock_soily_cls.return_value = mock_model

        r = Retraining()
        result = r.train()

        mock_model.train.assert_not_called()
        assert result is False

    @patch("retraining.sqlite3.connect")
    @patch("retraining.pd.read_sql_query")
    @patch("retraining.soily")
    def test_train_calls_flush_on_success(self, mock_soily_cls, mock_read_sql, mock_connect):
        mock_read_sql.return_value = self._make_df()
        mock_model = MagicMock()
        mock_model.train.return_value = True
        mock_soily_cls.return_value = mock_model

        r = Retraining()
        with patch.object(r, "flush") as mock_flush:
            r.train()
        mock_flush.assert_called_once()

    @patch("retraining.sqlite3.connect")
    @patch("retraining.pd.read_sql_query")
    @patch("retraining.soily")
    def test_train_returns_false_on_model_value_error(self, mock_soily_cls, mock_read_sql, mock_connect):
        mock_read_sql.return_value = self._make_df()
        mock_model = MagicMock()
        mock_model.train.side_effect = ValueError("df contains NaN values")
        mock_soily_cls.return_value = mock_model

        r = Retraining()
        result = r.train()

        assert result is False

    @patch("retraining.sqlite3.connect")
    @patch("retraining.pd.read_sql_query")
    @patch("retraining.soily")
    def test_train_does_not_flush_on_model_error(self, mock_soily_cls, mock_read_sql, mock_connect):
        mock_read_sql.return_value = self._make_df()
        mock_model = MagicMock()
        mock_model.train.side_effect = TypeError("wrong type")
        mock_soily_cls.return_value = mock_model

        r = Retraining()
        with patch.object(r, "flush") as mock_flush:
            r.train()
        mock_flush.assert_not_called()

    @patch("retraining.sqlite3.connect")
    @patch("retraining.pd.read_sql_query")
    @patch("retraining.soily")
    def test_train_drops_id_column_before_training(self, mock_soily_cls, mock_read_sql, mock_connect):
        # _make_df already includes an id column — verify it is absent from the trained df
        mock_read_sql.return_value = self._make_df()
        mock_model = MagicMock()
        mock_model.train.return_value = True
        mock_soily_cls.return_value = mock_model

        r = Retraining()
        r.train()

        trained_df = mock_model.train.call_args[0][0]
        assert "id" not in trained_df.columns

    @patch("retraining.sqlite3.connect")
    @patch("retraining.pd.read_sql_query")
    @patch("retraining.soily")
    def test_train_only_reads_completed_rows(self, mock_soily_cls, mock_read_sql, mock_connect):
        """Verify the SQL query filters on future_moisture IS NOT NULL."""
        mock_read_sql.return_value = self._make_df()
        mock_soily_cls.return_value = MagicMock(train=MagicMock(return_value=True))

        r = Retraining()
        r.train()

        sql_arg = mock_read_sql.call_args[0][0]
        assert "future_moisture IS NOT NULL" in sql_arg


# ---------------------------------------------------------------------------
# TestFlush — unit tests
# ---------------------------------------------------------------------------

class TestFlush:

    def test_flush_executes_delete_and_commits(self):
        r = Retraining()
        with patch("retraining.sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            r.flush()

        executed_sql = mock_conn.execute.call_args[0][0]
        assert "DELETE" in executed_sql
        assert "future_moisture IS NOT NULL" in executed_sql
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_flush_connects_to_correct_db(self):
        r = Retraining(temp_db_path="/custom/path.db")
        with patch("retraining.sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            r.flush()

        mock_connect.assert_called_once_with("/custom/path.db")


# ---------------------------------------------------------------------------
# TestIntegration — real SQLite + real soily model
# ---------------------------------------------------------------------------

# Path to the real model artefact, resolved relative to this file
_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "model", "soil_moisture_model"
)
_PKL_EXISTS = os.path.exists(os.path.join(_MODEL_DIR, "soily.pkl"))

integration = pytest.mark.skipif(
    not _PKL_EXISTS,
    reason="soily.pkl not found — skipping integration tests"
)


class TestIntegration:
    """
    End-to-end tests using a real SQLite temp DB and the real soily model.
    soily is loaded directly from its file path to bypass the stub modules
    registered in sys.modules at collection time.
    """

    @staticmethod
    def _real_soily():
        """Load soily directly from file, bypassing the sys.modules stub."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_real_soily",
            os.path.join(_MODEL_DIR, "soil_moisture_model.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.soily

    @pytest.fixture
    def temp_db(self, tmp_path):
        path = str(tmp_path / "temporary.db")
        _seed_db(path, rows=5, include_null=True)
        return path

    @pytest.fixture
    def empty_db(self, tmp_path):
        path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(path)
        conn.execute(_SCHEMA)
        conn.commit()
        conn.close()
        return path

    @integration
    def test_train_triggers_partial_fit_and_returns_true(self, temp_db):
        original_dir = os.getcwd()
        os.chdir(_MODEL_DIR)
        try:
            with patch("retraining.soily", self._real_soily()):
                r = Retraining(temp_db_path=temp_db)
                result = r.train()
        finally:
            os.chdir(original_dir)
        assert result is True

    @integration
    def test_flush_removes_completed_rows_only(self, temp_db):
        """After flush, only the NULL-future_moisture row should remain."""
        original_dir = os.getcwd()
        os.chdir(_MODEL_DIR)
        try:
            with patch("retraining.soily", self._real_soily()):
                r = Retraining(temp_db_path=temp_db)
                r.train()
        finally:
            os.chdir(original_dir)

        nulls = _row_count(temp_db, "future_moisture IS NULL")
        completed = _row_count(temp_db, "future_moisture IS NOT NULL")
        assert nulls == 1
        assert completed == 0

    @integration
    def test_train_skips_when_no_completed_rows(self, empty_db):
        """When temp DB has no completed rows, train returns False."""
        original_dir = os.getcwd()
        os.chdir(_MODEL_DIR)
        try:
            with patch("retraining.soily", self._real_soily()):
                r = Retraining(temp_db_path=empty_db)
                result = r.train()
        finally:
            os.chdir(original_dir)
        assert result is False

    @integration
    def test_train_does_not_delete_null_row(self, temp_db):
        """The pending (NULL future_moisture) row must not be deleted by flush."""
        original_dir = os.getcwd()
        os.chdir(_MODEL_DIR)
        try:
            with patch("retraining.soily", self._real_soily()):
                r = Retraining(temp_db_path=temp_db)
                r.train()
        finally:
            os.chdir(original_dir)

        conn = sqlite3.connect(temp_db)
        null_rows = conn.execute(
            "SELECT * FROM soil_moisture WHERE future_moisture IS NULL"
        ).fetchall()
        conn.close()
        assert len(null_rows) == 1

