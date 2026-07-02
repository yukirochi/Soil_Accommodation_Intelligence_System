# Retraining Test Report

**Project:** Soil Accommodation Intelligence System  
**Module Under Test:** `retraining.py` — `Retraining` class  
**Test File:** `tests/test_retraining.py`  
**Date:** 2026-07-02  
**Test Runner:** pytest 9.1.1, Python 3.12.3  

---

## 1. Overview

This report documents the review, defect identification, corrective actions, and test results for the `Retraining` class. The `Retraining` class is responsible for reading completed soil moisture records from the temporary database, passing them to the `soily` model for online learning via `partial_fit`, and then flushing consumed records from the temporary database to prevent duplicate training.

The review was conducted in two phases:

- **Static code review** — manual inspection of `retraining.py` prior to writing any tests. This phase identified four defects, three of which would have caused immediate runtime crashes.
- **Automated testing** — a pytest suite of 16 tests covering unit and integration scenarios. Unit tests mock all external dependencies. Integration tests use real SQLite files and the real trained `soily.pkl` model artefact.

---

## 2. Defects Discovered During Static Review

All four defects below were found by reading `retraining.py` before a single test was executed.

### Defect 1 — `sqlite3` Module Called as a Function

**Severity:** Critical  
**Location:** `flush()`, line 31  
**Original code:**
```python
conn = sqlite3(self.temp_db)
```
**Symptom:** Any call to `flush()` would raise `TypeError: 'module' object is not callable`. The `sqlite3` identifier refers to the imported module, not a callable class.  
**Fix applied:**
```python
conn = sqlite3.connect(self.temp_db)
```

---

### Defect 2 — DELETE Query Defined but Never Executed

**Severity:** Critical  
**Location:** `flush()`, line 32  
**Original code:**
```python
def flush(self):
    conn = sqlite3(self.temp_db)
    query = """DELETE FROM soil_moisture WHERE future_moisture IS NOT NULL"""
    conn.commit()
    conn.close()
```
**Symptom:** Even after fixing Defect 1, `conn.execute(query)` was absent. The DELETE statement was defined as a string but never passed to the database. `conn.commit()` would commit an empty transaction, leaving all completed rows in the temporary database permanently. The flush operation would silently do nothing.  
**Fix applied:** Added `conn.execute(query)` before `conn.commit()`.

---

### Defect 3 — `conn.commit()` Called on a Read-Only Query and Connection Not Closed

**Severity:** High  
**Location:** `train()`, lines 20–21  
**Original code:**
```python
conn = sqlite3.connect(self.temp_db)
query = """SELECT * FROM soil_moisture WHERE future_moisture IS NOT NULL"""
df = pd.read_sql_query(query, conn)
conn.commit()
conn.close()
```
**Symptom:** `conn.commit()` on a read-only `SELECT` is a no-op but is misleading — it implies a write occurred. More significantly, the original code placed `conn.close()` after `conn.commit()`, meaning the connection was correctly closed in the nominal path but could remain open if `pd.read_sql_query` raised an exception before reaching the close call.  
**Fix applied:** Removed `conn.commit()`. Retained `conn.close()` in the correct position. A future improvement would wrap the block in a `try/finally` to guarantee closure on exception.

---

### Defect 4 — `df.drop(columns='id')` Raises `KeyError` When Column Is Absent

**Severity:** Medium  
**Location:** `train()`, line 23  
**Original code:**
```python
df.drop(columns='id', inplace=True)
```
**Symptom:** The `id` column is present when data is fetched from a real SQLite database (where it is the primary key). However, if the DataFrame originates from a source that does not include `id`, this line raises `KeyError: "['id'] not found in axis"`. The code provides no defensive handling.  
**Fix applied:**
```python
df.drop(columns='id', inplace=True, errors='ignore')
```

---

## 3. Corrected `retraining.py`

The following shows the state of `retraining.py` after all fixes were applied:

```python
import pandas as pd
import sqlite3
from model.soil_moisture_model.soil_moisture_model import soily


class Retraining:

    def __init__(self, temp_db_path='database/temporary.db', table_name='soil_moisture'):
        self.temp_db = temp_db_path
        self.table_name = table_name

    def train(self):
        model = soily()

        conn = sqlite3.connect(self.temp_db)
        query = """SELECT * FROM soil_moisture WHERE future_moisture IS NOT NULL"""
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            print("Retraining skipped: no completed rows in temporary DB yet.")
            return False

        df.drop(columns='id', inplace=True, errors='ignore')

        try:
            result = model.train(df)
        except (TypeError, ValueError) as e:
            print(f"Retraining aborted: model.train() rejected the data — {e}")
            return False

        if result is True:
            self.flush()

        return result

    def flush(self):
        conn = sqlite3.connect(self.temp_db)
        query = """DELETE FROM soil_moisture WHERE future_moisture IS NOT NULL"""
        conn.execute(query)
        conn.commit()
        conn.close()
```

---

## 4. Test Structure

### 4.1 Test Classes

| Class | Scope | Tests |
|---|---|---|
| `TestInit` | Constructor parameter defaults and overrides | 3 |
| `TestTrain` | `train()` logic — mocked soily and sqlite3 | 7 |
| `TestFlush` | `flush()` correctness — mocked sqlite3 | 2 |
| `TestIntegration` | Real SQLite files and real soily model | 4 |
| **Total** | | **16** |

### 4.2 Key Testing Decisions

**Model stub at import time.** `retraining.py` imports `soily` at the module level via `from model.soil_moisture_model.soil_moisture_model import soily`. To allow the test file to be collected on machines without the model package installed, a stub module is injected into `sys.modules` before `Retraining` is imported.

**Patching `retraining.soily` in unit tests.** Because the `soily` name is bound in the `retraining` module's namespace at import time, unit tests use `patch("retraining.soily", ...)` rather than patching the source module. This correctly replaces the reference used by `Retraining.train()`.

**File-based loading of real `soily` in integration tests.** The stub registered at collection time blocks `importlib.import_module` from traversing the `model` package hierarchy because `model.soil_moisture_model` is a plain `ModuleType` stub rather than a real package. Integration tests therefore load the real `soily` class directly from its file path using `importlib.util.spec_from_file_location`, bypassing `sys.modules` entirely. This class is then injected via `patch("retraining.soily", ...)` for the duration of each integration test.

**Working directory management in integration tests.** The `soily` model resolves `soily.pkl` relative to the current working directory. Integration tests use `os.chdir(_MODEL_DIR)` inside a `try/finally` block to ensure the working directory is always restored, even if the test raises an exception.

---

## 5. Execution Results

### 5.1 Final Run (after all fixes applied)

```
platform linux -- Python 3.12.3, pytest-9.1.1
collected 16 items

tests/test_retraining.py::TestInit::test_default_paths                              PASSED
tests/test_retraining.py::TestInit::test_custom_temp_db_path                        PASSED
tests/test_retraining.py::TestInit::test_custom_table_name                          PASSED
tests/test_retraining.py::TestTrain::test_train_calls_model_train                   PASSED
tests/test_retraining.py::TestTrain::test_train_returns_false_when_df_is_empty      PASSED
tests/test_retraining.py::TestTrain::test_train_calls_flush_on_success              PASSED
tests/test_retraining.py::TestTrain::test_train_returns_false_on_model_value_error  PASSED
tests/test_retraining.py::TestTrain::test_train_does_not_flush_on_model_error       PASSED
tests/test_retraining.py::TestTrain::test_train_drops_id_column_before_training     PASSED
tests/test_retraining.py::TestTrain::test_train_only_reads_completed_rows           PASSED
tests/test_retraining.py::TestFlush::test_flush_executes_delete_and_commits         PASSED
tests/test_retraining.py::TestFlush::test_flush_connects_to_correct_db             PASSED
tests/test_retraining.py::TestIntegration::test_train_triggers_partial_fit_and_returns_true  PASSED
tests/test_retraining.py::TestIntegration::test_flush_removes_completed_rows_only   PASSED
tests/test_retraining.py::TestIntegration::test_train_skips_when_no_completed_rows  PASSED
tests/test_retraining.py::TestIntegration::test_train_does_not_delete_null_row      PASSED

16 passed, 20 warnings in 2.91s
```

The 20 warnings originate from `joblib` and relate to a NumPy 2.5 deprecation in array shape assignment. These are third-party warnings unrelated to the `Retraining` class and require no corrective action.

---

## 6. Integration Test Verification

The four integration tests collectively verify the complete retraining cycle against a real SQLite database seeded with five completed rows and one pending row (with `future_moisture = NULL`).

### 6.1 Verified Behaviours

| Test | Verified Behaviour |
|---|---|
| `test_train_triggers_partial_fit_and_returns_true` | `model.train()` is invoked and returns `True`, confirming `partial_fit` ran without error |
| `test_flush_removes_completed_rows_only` | After `train()`, all 5 completed rows are deleted; only the 1 NULL row remains |
| `test_train_skips_when_no_completed_rows` | When the temporary DB contains no completed rows, `train()` returns `False` without calling the model |
| `test_train_does_not_delete_null_row` | The pending row (`future_moisture IS NULL`) is preserved after `flush()` |

### 6.2 Database State Before and After

**Before `train()`:**

| Rows with `future_moisture IS NOT NULL` | Rows with `future_moisture IS NULL` | Total |
|---|---|---|
| 5 | 1 | 6 |

**After `train()`:**

| Rows with `future_moisture IS NOT NULL` | Rows with `future_moisture IS NULL` | Total |
|---|---|---|
| 0 | 1 | 1 |

The flush correctly consumed only completed records, leaving the pending row intact for the next prediction cycle to backfill.

---

## 7. Summary of All Fixes Applied

| # | File | Change | Severity |
|---|---|---|---|
| 1 | `retraining.py` | `flush()`: replaced `sqlite3(self.temp_db)` with `sqlite3.connect(self.temp_db)` | Critical |
| 2 | `retraining.py` | `flush()`: added `conn.execute(query)` so DELETE is actually executed | Critical |
| 3 | `retraining.py` | `train()`: removed unnecessary `conn.commit()` on read-only connection; retained `conn.close()` | High |
| 4 | `retraining.py` | `train()`: added `errors='ignore'` to `df.drop(columns='id')` to prevent `KeyError` | Medium |

---

## 8. How to Run the Tests

```bash
# From the project root
.venv/bin/python -m pytest tests/test_retraining.py -v
```

Unit tests require no external artefacts. Integration tests require `model/soil_moisture_model/soily.pkl` to be present. If the file is absent, integration tests are automatically skipped with the message `soily.pkl not found — skipping integration tests`.

---

## 9. Combined Suite

To run all tests across both `InputManager` and `Retraining` in a single command:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected result: 53 tests collected (37 from `test_input_manager.py` + 16 from `test_retraining.py`). 51 pass cleanly. The 2 tests marked as state-dependent below are sensitive to the runtime state of the live `permanent.db` file and may fail if that file has been modified by the live prediction simulation outside of the test run.

| Test | Dependency |
|---|---|
| `TestIntegration::test_moisture_temp_db_accumulates_across_calls` | Assumes `permanent.db` has exactly the seeded rows present |
| `TestIntegration::test_moisture_permanent_db_future_moisture_backfill` | Assumes the most recent `permanent.db` row has `future_moisture IS NULL` |

These two tests pass in complete isolation (using `tmp_path` fixtures). The failures only occur when the full suite is run against the live database after a prediction session has already been conducted.

