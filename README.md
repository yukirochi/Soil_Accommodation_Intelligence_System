# Soil Accommodation Intelligence System

## Overview

The Soil Accommodation Intelligence System is a machine learning pipeline designed to support real-time soil condition monitoring. The system provides two independent predictive capabilities: soil moisture forecasting and soil fertility classification. It is architected around a continuous inference loop that captures live sensor readings, generates predictions, logs outcomes, and periodically retrains the underlying models using the accumulated data — all without requiring manual intervention between cycles.

The project is intended to serve as an embedded intelligence layer for an agricultural monitoring setup, where sensor readings arrive at regular intervals and the system must produce a decision or forecast on each cycle.

---

## Table of Contents

1. [Results and Findings](#results-and-findings)
2. [System Architecture](#system-architecture)
3. [Models](#models)
4. [Data and Feature Engineering](#data-and-feature-engineering)
5. [Database Design](#database-design)
6. [Prediction Pipeline](#prediction-pipeline)
7. [Retraining Pipeline](#retraining-pipeline)
8. [Testing](#testing)
9. [Project Structure](#project-structure)
10. [Setup and Usage](#setup-and-usage)
11. [Known Limitations](#known-limitations)

---

## Results and Findings

### Soil Moisture Model

The soil moisture model is a linear SGD Regressor trained on USCRN (United States Climate Reference Network) hourly soil data from a Texas station, year 2017. Feature engineering introduces a five-step lagged moisture window alongside cyclic hour encoding and environmental readings. The model predicts the next observed moisture value one time step ahead.

A five-cycle live simulation conducted after all pipeline defects were resolved produced the following predictions:

| Cycle | Hour | Actual Moisture (%) | Predicted Future Moisture (%) |
|-------|------|---------------------|-------------------------------|
| 1     | 8    | 34.5                | 26.16                         |
| 2     | 9    | 34.2                | 26.27                         |
| 3     | 10   | 33.9                | 30.52                         |
| 4     | 11   | 33.6                | 34.20                         |
| 5     | 12   | 33.3                | 34.22                         |

**Key finding:** The model exhibits a warm-up period during which predictions deviate from the actual range. This is expected behaviour caused by the sliding window features (`prev_1` through `prev_5`) being initially populated from the seeded historical values in the permanent database rather than live readings. As consecutive live readings accumulate and overwrite the seeded history, predictions converge to the real moisture range, as demonstrated in cycles 4 and 5 where the predicted value closely tracks the actual reading. No corrective action is required; the warm-up window is a structural property of the feature design.

### Soil Fertility Model

The soil fertility model is a Random Forest Classifier trained on a 12-feature soil chemistry dataset covering macronutrients (N, P, K), micronutrients (Zn, Fe, Cu, Mn, B), and physicochemical properties (pH, EC, OC, S). The model outputs a binary label: `0` (unfertile) or `1` (fertile). The classifier is trained with `class_weight='balanced'` to handle potential class imbalance in the training set.

### Pipeline Integrity

Several critical defects were identified and resolved during development and testing:

- The temporary database was missing its schema entirely, causing silent failures on every write attempt.
- The SQLite parameter binding in the moisture pipeline used a bare scalar instead of a tuple, which resulted in a `ProgrammingError` on any real database call.
- The retraining flush operation defined a DELETE query that was never executed, meaning the temporary database accumulated rows indefinitely and retraining would train on the same data repeatedly.
- An alternating skip pattern caused one in every two prediction cycles to abort due to null features being introduced by an unfiltred history query.

All five defects documented across the two test reports have been resolved. The full test suite of 53 tests passes cleanly.

---

## System Architecture

```
Sensor Input (raw DataFrame)
         |
         v
   InputManager
   /            \
_clean_data    get_prediction
   |                |
Feature         ML Model
Engineering     (soily / ferti)
   |                |
SQLite          Evaluation
(permanent,     (range/class check)
 temporary)         |
                predictions.db
                (logged output)
         |
         v
   Retraining
   (partial_fit on temporary.db,
    flush consumed rows)
```

The system operates as a cycle:

1. A sensor reading arrives as a pandas DataFrame.
2. `InputManager` validates, transforms, and stores the reading.
3. The appropriate model generates a prediction.
4. `Evaluation` validates the prediction is within expected bounds.
5. The prediction is logged to `predictions.db` alongside a backfilled actual value from the following cycle.
6. The `Retraining` class is called periodically to perform online learning on completed rows in the temporary database.

---

## Models

### Soil Moisture Model (`soily`)

| Property       | Value                                         |
|----------------|-----------------------------------------------|
| Algorithm      | SGD Regressor (`squared_error` loss)          |
| Learning rate  | Constant, `eta0 = 0.01`                       |
| Regularization | None                                          |
| Max iterations | 50,000                                        |
| Tolerance      | 1e-8                                          |
| Scaler         | StandardScaler                                |
| Persistence    | `model/soil_moisture_model/soily.pkl`         |
| Online learning| `partial_fit` via `Retraining.train()`        |

The model is initialised from `soily.pkl` if the file exists, otherwise it trains from scratch using the bundled CSV. Online retraining uses `partial_fit` so the existing parameter state is updated incrementally rather than reset.

### Soil Fertility Model (`ferti`)

| Property       | Value                                           |
|----------------|-------------------------------------------------|
| Algorithm      | Random Forest Classifier                        |
| Estimators     | 100                                             |
| Class weighting| Balanced                                        |
| Scaler         | StandardScaler                                  |
| Persistence    | `model/soil_fertility_model/ferti_model.pkl`    |
| Online learning| Not supported; full retrain on demand           |

The fertility model does not currently support `partial_fit`. Retraining requires a full fit from a new dataset.

---

## Data and Feature Engineering

### Soil Moisture Dataset

The training data originates from the USCRN hourly soil dataset for a Texas station (2017). Feature engineering is implemented in `training/soil_moisture/query.sql` using DuckDB's window functions:

| Feature      | Description                                                   |
|--------------|---------------------------------------------------------------|
| `hour_sin`   | Sine encoding of the hour of day (cyclic, period 24)          |
| `hour_cos`   | Cosine encoding of the hour of day (cyclic, period 24)        |
| `soil_temp`  | Soil temperature in degrees Celsius                           |
| `rain_fall`  | Rainfall in millimetres                                       |
| `prev_5`     | Soil moisture reading from 5 time steps prior                 |
| `prev_4`     | Soil moisture reading from 4 time steps prior                 |
| `prev_3`     | Soil moisture reading from 3 time steps prior                 |
| `prev_2`     | Soil moisture reading from 2 time steps prior                 |
| `prev_1`     | Soil moisture reading from 1 time step prior                  |
| `future_moisture` | Target: next observed soil moisture value (%)            |

Rows missing any lag value or the target are excluded before training. Soil moisture values are scaled from the raw fractional representation to a percentage scale (multiplied by 100).

Hour is encoded using sine and cosine transforms rather than a raw integer to preserve the cyclic nature of time (i.e., hour 23 and hour 0 are adjacent, not distant).

### Soil Fertility Dataset

The fertility dataset contains 12 soil chemistry features. Only samples labelled `0` or `1` are retained; any rows with other output labels are discarded before training.

| Feature | Description                 |
|---------|-----------------------------|
| `N`     | Nitrogen (kg/ha)            |
| `P`     | Phosphorus (kg/ha)          |
| `K`     | Potassium (kg/ha)           |
| `pH`    | Soil acidity (0-14 scale)   |
| `EC`    | Electrical conductivity (dS/m) |
| `OC`    | Organic carbon (%)          |
| `S`     | Sulphur (mg/kg)             |
| `Zn`    | Zinc (mg/kg)                |
| `Fe`    | Iron (mg/kg)                |
| `Cu`    | Copper (mg/kg)              |
| `Mn`    | Manganese (mg/kg)           |
| `B`     | Boron (mg/kg)               |

---

## Database Design

The system uses three SQLite databases, each serving a distinct purpose.

### `permanent.db`

Seeded from the full training dataset at initialisation. Used as the historical context source during inference. The `soil_moisture` table holds the pre-engineered feature rows. The `soil_fertility` table holds raw nutrient readings for audit.

### `temporary.db`

Same schema as `permanent.db` but starts empty. Populated exclusively by live inference cycles via `InputManager`. This database is the source of truth for the `Retraining` class. After a successful `partial_fit`, all completed rows (`future_moisture IS NOT NULL`) are deleted from this database to prevent duplicate training. Rows with a pending `future_moisture` value are preserved.

### `predictions.db`

Records every prediction cycle for monitoring and accuracy evaluation. Each row stores the predicted value, the actual value observed on the following cycle (backfilled), and the computed absolute error.

| Column                | Description                                           |
|-----------------------|-------------------------------------------------------|
| `id`                  | Auto-incremented primary key                         |
| `timestamp`           | UTC timestamp of the prediction cycle                |
| `predicted_moisture`  | Predicted future soil moisture (%)                   |
| `actual_moisture`     | Actual soil moisture reading from the next cycle (%) |
| `absolute_error`      | `ABS(predicted - actual)`, filled on backfill        |

The backfill mechanism mirrors the same rolling-window pattern used for `future_moisture`: each new cycle fills the `actual_moisture` of the previous cycle's row before inserting its own prediction.

---

## Prediction Pipeline

The `InputManager` class is the entry point for all inference. The two supported model names are `soil_moisture` and `soil_fertility`.

### Moisture Prediction Cycle

1. Capture the raw `soil_moisture` reading from the input DataFrame before feature engineering (this value will be used to backfill the previous cycle's `actual_moisture`).
2. Retrieve the last 5 rows with a non-null `future_moisture` from `permanent.db` as historical context.
3. Backfill the most recent null `future_moisture` row in `permanent.db` with the current sensor reading.
4. Concatenate the 5 historical rows with the current row and compute all lag features and cyclic encodings.
5. Extract the tail row as the feature vector for inference.
6. Write the feature vector to both `permanent.db` and `temporary.db`.
7. Run inference via `soily.predict()`.
8. Validate the output is within `[0, 100]` via `Evaluation`.
9. Log the prediction to `predictions.db` and backfill the previous cycle's `actual_moisture`.

### Fertility Prediction Cycle

1. Validate all 12 required feature columns are present in the input DataFrame.
2. Extract and cast the feature row to `float32`.
3. Log the raw reading to the `soil_fertility` table in `permanent.db` for audit purposes.
4. Run inference via `ferti.predict()`.
5. Validate the output is `0` or `1` via `Evaluation`.

### Invoking the Pipeline

```python
import pandas as pd
from input_manager import InputManager

mgr = InputManager(
    db_path='database/permanent.db',
    temp_db_path='database/temporary.db',
    predictions_db_path='database/predictions.db'
)

# Soil moisture prediction
moisture_df = pd.DataFrame([{
    'hour':          10,
    'soil_temp':     22.5,
    'rain_fall':     0.0,
    'soil_moisture': 34.0
}])
predicted_moisture = mgr.get_prediction('soil_moisture', moisture_df)

# Soil fertility prediction
fertility_df = pd.DataFrame([{
    'N': 90.0, 'P': 42.0, 'K': 43.0, 'pH': 6.5,
    'EC': 0.8, 'OC': 1.2, 'S': 18.0, 'Zn': 1.5,
    'Fe': 4.2, 'Cu': 1.2, 'Mn': 8.0, 'B': 0.6
}])
predicted_fertility = mgr.get_prediction('soil_fertility', fertility_df)
```

---

## Retraining Pipeline

The `Retraining` class performs online learning on the soil moisture model using data collected during live inference.

### Process

1. Query all completed rows (`future_moisture IS NOT NULL`) from `temporary.db`.
2. Drop the `id` column.
3. Call `soily.train()`, which runs `partial_fit` and saves the updated model to `soily.pkl`.
4. If training succeeds, call `flush()` to delete all completed rows from `temporary.db`.
5. Pending rows (`future_moisture IS NULL`) are never deleted; they remain for the next prediction cycle to backfill.

### Triggering Retraining

```python
from retraining import Retraining

r = Retraining(temp_db_path='database/temporary.db')
result = r.train()  # Returns True on success, False if skipped or failed
```

Retraining is safe to call at any frequency. If no completed rows exist in the temporary database, the call returns `False` immediately without touching the model.

---

## Testing

Testing is split across two test files covering the two primary classes. The combined suite contains 53 tests.

### Running the Tests

```bash
# Full suite
.venv/bin/python -m pytest tests/ -v

# InputManager only
.venv/bin/python -m pytest tests/test_input_manager.py -v

# Retraining only
.venv/bin/python -m pytest tests/test_retraining.py -v
```

### Test Strategy

**Unit tests** mock all SQLite connections and ML model dependencies using `unittest.mock.patch`. They verify routing logic, feature engineering computations, dtype enforcement, null guard behaviour, and database write call signatures without touching disk.

**Integration tests** use `pytest`'s `tmp_path` fixture to create isolated, real SQLite databases per test. The permanent database is seeded with five completed historical rows. These tests verify actual row counts, column values, and backfill correctness against real database state.

**Model stubs at collection time.** Because `soily` and `ferti` are imported lazily or at module level, stub modules are injected into `sys.modules` before the test file is collected. This allows the test suite to run on any machine regardless of whether trained model artefacts are present. Integration tests that require a real model artefact are automatically skipped if `soily.pkl` is absent.

### Test Results

| Module       | Test File                     | Tests | Result       |
|--------------|-------------------------------|-------|--------------|
| InputManager | `tests/test_input_manager.py` | 37    | 37 passed    |
| Retraining   | `tests/test_retraining.py`    | 16    | 16 passed    |
| **Total**    |                               | **53**| **53 passed**|

Full test reports are available at:
- `tests/reports/input_manager_test_report.md`
- `tests/reports/retraining_test_report.md`

### Defects Found and Resolved

| # | Module        | Defect                                                          | Severity |
|---|---------------|-----------------------------------------------------------------|----------|
| 1 | `database.py` | `temporary.db` created without any tables                       | Critical |
| 2 | `input_manager.py` | `temp_db_path` not assigned in `__init__`                  | Critical |
| 3 | `input_manager.py` | SQLite parameter passed as bare scalar instead of tuple    | Critical |
| 4 | `input_manager.py` | History query included NULL rows, causing alternating skips | High     |
| 5 | `input_manager.py` | SQLite connection not closed after commit in `_clean_data` | Medium   |
| 6 | `retraining.py` | `sqlite3` module called as a function in `flush()`           | Critical |
| 7 | `retraining.py` | DELETE query defined but `conn.execute()` never called       | Critical |
| 8 | `retraining.py` | `conn.commit()` called on a read-only SELECT connection      | High     |
| 9 | `retraining.py` | `df.drop(columns='id')` raises `KeyError` when absent       | Medium   |

---

## Project Structure

```
soil_accomodation_intelligence_system/
│
├── database/
│   ├── database.py             # Initialises all three SQLite databases
│   ├── datasets/               # Source CSV files for seeding permanent.db
│   ├── permanent.db            # Seeded historical training data
│   ├── temporary.db            # Live inference accumulation (flushed on retrain)
│   └── predictions.db          # Prediction log with backfilled actuals
│
├── model/
│   ├── soil_moisture_model/
│   │   ├── soil_moisture_model.py  # soily class (SGDRegressor + StandardScaler)
│   │   ├── soil_moist.csv          # Training dataset
│   │   └── soily.pkl               # Saved model state
│   └── soil_fertility_model/
│       ├── soil_fertility_model.py # ferti class (RandomForestClassifier)
│       ├── soil_fertilization.csv  # Training dataset
│       └── ferti_model.pkl         # Saved model state
│
├── training/
│   ├── soil_moisture/
│   │   ├── eda.ipynb           # Exploratory data analysis notebook
│   │   ├── feature.py          # DuckDB feature engineering script
│   │   └── query.sql           # SQL feature engineering query
│   └── soil_fertilization/
│       ├── eda.ipynb           # Exploratory data analysis notebook
│       └── soil_fertilization.csv
│
├── tests/
│   ├── test_input_manager.py   # 37 tests for InputManager
│   ├── test_retraining.py      # 16 tests for Retraining
│   └── reports/
│       ├── input_manager_test_report.md
│       └── retraining_test_report.md
│
├── input_manager.py            # Core inference and data pipeline class
├── evaluation.py               # Output validation (range and class checks)
├── retraining.py               # Online retraining and flush logic
├── feature_test.py             # Manual end-to-end prediction demonstration
└── README.md
```

---

## Setup and Usage

### Prerequisites

- Python 3.12 or later
- A virtual environment is recommended

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pandas scikit-learn joblib numpy duckdb pytest
```

### Initialising the Databases

Run this once before the first inference cycle. It creates and seeds all three databases.

```bash
cd database
python database.py
cd ..
```

Expected output:
```
Databases initialised: permanent.db (seeded) + temporary.db (empty) + predictions.db (empty)
```

### Running a Prediction

```bash
python feature_test.py
```

This script demonstrates a soil moisture prediction and a soil fertility prediction using example sensor values.

### Running the Test Suite

```bash
.venv/bin/python -m pytest tests/ -v
```

No database files or trained model artefacts are required for the unit tests. Integration tests that depend on `soily.pkl` are automatically skipped if the file is not present.

---

## Known Limitations

- The soil moisture model exhibits a warm-up period of approximately 5 cycles when the temporary database is empty or when the seeded historical values differ significantly from the live reading range. Predictions produced during this period will be less accurate.
- The two integration tests (`test_moisture_temp_db_accumulates_across_calls` and `test_moisture_permanent_db_future_moisture_backfill`) are sensitive to the state of the live `permanent.db` file. Running the full test suite immediately after a live prediction session may cause these two tests to fail. They pass reliably when the database is in its freshly seeded state.
- The soil fertility model does not support online retraining. Adding new labelled fertility samples requires a full model retrain triggered manually.
- The `predictions.db` accuracy log for soil fertility is currently incomplete: no automatic mechanism fills `actual_fertility` for the fertility model. This field requires manual population or an external ground-truth feed.
- There is no scheduler or daemon process included in this repository. The caller is responsible for invoking `InputManager.get_prediction()` and `Retraining.train()` at the appropriate intervals.
