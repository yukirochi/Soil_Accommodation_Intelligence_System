
import pandas as pd
from input_manager import InputManager

mgr = InputManager(
    db_path='database/permanent.db',
    temp_db_path='database/temporary.db'
)

# ── Your 1-row input ──────────────────────────────────────────────
df = pd.DataFrame([{
    'hour':          10,     # hour of day (0–23)
    'soil_temp':     22.5,   # degrees Celsius
    'rain_fall':     0.0,    # mm
    'soil_moisture': 34.0    # current reading (%)
}])

result = mgr.get_prediction('soil_moisture', df)
print('Predicted future soil moisture:', result)



from input_manager import InputManager

mgr_fr = InputManager(
    db_path='database/permanent.db',
    temp_db_path='database/temporary.db'
)

df = pd.DataFrame([{
    'N':   90.0,   # Nitrogen        (kg/ha)
    'P':   42.0,   # Phosphorus      (kg/ha)
    'K':   43.0,   # Potassium       (kg/ha)
    'pH':  6.5,    # Soil pH         (0–14)
    'EC':  0.8,    # Electrical Conductivity (dS/m)
    'OC':  1.2,    # Organic Carbon  (%)
    'S':   18.0,   # Sulphur         (mg/kg)
    'Zn':  1.5,    # Zinc            (mg/kg)
    'Fe':  4.2,    # Iron            (mg/kg)
    'Cu':  1.2,    # Copper          (mg/kg)
    'Mn':  8.0,    # Manganese       (mg/kg)
    'B':   0.6,    # Boron           (mg/kg)
}])

results = mgr_fr.get_prediction('soil_fertility', df)
print('Fertility prediction:', results)
