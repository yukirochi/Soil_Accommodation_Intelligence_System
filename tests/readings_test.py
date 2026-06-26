import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)


# from temp_storage_class.temporary_storage import temp_storage



# storage = temp_storage(
#     db_path=os.path.join(parent_dir, "database", "temporary.db"),
#     table_name="sensor_data",
# )

# storage.store({
#     "atmospheric_temp": 22.5,
#     "soil_temp": 18.3,
#     "humidity": 65.0,
#     "rainfall": 0.0,
#     "soil_moisture": 55.2,
#     "grid_x": 0,
#     "grid_y": 0,
#     "timestamp": "2025-06-26T22:00:00"
# })
from cleaning_pipeline.cleaning_input import cleaning

input = {
    "atmospheric_temp": 22.5,
    "soil_temp": 18.3,
    "humidity": 65.0,
    "rainfall": 0.0,
    "grid_x": 0,
    "grid_y": 0,
    "timestamp": "2025-06-26T22:00:00"
}

clean = cleaning(
    id= 'soil_moisture'
    )

print(clean.clean_input(input))