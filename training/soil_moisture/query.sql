WITH raw_data AS (
    SELECT 
        row_number() OVER () AS row_id,
        *
    FROM read_csv_auto('training/uscrn_soil_2017_TX_hourly.csv')
),
casing AS (
    SELECT 
        row_id,
        Hour_sin AS hour_sin,
        Hour_cos AS hour_cos,
        Soil_Temp AS soil_temp,
        Rainfall AS rain_fall,
        Soil_Moisture * 100 AS soil_moisture
    FROM raw_data
),
cleaning AS (
    SELECT 
        hour_sin,
        hour_cos,
        soil_temp,
        rain_fall,
        LAG(soil_moisture, 5) OVER (ORDER BY row_id) AS prev_5,
        LAG(soil_moisture, 4) OVER (ORDER BY row_id) AS prev_4,
        LAG(soil_moisture, 3) OVER (ORDER BY row_id) AS prev_3,
        LAG(soil_moisture, 2) OVER (ORDER BY row_id) AS prev_2,
        LAG(soil_moisture, 1) OVER (ORDER BY row_id) AS prev_1,
        LEAD(soil_moisture) OVER (ORDER BY row_id) AS future_moisture
    FROM casing
)
SELECT * FROM cleaning
WHERE prev_5 IS NOT NULL
  AND future_moisture IS NOT NULL