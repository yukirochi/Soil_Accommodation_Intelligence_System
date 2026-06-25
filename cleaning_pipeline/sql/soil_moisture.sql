WITH raw_data AS (
    SELECT
        *
    FROM read_csv_auto('cleaning_pipeline/assets/soil_dataset.csv')
),

cleaned_data AS (
    SELECT
        Atmospheric_Temp::DECIMAL(10, 2) AS atmospheric_temp,
        Soil_Temp::DECIMAL(10, 2) AS soil_temp,
        Humidity::FLOAT as humidity,
        Rainfall::FLOAT as rainfall,
        LAG(Soil_Moisture::FLOAT) OVER (ORDER BY time) as previous_soil_moisture,
        Soil_Moisture::FLOAT as soil_moisture,
        LEAD(Soil_Moisture::FLOAT) OVER (ORDER BY time) as future_soil_moisture
    FROM
        raw_data
    WHERE
        Atmospheric_Temp IS NOT NULL
        AND Soil_Temp IS NOT NULL
        AND Humidity IS NOT NULL
        AND Soil_Moisture IS NOT NULL
    ORDER BY
        time
)

SELECT
    *
FROM
    cleaned_data
WHERE
    previous_soil_moisture IS NOT NULL
    AND future_soil_moisture IS NOT NULL