WITH raw_data AS (
    SELECT
        *
    FROM {inp}
),

cleaned_data AS (
    SELECT
        Atmospheric_Temp::DECIMAL(10, 2) AS atmospheric_temp,
        Soil_Temp::DECIMAL(10, 2) AS soil_temp,
        Humidity::FLOAT as humidity,
        Rainfall::FLOAT as rainfall,
        Soil_Moisture::FLOAT as soil_moisture,
        
    FROM
        raw_data
    WHERE
        Atmospheric_Temp IS NOT NULL
        AND Soil_Temp IS NOT NULL
        AND Humidity IS NOT NULL
        AND Soil_Moisture IS NOT NULL
)

SELECT
    *
FROM
    cleaned_data
