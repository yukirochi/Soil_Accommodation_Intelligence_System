WITH raw_data AS (
    SELECT 
    * 
    FROM  
    read_csv_auto('cleaning_pipeline/assets/soil_fertilization.csv')
),

cleaned_data AS (
    SELECT 
        N::FLOAT AS N,
        P::FLOAT AS P,
        K::FLOAT AS K,
        output::INT AS output
    FROM 
        raw_data
)

SELECT 
* 
FROM 
cleaned_data 
WHERE 
N IS NOT NULL AND
P IS NOT NULL AND
K IS NOT NULL AND
output IS NOT NULL