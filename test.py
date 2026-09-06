import os
import duckdb
import requests
from tqdm import tqdm
import hashlib

# 1. Setup file paths safely
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

input_file = os.path.join(data_dir, "finalwithscore.parquet")
output_file = os.path.join(data_dir, "benchmark_sites.parquet")
boundary_file = os.path.join(data_dir, "countries.geojson")

# 2. Download the country boundaries if they don't exist yet
if not os.path.exists(boundary_file):
    print("Fetching official country boundaries from web...")
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(boundary_file, 'wb') as f, tqdm(
        desc="Downloading boundaries",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            bar.update(len(data))
            f.write(data)
    print("Saved boundaries locally.")
else:
    print("Country boundaries found locally.")

# 3. Run the pure DuckDB spatial pipeline
print("Running DuckDB spatial pipeline...")
con = duckdb.connect()

# Install and load the spatial extension
con.execute("INSTALL spatial; LOAD spatial;")

# Construct and execute the query
query = f"""
COPY (
    WITH india_poly AS (
        -- Load India's polygon directly from the GeoJSON
        -- The column was renamed from 'ADMIN' to 'name' by the dataset maintainers
        SELECT geom 
        FROM st_read('{boundary_file}') 
        WHERE "name" = 'India'
    ),
    points_in_india AS (
        -- Read parquet and Spatial Join points falling inside India
        SELECT f.*
        FROM '{input_file}' f
        JOIN india_poly i 
          ON ST_Within(ST_Point(f.long, f.lat), i.geom)
    ),
    gridded AS (
        -- Create the ~11km grid
        SELECT *, 
               round(lat, 1) as grid_lat, 
               round(long, 1) as grid_lon
        FROM points_in_india
    ),
    sampled AS (
        -- Pick 1 per grid, then sample 1000 randomly
        SELECT *
        FROM (
            SELECT * EXCLUDE (grid_lat, grid_lon),
                   ROW_NUMBER() OVER(PARTITION BY grid_lat, grid_lon) as rn
            FROM gridded
        )
        WHERE rn = 1
        ORDER BY random()
        LIMIT 5000
    )
    -- Remove the row number column and save to parquet
    SELECT * EXCLUDE (rn) FROM sampled
) TO '{output_file}' (FORMAT PARQUET);
"""

con.execute(query)
print(f"Success! Saved filtered sites to {output_file}")