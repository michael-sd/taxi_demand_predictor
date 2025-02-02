from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR

def download_raw_data_file(year: int, month: int) -> Path:
  """
  Downloads Parquet file with historical taxi rides for the given year and month
  """
  URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
  response = requests.get(URL)

  if response.status_code == 200:
    path = RAW_DATA_DIR / f'rides_{year}_{month:02d}.parquet'
    open(path, "wb").write(response.content)
    return path
  else:
    raise Exception(f'{URL} is not available')

def validate_raw_data(
    rides: pd.DataFrame,
    year: int,
    month: int,
) -> pd.DataFrame:
  """
  Removes rows with pickup_datetimes outside of their valid range
  """
  # keep only rides for this month
  this_month_start = f'{year}-{month:02d}-01'
  next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
  rides = rides[rides.pickup_datetime >= this_month_start]
  rides = rides[rides.pickup_datetime < next_month_start]

  return rides