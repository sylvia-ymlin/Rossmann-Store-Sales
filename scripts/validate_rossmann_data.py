import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest.ingestor import DataIngestorFactory
from src.core.logger import setup_logger

logger = setup_logger(__name__)

def validate_data(train_path):
    """
    Performs data integrity and continuity checks on the Rossmann dataset.
    """
    try:
        # 1. Ingest Data
        factory = DataIngestorFactory()
        ingestor = factory.get_data_ingestor("rossmann")
        df = ingestor.ingest(train_path)
        
        logger.info(f"Loaded dataset with {len(df)} rows.")
        
        # 2. Check for missing values
        missing_values = df.isnull().sum()
        logger.info(f"Missing values per column:\n{missing_values[missing_values > 0]}")
        
        # 3. Check Date Continuity
        # Group by Store and check if dates are continuous
        logger.info("Checking date continuity per store...")
        store_id = df['Store'].unique()[0] # Check first store as sample for efficiency
        store_data = df[df['Store'] == store_id].sort_values('Date')
        
        min_date = store_data['Date'].min()
        max_date = store_data['Date'].max()
        expected_range = pd.date_range(start=min_date, end=max_date)
        
        missing_dates = expected_range.difference(store_data['Date'])
        if len(missing_dates) > 0:
            logger.warning(f"Store {store_id} has {len(missing_dates)} missing dates in range {min_date.date()} to {max_date.date()}")
        else:
            logger.info(f"Store {store_id} has a continuous date range.")

        # 4. Check for Store x Product (Rossmann is Store x Date, but we can check if all Stores have entries)
        num_stores = df['Store'].nunique()
        logger.info(f"Total unique stores: {num_stores}")
        
        if 'StoreType' in df.columns:
            logger.info(f"Store Types distribution:\n{df['StoreType'].value_counts()}")
        
        # 5. Sales Statistics
        logger.info(f"Sales Stats:\n{df['Sales'].describe()}")
        
        # Check for non-stationarity (sample trend)
        monthly_sales = df.set_index('Date').resample('ME')['Sales'].mean()
        logger.info(f"Monthly Avg Sales Trend (first 5 months):\n{monthly_sales.head()}")

        return True
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    train_csv = os.path.abspath("data/raw/train.csv")
    if os.path.exists(train_csv):
        success = validate_data(train_csv)
        if success:
            print("Rossmann data validation completed successfully.")
        else:
            print("Rossmann data validation failed.")
    else:
        print(f"File not found: {train_csv}")
