import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import RossmannPipeline
from src.core import setup_logger

logger = setup_logger(__name__)

def generate_submission():
    """
    Generates the Kaggle submission file using the production model.
    """
    test_csv = os.path.abspath("data/raw/test.csv")
    model_path = os.path.abspath("models/rossmann_production_model.pkl")
    store_csv = os.path.abspath("data/raw/store.csv")
    
    if not os.path.exists(test_csv):
        logger.error(f"Test data not found at {test_csv}.")
        return
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Run production training first.")
        return

    logger.info("Initializing Submission Generation...")
    
    # We use RossmannPipeline to handle ingestion and feature engineering
    # Note: Test data needs to be merged with store data just like train data
    pipeline = RossmannPipeline(test_csv)
    
    # Load production model
    with open(model_path, 'rb') as f:
        pipeline.model = pickle.load(f)
        
    # 1. Ingest and Merge Test Data
    # The ingestor logic in RossmannDataIngestor already handles merge with store.csv 
    # if it's in the same directory as test.csv
    logger.info("Ingesting and merging test data...")
    df_test = pipeline.ingestor.ingest(test_csv)
    
    # 2. Run Feature Engineering
    # We need to preserve the 'Id' column for the submission
    logger.info("Running feature engineering on test data...")
    # RossmannPipeline.run_feature_engineering filters out 'Open' == 0 for Sales transform,
    # but for test data we need to predict 0 for 'Open' == 0 manually.
    
    # Separate open and closed stores
    df_open = df_test[df_test['Open'] != 0].copy()
    df_closed = df_test[df_test['Open'] == 0].copy()
    
    logger.info(f"Test data split: {len(df_open)} open, {len(df_closed)} closed.")
    
    # 3. Predict for Open Stores
    if len(df_open) > 0:
        df_open_feat = pipeline.run_feature_engineering(df_open)
        
        # Determine feature columns (must match training)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in ['StoreType', 'Assortment']:
            if col in df_open_feat.columns:
                df_open_feat[col] = le.fit_transform(df_open_feat[col].astype(str))

        feature_cols = [
            'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
            'Year', 'Month', 'Day', 'IsWeekend', 'DayOfMonth',
            'CompetitionDistance', 'CompetitionOpenTime', 'StoreType', 'Assortment'
        ] + [c for c in df_open_feat.columns if 'fourier' in c or 'easter' in c]
        
        # Fill missing values for test data
        X_test = df_open_feat[feature_cols].fillna(0)
        
        # Predict in log space and transform back
        y_pred_log = pipeline.model.predict(X_test)
        df_open_feat['Sales'] = np.expm1(y_pred_log)
        
        # Join back to get Sales for open stores
        res_open = df_open_feat[['Id', 'Sales']]
    else:
        res_open = pd.DataFrame(columns=['Id', 'Sales'])
        
    # 4. Handle Closed Stores (Sales = 0)
    res_closed = pd.DataFrame({
        'Id': df_closed['Id'],
        'Sales': 0.0
    })
    
    # 5. Combine and Sort
    submission = pd.concat([res_open, res_closed]).sort_values('Id')
    
    # 6. Final Formatting and Save
    submission['Id'] = submission['Id'].astype(int)
    submission['Sales'] = submission['Sales'].apply(lambda x: max(0, x)) # Ensure no negative sales
    
    output_path = 'data/output/submission.csv'
    os.makedirs('data/output', exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Total records: {len(submission)}")
    print(submission.head())

if __name__ == "__main__":
    generate_submission()
