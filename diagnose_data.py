import pandas as pd
import os

def diagnose():
    train_path = "data/tabular-playground-series-jan-2022/train.csv"
    holidays_path = "data/nordic_holidays.csv"
    
    train = pd.read_csv(train_path)
    print(f"Train head:\n{train.head()}")
    print(f"Train shape: {train.shape}")
    print(f"Train date types:\n{train['date'].dtype}")
    
    holidays = pd.read_csv(holidays_path)
    print(f"Holidays head:\n{holidays.head()}")
    print(f"Holidays shape: {holidays.shape}")
    print(f"Holidays date types:\n{holidays['date'].dtype}")
    
    # Simulate current ingestor logic
    train['date'] = pd.to_datetime(train['date'])
    holidays['date'] = pd.to_datetime(holidays['date'])
    
    merged = pd.merge(train, holidays, on=['date', 'country'], how='left')
    print(f"Merged shape: {merged.shape}")
    print(f"Merged head:\n{merged.head()}")
    print(f"Holiday counts:\n{merged['holiday'].value_counts(dropna=False)}")

    # Check for empty cleaning
    cleaned = merged.dropna(axis=0)
    print(f"Cleaned shape: {cleaned.shape}")

if __name__ == "__main__":
    diagnose()
