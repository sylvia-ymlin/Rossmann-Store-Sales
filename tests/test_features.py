import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import (
    FeatureEngineer, DateTransformation, FourierSeriesSeasonality, EasterFeature
)

def test_date_transformation():
    df = pd.DataFrame({'Date': ['2015-01-01', '2015-01-02']})
    eng = FeatureEngineer(DateTransformation())
    df_transformed = eng.apply_feature_engineering(df)
    
    assert 'Year' in df_transformed.columns
    assert 'Month' in df_transformed.columns
    assert df_transformed['Year'].iloc[0] == 2015
    assert df_transformed['Month'].iloc[0] == 1
    assert df_transformed['DayOfWeek'].iloc[0] == 3 # 2015-01-01 was Thursday

def test_fourier_seasonality():
    # Create a 1-year range
    dates = pd.date_range(start='2015-01-01', periods=366, freq='D')
    df = pd.DataFrame({'Date': dates})
    
    order = 3
    eng = FeatureEngineer(FourierSeriesSeasonality(period=365.25, order=order))
    df_transformed = eng.apply_feature_engineering(df)
    
    # Check if all terms exist
    for i in range(1, order + 1):
        assert f'fourier_sin_{i}' in df_transformed.columns
        assert f'fourier_cos_{i}' in df_transformed.columns
    
    # Check periodicity roughly (first and last day of a year should be similar)
    # 2015-01-01 and 2016-01-01
    assert np.allclose(df_transformed['fourier_sin_1'].iloc[0], 
                       df_transformed['fourier_sin_1'].iloc[365], atol=0.1)

def test_easter_feature():
    # 2014-04-20 was Easter
    df = pd.DataFrame({'Date': ['2014-04-13', '2014-04-20', '2014-04-27']})
    eng = FeatureEngineer(EasterFeature())
    df_transformed = eng.apply_feature_engineering(df)
    
    # Easter day (index 1) should have easter_effect = 1
    assert df_transformed['easter_effect'].iloc[1] == 1
    # 7 days before (index 0) should be included in window (e.g., +/- 7 days)
    assert df_transformed['easter_effect'].iloc[0] == 1
    # 7 days after (index 2) should be included
    assert df_transformed['easter_effect'].iloc[2] == 1
    
    # Check days_to_easter
    assert df_transformed['days_to_easter'].iloc[1] == 0

if __name__ == "__main__":
    # Manual run if pytest not used
    test_date_transformation()
    test_fourier_seasonality()
    test_easter_feature()
    print("All feature engineering tests passed!")
