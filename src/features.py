from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from src.core import setup_logger

logger = setup_logger(__name__)

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class DateTransformation(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying date transformation.")
        df_transformed = df.copy()
        date_col = 'date' if 'date' in df.columns else 'Date'
        if date_col not in df.columns:
            return df
        df_transformed[date_col] = pd.to_datetime(df[date_col])
        df_transformed['Year'] = df_transformed[date_col].dt.year
        df_transformed['Month'] = df_transformed[date_col].dt.month
        df_transformed['Day'] = df_transformed[date_col].dt.day
        df_transformed['DayOfWeek'] = df_transformed[date_col].dt.dayofweek
        df_transformed['IsWeekend'] = (df_transformed[date_col].dt.dayofweek >= 5).astype(int)
        df_transformed['DayOfMonth'] = df_transformed[date_col].dt.day
        return df_transformed

class FourierSeriesSeasonality(FeatureEngineeringStrategy):
    def __init__(self, period: float = 365.25, order: int = 3):
        self.period = period
        self.order = order

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Applying Fourier terms (order={self.order})")
        df_transformed = df.copy()
        date_col = 'date' if 'date' in df.columns else 'Date'
        times = pd.to_datetime(df_transformed[date_col]).values.view(np.int64) / 10**9 / (60 * 60 * 24)
        for i in range(1, self.order + 1):
            df_transformed[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * times / self.period)
            df_transformed[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * times / self.period)
        return df_transformed

class EasterFeature(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying Easter feature.")
        df_transformed = df.copy()
        date_col = 'date' if 'date' in df.columns else 'Date'
        dates = pd.to_datetime(df_transformed[date_col])
        easter_dates = {2013: '2013-03-31', 2014: '2014-04-20', 2015: '2015-04-05', 2016: '2016-03-27'}
        df_transformed['days_to_easter'] = 999
        for year, date_str in easter_dates.items():
            mask = dates.dt.year == year
            df_transformed.loc[mask, 'days_to_easter'] = (dates[mask] - pd.to_datetime(date_str)).dt.days
        df_transformed['easter_effect'] = ((df_transformed['days_to_easter'] >= -7) & (df_transformed['days_to_easter'] <= 7)).astype(int)
        return df_transformed

class RossmannFeatureEngineering(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying Rossmann retail features.")
        df_transformed = df.copy()
        if 'StateHoliday' in df_transformed.columns:
            df_transformed['StateHoliday'] = df_transformed['StateHoliday'].astype(str).map({'0': 0, 'a': 1, 'b': 2, 'c': 3}).fillna(0)
        if 'CompetitionDistance' in df_transformed.columns:
            df_transformed['CompetitionDistance'] = df_transformed['CompetitionDistance'].fillna(100000)
        if 'CompetitionOpenSinceYear' in df_transformed.columns and 'Year' in df_transformed.columns:
            df_transformed['CompetitionOpenTime'] = 12 * (df_transformed['Year'] - df_transformed['CompetitionOpenSinceYear']) + (df_transformed['Month'] - df_transformed['CompetitionOpenSinceMonth'])
            df_transformed['CompetitionOpenTime'] = df_transformed['CompetitionOpenTime'].apply(lambda x: x if x > 0 else 0)
        return df_transformed

class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy
    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._strategy.apply_transformation(df)
