"""
Feature Engineering Agent - Creates features for ML models
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from sklearn.preprocessing import StandardScaler
import joblib

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureAgent:
    """Agent responsible for feature engineering and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.data_dir = config.DATA_DIR
        self.models_dir = config.MODELS_DIR
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: Input DataFrame with timestamp column
            
        Returns:
            DataFrame with time features
        """
        logger.info("Creating time-based features...")
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time of day categories
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        df['is_peak_solar'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'energy_output_kWh',
                           lags: list = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create lag features from target variable
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        logger.info(f"Creating lag features for {target_col}...")
        
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [6, 12, 24]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between weather variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        df = df.copy()
        
        # Weather interactions
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        df['wind_temp_interaction'] = df['wind_speed'] * df['temperature']
        df['solar_temp_interaction'] = df['solar_irradiance'] * df['temperature']
        
        # Polynomial features for key variables
        df['wind_speed_squared'] = df['wind_speed'] ** 2
        df['wind_speed_cubed'] = df['wind_speed'] ** 3
        df['solar_irradiance_squared'] = df['solar_irradiance'] ** 2
        
        # Derived features
        df['apparent_temp'] = df['temperature'] - 0.4 * (df['temperature'] - 10) * (1 - df['humidity'] / 100)
        df['wind_chill'] = 13.12 + 0.6215 * df['temperature'] - 11.37 * (df['wind_speed'] ** 0.16) + 0.3965 * df['temperature'] * (df['wind_speed'] ** 0.16)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing features...")
        
        df = df.copy()
        
        # Columns to normalize (exclude timestamp and target)
        exclude_cols = ['timestamp', 'energy_output_kWh']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
        
        if fit:
            df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
            # Save scaler
            scaler_path = self.models_dir / 'scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        else:
            df[cols_to_normalize] = self.scaler.transform(df[cols_to_normalize])
        
        return df
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            df: Input DataFrame
            train_ratio: Ratio of training data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Splitting data with train ratio: {train_ratio}")
        
        # Remove rows with NaN (from lag features)
        df = df.dropna()
        
        # Time-based split (no shuffling for time series)
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
        
        return train_df, test_df
    
    def run(self, data: pd.DataFrame) -> Dict:
        """
        Execute the feature engineering agent
        
        Args:
            data: Input DataFrame from DataAgent
            
        Returns:
            Dictionary with processed data
        """
        logger.info("FeatureAgent: Starting feature engineering...")
        
        # Create time features
        df = self.create_time_features(data)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Split data before normalization
        train_df, test_df = self.split_data(df)
        
        # Normalize features (fit on train, transform both)
        train_df = self.normalize_features(train_df, fit=True)
        test_df = self.normalize_features(test_df, fit=False)
        
        # Save processed data
        train_path = self.data_dir / 'train_data.csv'
        test_path = self.data_dir / 'test_data.csv'
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Processed data saved. Train: {train_path}, Test: {test_path}")
        
        return {
            "status": "success",
            "message": f"Feature engineering completed. Train: {len(train_df)}, Test: {len(test_df)}",
            "train_data": train_df,
            "test_data": test_df,
            "train_path": str(train_path),
            "test_path": str(test_path),
            "feature_count": len(train_df.columns) - 2,  # Exclude timestamp and target
            "features": list(train_df.columns)
        }


if __name__ == "__main__":
    # Test the agent
    from agents.data_agent import DataAgent
    
    data_agent = DataAgent()
    data_result = data_agent.run(days=30)
    
    feature_agent = FeatureAgent()
    result = feature_agent.run(data_result['data'])
    
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Feature count: {result['feature_count']}")
