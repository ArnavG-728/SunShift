"""
Solar Forecaster ML Model
LSTM-based deep learning model for solar energy forecasting
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import os

# TensorFlow imports with logging suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Input, Conv1D, MaxPooling1D, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


def custom_loss(y_true, y_pred):
    """
    Custom loss function that:
    1. Penalizes negative predictions heavily
    2. Uses Huber loss for robustness to outliers
    """
    # Huber loss (more robust to outliers than MSE)
    huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
    
    # Heavy penalty for negative predictions
    negative_penalty = K.mean(K.square(K.minimum(y_pred, 0.0))) * 10.0
    
    return huber + negative_penalty


class SolarForecasterML:
    """
    LSTM-based solar energy forecasting model.
    
    Features:
    - Bidirectional LSTM layers for temporal pattern learning
    - Attention-like mechanism through Conv1D layers
    - Multiple time horizons (24h hourly, 7d daily)
    - Physics-informed constraints (non-negative outputs)
    - Uncertainty quantification
    """
    
    # Feature columns used for training (in order)
    FEATURE_COLS = [
        'temperature', 'humidity', 'clouds', 'wind_speed', 'pressure',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'is_daytime', 'solar_declination', 'clearsky_ratio',
        'temp_humidity', 'cloud_wind',
        'solar_lag_1h', 'solar_lag_24h', 'temp_lag_1h', 'clouds_lag_1h',
        'solar_rolling_3h', 'solar_rolling_24h', 'temp_rolling_24h', 'clouds_rolling_6h',
        'solar_irradiance_wm2', 'clearsky_irradiance_wm2'
    ]
    
    TARGET_COL = 'energy_output_kWh'
    
    def __init__(
        self, 
        sequence_length: int = 24,
        model_dir: str = None
    ):
        """
        Initialize the forecaster.
        
        Args:
            sequence_length: Number of hours of historical data to use for prediction
            model_dir: Directory to save/load models
        """
        self.sequence_length = sequence_length
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent.parent / "models" / "ml_saved"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))  # Ensure positive outputs
        self.feature_cols = self.FEATURE_COLS.copy()
        self.bias_correction = 0.0
        self.is_trained = False
        
        # Model metadata
        self.metadata = {
            'trained_at': None,
            'training_samples': 0,
            'val_mae': 0.0,
            'val_rmse': 0.0,
            'location': {'lat': 0, 'lon': 0}
        }
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Input layer
            Input(shape=input_shape),
            
            # Conv1D for local pattern extraction
            Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
            BatchNormalization(),
            
            # First Bidirectional LSTM
            Bidirectional(LSTM(128, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second Bidirectional LSTM
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third LSTM (final sequence processing)
            LSTM(32),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            # Output layer with ReLU to ensure non-negative
            Dense(1, activation='relu')
        ])
        
        # Compile with custom loss
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss=custom_loss,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_future: bool = False
    ) -> pd.DataFrame:
        """
        Prepare and validate features for the model.
        
        Args:
            df: Input dataframe
            is_future: If True, indicates this is future data without targets
            
        Returns:
            DataFrame with required features
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add time features if missing
        if 'hour' not in df.columns:
            df['hour'] = df['timestamp'].dt.hour
        
        if 'hour_sin' not in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'day_of_year' not in df.columns:
            df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        if 'day_sin' not in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        if 'month' not in df.columns:
            df['month'] = df['timestamp'].dt.month
        
        if 'month_sin' not in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        if 'is_daytime' not in df.columns:
            df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        
        if 'solar_declination' not in df.columns:
            df['solar_declination'] = 23.45 * np.sin(np.radians(360 * (284 + df['day_of_year']) / 365))
        
        # Add interaction features if missing
        if 'temp_humidity' not in df.columns:
            df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
        
        if 'cloud_wind' not in df.columns:
            df['cloud_wind'] = df['clouds'] * df['wind_speed'] / 10
        
        # Add lagged features if missing
        if 'solar_lag_1h' not in df.columns:
            if 'solar_irradiance_wm2' in df.columns:
                df['solar_lag_1h'] = df['solar_irradiance_wm2'].shift(1).fillna(0)
                df['solar_lag_24h'] = df['solar_irradiance_wm2'].shift(24).fillna(0)
            else:
                df['solar_lag_1h'] = 0
                df['solar_lag_24h'] = 0
        
        if 'temp_lag_1h' not in df.columns:
            df['temp_lag_1h'] = df['temperature'].shift(1).fillna(df['temperature'])
        
        if 'clouds_lag_1h' not in df.columns:
            df['clouds_lag_1h'] = df['clouds'].shift(1).fillna(df['clouds'])
        
        # Rolling features
        if 'solar_rolling_3h' not in df.columns:
            if 'solar_irradiance_wm2' in df.columns:
                df['solar_rolling_3h'] = df['solar_irradiance_wm2'].rolling(3, min_periods=1).mean()
                df['solar_rolling_24h'] = df['solar_irradiance_wm2'].rolling(24, min_periods=1).mean()
            else:
                df['solar_rolling_3h'] = 0
                df['solar_rolling_24h'] = 0
        
        if 'temp_rolling_24h' not in df.columns:
            df['temp_rolling_24h'] = df['temperature'].rolling(24, min_periods=1).mean()
        
        if 'clouds_rolling_6h' not in df.columns:
            df['clouds_rolling_6h'] = df['clouds'].rolling(6, min_periods=1).mean()
        
        # Clearsky ratio
        if 'clearsky_ratio' not in df.columns:
            if 'clearsky_irradiance_wm2' in df.columns and 'solar_irradiance_wm2' in df.columns:
                df['clearsky_ratio'] = np.where(
                    df['clearsky_irradiance_wm2'] > 0,
                    df['solar_irradiance_wm2'] / df['clearsky_irradiance_wm2'],
                    0
                ).clip(0, 1.5)
            else:
                df['clearsky_ratio'] = 1.0
        
        # Fill missing columns with defaults
        for col in self.feature_cols:
            if col not in df.columns:
                logger.warning(f"Missing feature '{col}', filling with 0")
                df[col] = 0.0
        
        # Handle NaN values
        df = df.ffill().bfill().fillna(0)
        
        return df
    
    def create_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM input.
        
        Args:
            X: Feature array
            y: Target array (optional)
            
        Returns:
            Tuple of (X_sequences, y_values) or just X_sequences
        """
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            if y is not None:
                y_seq.append(y[i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        
        return X_seq, None
    
    def train(
        self, 
        train_data: pd.DataFrame,
        val_data: pd.DataFrame = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        location: Dict = None
    ) -> Dict:
        """
        Train the forecasting model.
        
        Args:
            train_data: Training dataset with features and target
            val_data: Validation dataset (optional, will split from train if not provided)
            epochs: Maximum training epochs
            batch_size: Training batch size
            early_stopping_patience: Patience for early stopping
            location: Dictionary with 'lat' and 'lon' for model metadata
            
        Returns:
            Training history and metrics
        """
        logger.info(f"Training solar forecaster with {len(train_data)} samples...")
        
        # Prepare features
        train_data = self.prepare_features(train_data)
        
        # Validate target column exists
        if self.TARGET_COL not in train_data.columns:
            raise ValueError(f"Target column '{self.TARGET_COL}' not found in training data")
        
        # Split validation if not provided
        if val_data is None:
            split_idx = int(len(train_data) * 0.8)
            val_data = train_data.iloc[split_idx:].copy()
            train_data = train_data.iloc[:split_idx].copy()
        else:
            val_data = self.prepare_features(val_data)
        
        logger.info(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")
        
        # Filter to only use available feature columns
        available_features = [c for c in self.feature_cols if c in train_data.columns]
        self.feature_cols = available_features
        
        logger.info(f"Using {len(self.feature_cols)} features: {self.feature_cols[:5]}...")
        
        # Extract features and target
        X_train = train_data[self.feature_cols].values
        y_train = train_data[self.TARGET_COL].values
        
        X_val = val_data[self.feature_cols].values
        y_val = val_data[self.TARGET_COL].values
        
        # Fit scalers on training data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_scaled)
        
        logger.info(f"Training sequences: {X_train_seq.shape}, Validation sequences: {X_val_seq.shape}")
        
        # Build model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self.build_model(input_shape)
        
        logger.info(f"Model built with input shape: {input_shape}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.model_dir / 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train
        logger.info("Starting training...")
        history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate bias correction
        y_val_pred_scaled = self.model.predict(X_val_seq, verbose=0)
        y_val_pred = self.scaler_y.inverse_transform(y_val_pred_scaled).flatten()
        y_val_actual = y_val[self.sequence_length:]
        
        self.bias_correction = np.mean(y_val_actual - y_val_pred)
        logger.info(f"Bias correction: {self.bias_correction:.4f}")
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(y_val_actual, y_val_pred + self.bias_correction)
        rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred + self.bias_correction))
        mape = np.mean(np.abs((y_val_actual - y_val_pred - self.bias_correction) / (y_val_actual + 1e-10))) * 100
        
        logger.info(f"Validation MAE: {mae:.4f} kWh")
        logger.info(f"Validation RMSE: {rmse:.4f} kWh")
        logger.info(f"Validation MAPE: {mape:.2f}%")
        
        # Update metadata
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(train_data),
            'val_mae': float(mae),
            'val_rmse': float(rmse),
            'val_mape': float(mape),
            'location': location or {'lat': 0, 'lon': 0},
            'feature_cols': self.feature_cols,
            'sequence_length': self.sequence_length
        }
        
        self.is_trained = True
        
        return {
            'history': history.history,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'bias_correction': self.bias_correction
        }
    
    def predict(
        self, 
        data: pd.DataFrame,
        return_uncertainty: bool = False
    ) -> np.ndarray:
        """
        Generate predictions for given data.
        
        Args:
            data: Input data with required features
            return_uncertainty: If True, return (predictions, lower_bound, upper_bound)
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        # Prepare features
        data = self.prepare_features(data)
        
        # Extract features
        X = data[self.feature_cols].values
        
        # Scale
        X_scaled = self.scaler_X.transform(X)
        
        # Create sequences
        X_seq, _ = self.create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            logger.warning("Not enough data for sequence creation")
            return np.zeros(len(data))
        
        # Predict
        y_pred_scaled = self.model.predict(X_seq, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        # Apply bias correction and ensure non-negative
        y_pred_corrected = np.maximum(y_pred + self.bias_correction, 0)
        
        if return_uncertainty:
            # Estimate uncertainty (15% relative uncertainty)
            uncertainty = y_pred_corrected * 0.15
            lower = np.maximum(y_pred_corrected - uncertainty, 0)
            upper = y_pred_corrected + uncertainty
            return y_pred_corrected, lower, upper
        
        return y_pred_corrected
    
    def predict_future(
        self, 
        historical_data: pd.DataFrame,
        future_weather: pd.DataFrame,
        lat: float = 28.6139,
        lon: float = 77.2090
    ) -> pd.DataFrame:
        """
        Predict future energy output given weather forecast.
        
        Args:
            historical_data: Recent historical data (at least sequence_length hours)
            future_weather: Future weather forecast data
            lat: Latitude for physics calculations
            lon: Longitude for physics calculations
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        logger.info(f"Predicting {len(future_weather)} future hours...")
        
        # Prepare historical data
        historical = self.prepare_features(historical_data.copy())
        
        # Prepare future weather
        future = future_weather.copy()
        future['timestamp'] = pd.to_datetime(future['timestamp'])
        
        # Add physics-based solar irradiance if not present
        if 'solar_irradiance_wm2' not in future.columns:
            from .data_collector import SolarDataCollector
            collector = SolarDataCollector()
            future = collector._add_physics_solar(future, lat)
        
        # Prepare features for future data
        future = self.prepare_features(future, is_future=True)
        
        # Combine historical and future for sequence generation
        combined = pd.concat([historical.tail(self.sequence_length), future], ignore_index=True)
        
        # Iterative prediction
        predictions = []
        
        for i in range(len(future)):
            # Get the sequence ending at current position
            start_idx = i
            end_idx = start_idx + self.sequence_length
            
            if end_idx > len(combined):
                break
            
            sequence_data = combined.iloc[start_idx:end_idx]
            
            # Extract features
            X = sequence_data[self.feature_cols].values
            X_scaled = self.scaler_X.transform(X)
            X_seq = X_scaled.reshape(1, self.sequence_length, -1)
            
            # Predict
            y_pred_scaled = self.model.predict(X_seq, verbose=0)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()[0]
            y_pred = max(0, y_pred + self.bias_correction)
            
            predictions.append({
                'timestamp': future.iloc[i]['timestamp'],
                'predicted_output_kWh': y_pred,
                'confidence_lower': y_pred * 0.85,
                'confidence_upper': y_pred * 1.15
            })
            
            # Update combined data with prediction for next iteration
            combined.loc[combined.index[end_idx - 1], 'energy_output_kWh'] = y_pred
        
        result_df = pd.DataFrame(predictions)
        
        # Add weather data back
        result_df = pd.merge(
            result_df, 
            future[['timestamp', 'temperature', 'humidity', 'clouds', 'wind_speed', 
                   'solar_irradiance_wm2']].rename(columns={'solar_irradiance_wm2': 'solar_irradiance'}),
            on='timestamp',
            how='left'
        )
        
        logger.info(f"Predicted {len(result_df)} future hours. Range: {result_df['predicted_output_kWh'].min():.2f} - {result_df['predicted_output_kWh'].max():.2f} kWh")
        
        return result_df
    
    def save(self, name: str = "solar_forecaster"):
        """Save the trained model and associated objects."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Save Keras model
        model_path = self.model_dir / f"{name}.keras"
        self.model.save(model_path)
        
        # Save scalers and metadata
        artifacts = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_cols': self.feature_cols,
            'sequence_length': self.sequence_length,
            'bias_correction': self.bias_correction,
            'metadata': self.metadata
        }
        
        artifacts_path = self.model_dir / f"{name}_artifacts.pkl"
        joblib.dump(artifacts, artifacts_path)
        
        logger.info(f"✓ Model saved to {model_path}")
        logger.info(f"✓ Artifacts saved to {artifacts_path}")
    
    def load(self, name: str = "solar_forecaster") -> bool:
        """Load a saved model."""
        model_path = self.model_dir / f"{name}.keras"
        artifacts_path = self.model_dir / f"{name}_artifacts.pkl"
        
        if not model_path.exists() or not artifacts_path.exists():
            logger.warning(f"Model not found at {model_path}")
            return False
        
        try:
            # Load Keras model
            self.model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
            
            # Load artifacts
            artifacts = joblib.load(artifacts_path)
            self.scaler_X = artifacts['scaler_X']
            self.scaler_y = artifacts['scaler_y']
            self.feature_cols = artifacts['feature_cols']
            self.sequence_length = artifacts['sequence_length']
            self.bias_correction = artifacts['bias_correction']
            self.metadata = artifacts['metadata']
            
            self.is_trained = True
            
            logger.info(f"✓ Model loaded from {model_path}")
            logger.info(f"  Trained at: {self.metadata.get('trained_at', 'Unknown')}")
            logger.info(f"  Val MAE: {self.metadata.get('val_mae', 0):.4f} kWh")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            'is_trained': self.is_trained,
            'sequence_length': self.sequence_length,
            'n_features': len(self.feature_cols),
            'feature_cols': self.feature_cols,
            'bias_correction': self.bias_correction,
            **self.metadata
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test model building
    forecaster = SolarForecasterML(sequence_length=24)
    
    # Create dummy data for testing
    print("Creating dummy training data...")
    n_samples = 1000
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='h')
    
    dummy_data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.random.normal(25, 5, n_samples),
        'humidity': np.random.normal(60, 10, n_samples),
        'clouds': np.random.uniform(0, 100, n_samples),
        'wind_speed': np.random.normal(5, 2, n_samples),
        'pressure': np.random.normal(1013, 5, n_samples),
        'solar_irradiance_wm2': np.maximum(0, np.random.normal(400, 200, n_samples)),
        'clearsky_irradiance_wm2': np.maximum(0, np.random.normal(500, 200, n_samples)),
        'energy_output_kWh': np.maximum(0, np.random.normal(1.5, 0.8, n_samples))
    })
    
    print("Training model...")
    result = forecaster.train(dummy_data, epochs=5, batch_size=32)
    
    print(f"\n✓ Training complete!")
    print(f"  MAE: {result['mae']:.4f} kWh")
    print(f"  RMSE: {result['rmse']:.4f} kWh")
    
    # Test prediction
    predictions = forecaster.predict(dummy_data.tail(100))
    print(f"\n✓ Predictions generated: {len(predictions)} values")
    print(f"  Range: {predictions.min():.2f} - {predictions.max():.2f} kWh")
    
    # Save and load test
    forecaster.save("test_model")
    
    new_forecaster = SolarForecasterML()
    loaded = new_forecaster.load("test_model")
    print(f"\n✓ Model loaded: {loaded}")
    print(f"  Info: {new_forecaster.get_model_info()}")
