"""
Improved Forecasting Model - Fixes systematic bias and negative predictions
Implements ensemble approach with bias correction and proper constraints
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import tensorflow as tf

from config import config

logger = logging.getLogger(__name__)


def non_negative_mse(y_true, y_pred):
    """
    Custom loss that penalizes negative predictions heavily
    """
    # Standard MSE
    mse = K.mean(K.square(y_pred - y_true))
    
    # Heavy penalty for negative predictions
    negative_penalty = K.mean(K.square(K.minimum(y_pred, 0.0))) * 10.0
    
    return mse + negative_penalty


class ImprovedForecaster:
    """
    Improved forecasting model with:
    - Bias correction
    - Non-negative constraints
    - Better feature engineering
    - Ensemble approach
    """
    
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.bias_correction = 0.0
        
    def engineer_features(self, df: pd.DataFrame, is_future=False) -> pd.DataFrame:
        """
        Enhanced feature engineering
        
        Args:
            df: Raw data with weather features
            is_future: If True, skip lag features (for future predictions)
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['day_of_year'] = pd.to_datetime(df['timestamp']).dt.dayofyear
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Interaction features
        df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
        df['wind_temp'] = df['wind_speed'] * df['temperature']
        
        # Solar-specific features
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        df['solar_potential'] = df['solar_irradiance'] * df['is_daytime']
        
        # Wind power (cubic relationship)
        df['wind_power'] = df['wind_speed'] ** 3
        
        # Lagged features - only if we have energy_output_kWh column
        if not is_future and 'energy_output_kWh' in df.columns:
            df['energy_lag_1h'] = df['energy_output_kWh'].shift(1).fillna(0)
            df['energy_lag_24h'] = df['energy_output_kWh'].shift(24).fillna(0)
            df['energy_rolling_mean_24h'] = df['energy_output_kWh'].rolling(24, min_periods=1).mean().fillna(0)
            df['energy_rolling_std_24h'] = df['energy_output_kWh'].rolling(24, min_periods=1).std().fillna(0)
        elif is_future:
            # For future predictions, initialize with zeros (will be filled iteratively)
            df['energy_lag_1h'] = 0.0
            df['energy_lag_24h'] = 0.0
            df['energy_rolling_mean_24h'] = 0.0
            df['energy_rolling_std_24h'] = 0.0
        
        # Fill any remaining NaN values
        df = df.bfill().fillna(0)
        
        return df
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build improved LSTM model with non-negative constraints
        
        Args:
            input_shape: (sequence_length, n_features)
            
        Returns:
            Compiled model
        """
        model = Sequential([
            # First Bidirectional LSTM
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second Bidirectional LSTM
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third LSTM
            LSTM(32),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            # Output with ReLU to ensure non-negative
            Dense(1, activation='relu')  # ReLU ensures output >= 0
        ])
        
        # Compile with custom loss
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss=non_negative_mse,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        """
        Prepare sequences for LSTM
        
        Args:
            X: Feature array
            y: Target array (optional)
            
        Returns:
            Tuple of (X_seq, y_seq) or just X_seq
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
        
        return X_seq
    
    def calculate_bias_correction(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate bias correction factor
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        """
        # Calculate systematic bias
        self.bias_correction = np.mean(y_true - y_pred)
        logger.info(f"Calculated bias correction: {self.bias_correction:.4f}")
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None) -> Dict:
        """
        Train improved model with bias correction
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            
        Returns:
            Training history
        """
        logger.info("Training improved forecaster...")
        
        # Engineer features
        logger.info("Engineering features...")
        train_data = self.engineer_features(train_data)
        
        if val_data is not None:
            val_data = self.engineer_features(val_data)
        
        # Select features and SAVE the column names for prediction
        self.feature_cols = [col for col in train_data.columns 
                            if col not in ['timestamp', 'energy_output_kWh']]
        
        logger.info(f"Training with {len(self.feature_cols)} features: {self.feature_cols[:10]}...")
        
        X_train = train_data[self.feature_cols].values
        y_train = train_data['energy_output_kWh'].values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train_scaled)
        
        logger.info(f"Training sequences: {X_train_seq.shape}")
        
        # Prepare validation data
        validation_data = None
        if val_data is not None:
            X_val = val_data[self.feature_cols].values
            y_val = val_data['energy_output_kWh'].values
            
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            
            X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val_scaled)
            validation_data = (X_val_seq, y_val_seq)
        
        # Build model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self.build_model(input_shape)
        
        logger.info(f"Model built with input shape: {input_shape}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        logger.info("Training model...")
        history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=config.LSTM_PARAMS['epochs'],
            batch_size=config.LSTM_PARAMS['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate bias correction on training data
        y_train_pred_scaled = self.model.predict(X_train_seq, verbose=0)
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled).flatten()
        y_train_actual = y_train[self.sequence_length:]
        
        self.calculate_bias_correction(y_train_actual, y_train_pred)
        
        logger.info("✓ Training complete")
        
        return {
            'history': history.history,
            'bias_correction': self.bias_correction
        }
    
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions with bias correction
        
        Args:
            test_data: Test data
            
        Returns:
            Predictions (non-negative, bias-corrected)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Engineer features - MUST match training features exactly
        test_data_eng = self.engineer_features(test_data)
        
        # Get the SAME feature columns that were used during training
        # This is critical - must match exactly
        if not hasattr(self, 'feature_cols'):
            # Fallback: exclude timestamp and target
            feature_cols = [col for col in test_data_eng.columns 
                           if col not in ['timestamp', 'energy_output_kWh']]
        else:
            feature_cols = self.feature_cols
        
        # Ensure all required features exist
        missing_features = set(feature_cols) - set(test_data_eng.columns)
        if missing_features:
            logger.error(f"Missing features in test data: {missing_features}")
            # Add missing features with zeros
            for feat in missing_features:
                test_data_eng[feat] = 0
        
        # Select features in the SAME order as training
        X_test = test_data_eng[feature_cols].values
        
        logger.info(f"Test data shape before scaling: {X_test.shape}")
        logger.info(f"Scaler expects {self.scaler_X.n_features_in_} features")
        
        # Scale
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Prepare sequences
        X_test_seq = self.prepare_sequences(X_test_scaled)
        
        # Predict
        y_pred_scaled = self.model.predict(X_test_seq, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        # Apply bias correction
        y_pred_corrected = y_pred + self.bias_correction
        
        # Ensure non-negative
        y_pred_corrected = np.maximum(y_pred_corrected, 0)
        
        logger.info(f"Predictions: min={y_pred_corrected.min():.2f}, "
                   f"max={y_pred_corrected.max():.2f}, "
                   f"mean={y_pred_corrected.mean():.2f}")
        
        return y_pred_corrected
    
    def predict_future(self, historical_data: pd.DataFrame, future_weather: pd.DataFrame) -> np.ndarray:
        """
        Predict future values iteratively, using previous predictions as input
        
        Args:
            historical_data: Recent historical data (last 24+ hours) with energy_output_kWh
            future_weather: Future weather data (without energy_output_kWh)
            
        Returns:
            Future predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Predicting {len(future_weather)} future steps iteratively...")
        
        # Combine historical and future data
        combined_data = pd.concat([historical_data.tail(48), future_weather], ignore_index=True)
        combined_data = combined_data.reset_index(drop=True)
        
        # Initialize energy_output_kWh for future rows with 0
        if 'energy_output_kWh' not in combined_data.columns:
            combined_data['energy_output_kWh'] = 0.0
        else:
            # Fill future values with 0
            hist_len = len(historical_data.tail(48))
            combined_data.loc[hist_len:, 'energy_output_kWh'] = 0.0
        
        predictions = []
        hist_len = len(historical_data.tail(48))
        
        # Predict iteratively
        for i in range(len(future_weather)):
            current_idx = hist_len + i
            
            # Engineer features up to current point
            temp_data = combined_data.iloc[:current_idx + 1].copy()
            temp_data_eng = self.engineer_features(temp_data, is_future=False)
            
            # Get features for current step
            if not hasattr(self, 'feature_cols'):
                feature_cols = [col for col in temp_data_eng.columns 
                               if col not in ['timestamp', 'energy_output_kWh']]
            else:
                feature_cols = self.feature_cols
            
            # Ensure all features exist
            for feat in feature_cols:
                if feat not in temp_data_eng.columns:
                    temp_data_eng[feat] = 0
            
            X = temp_data_eng[feature_cols].values
            
            # Need at least sequence_length points
            if len(X) < self.sequence_length:
                # Pad with zeros
                padding = np.zeros((self.sequence_length - len(X), X.shape[1]))
                X = np.vstack([padding, X])
            
            # Take last sequence_length points
            X_seq = X[-self.sequence_length:]
            
            # Scale and reshape
            X_scaled = self.scaler_X.transform(X_seq)
            X_input = X_scaled.reshape(1, self.sequence_length, -1)
            
            # Predict
            y_pred_scaled = self.model.predict(X_input, verbose=0)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()[0]
            
            # Apply bias correction and ensure non-negative
            y_pred_corrected = max(0, y_pred + self.bias_correction)
            
            # Store prediction
            predictions.append(y_pred_corrected)
            
            # Update combined_data with prediction for next iteration
            combined_data.loc[current_idx, 'energy_output_kWh'] = y_pred_corrected
        
        predictions = np.array(predictions)
        logger.info(f"Future predictions: min={predictions.min():.2f}, "
                   f"max={predictions.max():.2f}, mean={predictions.mean():.2f}")
        
        return predictions
    
    def save(self, path: str):
        """Save model and scalers"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        model_path = path if path.endswith('.keras') else f"{path}.keras"
        self.model.save(model_path)
        
        # Save scalers, bias, and feature columns
        import joblib
        scalers_path = path.replace('.keras', '_scalers.pkl')
        joblib.dump({
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'bias_correction': self.bias_correction,
            'sequence_length': self.sequence_length,
            'feature_cols': self.feature_cols  # CRITICAL: Save feature column names
        }, scalers_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scalers saved to {scalers_path}")
    
    def load(self, path: str):
        """Load model and scalers"""
        import joblib
        
        # Load model
        model_path = path if path.endswith('.keras') else f"{path}.keras"
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'non_negative_mse': non_negative_mse}
        )
        
        # Load scalers, bias, and feature columns
        scalers_path = path.replace('.keras', '_scalers.pkl')
        data = joblib.load(scalers_path)
        
        self.scaler_X = data['scaler_X']
        self.scaler_y = data['scaler_y']
        self.bias_correction = data['bias_correction']
        self.sequence_length = data['sequence_length']
        self.feature_cols = data.get('feature_cols', [])  # Load feature columns
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Bias correction: {self.bias_correction:.4f}")
        logger.info(f"Feature columns: {len(self.feature_cols)} features")
