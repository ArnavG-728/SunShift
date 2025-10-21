"""
Forecast Agent - Trains and generates predictions using ML models
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
import joblib
from pathlib import Path

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastAgent:
    """Agent responsible for training models and generating forecasts"""
    
    def __init__(self, model_type: str = "LSTM"):
        self.model_type = model_type
        self.models_dir = config.MODELS_DIR
        self.model = None
        
    def prepare_sequences(self, data: np.ndarray, seq_length: int = 24) -> tuple:
        """Prepare sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    def train_improved_lstm(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None) -> Dict:
        """Train improved LSTM with bias correction and non-negative constraints"""
        from models.improved_forecaster import ImprovedForecaster
        
        logger.info("Training IMPROVED LSTM model with bias correction...")
        
        # Create improved forecaster
        forecaster = ImprovedForecaster(sequence_length=config.LSTM_PARAMS['sequence_length'])
        
        # Train
        result = forecaster.train(train_data, val_data)
        
        # Save model
        model_path = self.models_dir / 'improved_lstm_model'
        forecaster.save(str(model_path))
        
        self.model = forecaster
        
        logger.info(f"✓ Improved model trained with bias correction: {result['bias_correction']:.4f}")
        
        return result
    
    def train_lstm(self, train_data: pd.DataFrame) -> Dict:
        """Train improved LSTM model with better architecture"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        logger.info("Training improved LSTM model...")
        
        # Prepare data
        feature_cols = [col for col in train_data.columns 
                       if col not in ['timestamp', 'energy_output_kWh']]
        X = train_data[feature_cols].values
        y = train_data['energy_output_kWh'].values
        
        seq_length = config.LSTM_PARAMS['sequence_length']
        X_seq, y_seq = self.prepare_sequences(X, seq_length)
        
        # Build improved model with Bidirectional LSTM
        model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(config.LSTM_PARAMS['units'], return_sequences=True), 
                         input_shape=(seq_length, X.shape[1])),
            BatchNormalization(),
            Dropout(config.LSTM_PARAMS['dropout']),
            
            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(config.LSTM_PARAMS['units'] // 2, return_sequences=True)),
            BatchNormalization(),
            Dropout(config.LSTM_PARAMS['dropout']),
            
            # Third LSTM layer
            LSTM(config.LSTM_PARAMS['units'] // 4),
            BatchNormalization(),
            Dropout(config.LSTM_PARAMS['dropout'] * 0.5),
            
            # Dense layers with residual-like connections
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        # Use Adam optimizer with custom learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train with callbacks
        history = model.fit(
            X_seq, y_seq,
            epochs=config.LSTM_PARAMS['epochs'],
            batch_size=config.LSTM_PARAMS['batch_size'],
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model in new Keras format
        model_path = self.models_dir / 'lstm_model.keras'
        model.save(model_path)
        self.model = model
        
        return {
            "model_path": str(model_path),
            "loss": float(history.history['loss'][-1]),
            "val_loss": float(history.history['val_loss'][-1])
        }
    
    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions"""
        logger.info(f"Generating predictions using {self.model_type}...")
        
        feature_cols = [col for col in test_data.columns 
                       if col not in ['timestamp', 'energy_output_kWh']]
        
        if self.model_type == "LSTM":
            from tensorflow.keras.models import load_model
            model_path = self.models_dir / 'lstm_model.keras'
            model = load_model(model_path, compile=True)
            
            X = test_data[feature_cols].values
            seq_length = config.LSTM_PARAMS['sequence_length']
            X_seq, _ = self.prepare_sequences(X, seq_length)
            
            predictions = model.predict(X_seq, verbose=0)
            
            # Create results DataFrame
            results = test_data.iloc[seq_length:].copy()
            results['predicted_output_kWh'] = predictions.flatten()
        
        return results
    
    def run(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """Execute forecasting agent with IMPROVED model"""
        logger.info("ForecastAgent: Starting IMPROVED model training...")
        
        # Split train data for validation
        split_idx = int(len(train_data) * 0.8)
        train_subset = train_data.iloc[:split_idx]
        val_subset = train_data.iloc[split_idx:]
        
        # Train IMPROVED model with bias correction
        logger.info("Using IMPROVED forecaster with bias correction and non-negative constraints...")
        train_result = self.train_improved_lstm(train_subset, val_subset)
        
        # Generate predictions using improved model
        if hasattr(self.model, 'predict'):
            # Using ImprovedForecaster
            pred_values = self.model.predict(test_data)
            
            # Create results DataFrame
            results = test_data.iloc[self.model.sequence_length:].copy()
            results['predicted_output_kWh'] = pred_values
            predictions = results
        else:
            # Fallback to old method
            predictions = self.predict(test_data)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(predictions['energy_output_kWh'], 
                                  predictions['predicted_output_kWh'])
        rmse = np.sqrt(mean_squared_error(predictions['energy_output_kWh'], 
                                          predictions['predicted_output_kWh']))
        
        # Fix timestamps to be current/future dates
        from datetime import datetime, timedelta
        current_time = datetime.now()
        predictions['timestamp'] = [current_time + timedelta(hours=i) for i in range(len(predictions))]
        
        # Save predictions
        pred_path = config.DATA_DIR / 'predictions.csv'
        predictions.to_csv(pred_path, index=False)
        
        return {
            "status": "success",
            "model_type": self.model_type,
            "predictions": predictions,
            "metrics": {"mae": mae, "rmse": rmse},
            "predictions_path": str(pred_path)
        }
