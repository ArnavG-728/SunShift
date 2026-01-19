"""
V2 Multi-Location LSTM Forecaster
Trains on real solar data from multiple geographic locations
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Attention, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

logger = logging.getLogger(__name__)


def non_negative_mse(y_true, y_pred):
    """Custom loss that penalizes negative predictions"""
    mse = K.mean(K.square(y_pred - y_true))
    negative_penalty = K.mean(K.square(K.minimum(y_pred, 0.0))) * 10.0
    return mse + negative_penalty


def encode_location(lat: float, lon: float) -> Dict[str, float]:
    """
    Encode lat/lon using sine/cosine for continuity
    This ensures that -179° and +179° longitude are treated as close
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    return {
        'lat_sin': np.sin(lat_rad),
        'lat_cos': np.cos(lat_rad),
        'lon_sin': np.sin(lon_rad),
        'lon_cos': np.cos(lon_rad),
        'abs_lat': abs(lat)  # Distance from equator
    }


def classify_climate_zone(lat: float) -> str:
    """
    Köppen climate classification (simplified)
    Used for one-hot encoding in the model
    """
    abs_lat = abs(lat)
    if abs_lat < 15:
        return 'tropical'  # Af, Am, Aw
    elif abs_lat < 35:
        return 'subtropical'  # Cfa, Csa
    elif abs_lat < 55:
        return 'temperate'  # Cfb, Dfb
    else:
        return 'polar'  # ET, EF


class MultiLocationForecaster:
    """
    V2 LSTM Forecaster with Location Awareness
    
    Key Improvements over V1:
    - Trained on real data from multiple locations
    - Location embeddings (lat/lon encoded as sin/cos)
    - Climate zone classification
    - Attention mechanism for temporal focus
    - Transfer learning support
    """
    
    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.bias_correction = 0.0
        self.feature_cols = []
        
    def engineer_features(self, df: pd.DataFrame, lat: float, lon: float, 
                         is_future: bool = False) -> pd.DataFrame:
        """
        Enhanced feature engineering with location awareness
        
        Args:
            df: Raw data with weather features
            lat: Latitude
            lon: Longitude
            is_future: If True, skip lag features
            
        Returns:
            DataFrame with engineered features including location embeddings
        """
        df = df.copy()
        
        # ===== TIME-BASED FEATURES =====
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['day_of_year'] = pd.to_datetime(df['timestamp']).dt.dayofyear
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # ===== WEATHER INTERACTION FEATURES =====
        df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
        df['wind_temp'] = df['wind_speed'] * df['temperature']
        
        # ===== SOLAR-SPECIFIC FEATURES =====
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        df['solar_potential'] = df['solar_irradiance'] * df['is_daytime']
        df['wind_power'] = df['wind_speed'] ** 3
        
        # ===== LOCATION EMBEDDINGS (NEW IN V2) =====
        loc_encoding = encode_location(lat, lon)
        for key, value in loc_encoding.items():
            df[key] = value
        
        # ===== CLIMATE ZONE (NEW IN V2) =====
        climate_zone = classify_climate_zone(lat)
        df['climate_tropical'] = 1 if climate_zone == 'tropical' else 0
        df['climate_subtropical'] = 1 if climate_zone == 'subtropical' else 0
        df['climate_temperate'] = 1 if climate_zone == 'temperate' else 0
        df['climate_polar'] = 1 if climate_zone == 'polar' else 0
        
        # ===== SEASONAL INTERACTION (NEW IN V2) =====
        # How temperature deviates from climate zone baseline
        climate_baselines = {
            'tropical': 28, 'subtropical': 22, 'temperate': 15, 'polar': 5
        }
        baseline_temp = climate_baselines.get(climate_zone, 20)
        df['temp_deviation'] = df['temperature'] - baseline_temp
        
        # ===== LAGGED FEATURES =====
        if not is_future and 'energy_output_kWh' in df.columns:
            df['energy_lag_1h'] = df['energy_output_kWh'].shift(1).fillna(0)
            df['energy_lag_24h'] = df['energy_output_kWh'].shift(24).fillna(0)
            df['energy_rolling_mean_24h'] = df['energy_output_kWh'].rolling(24, min_periods=1).mean().fillna(0)
            df['energy_rolling_std_24h'] = df['energy_output_kWh'].rolling(24, min_periods=1).std().fillna(0)
        elif is_future:
            df['energy_lag_1h'] = 0.0
            df['energy_lag_24h'] = 0.0
            df['energy_rolling_mean_24h'] = 0.0
            df['energy_rolling_std_24h'] = 0.0
        
        # Fill any remaining NaN
        df = df.bfill().fillna(0)
        
        return df
    
    def build_model(self, weather_input_shape: Tuple[int, int]) -> Model:
        """
        Build location-aware LSTM with attention mechanism
        
        Args:
            weather_input_shape: (sequence_length, n_weather_features)
            
        Returns:
            Compiled Keras Model
        """
        # ===== INPUT LAYERS =====
        weather_input = Input(shape=weather_input_shape, name='weather_sequence')
        location_input = Input(shape=(5,), name='location_embedding')  # lat_sin, lat_cos, lon_sin, lon_cos, abs_lat
        
        # ===== WEATHER BRANCH (LSTM) =====
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(weather_input)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        
        # Attention mechanism (focus on relevant time steps per location)
        attention = Attention()([lstm_out, lstm_out])
        
        # Second LSTM
        lstm_out2 = Bidirectional(LSTM(64))(attention)
        lstm_out2 = BatchNormalization()(lstm_out2)
        lstm_out2 = Dropout(0.3)(lstm_out2)
        
        # ===== LOCATION BRANCH (DENSE) =====
        loc_dense = Dense(32, activation='relu')(location_input)
        loc_dense = BatchNormalization()(loc_dense)
        loc_dense = Dropout(0.2)(loc_dense)
        
        # ===== MERGE BRANCHES =====
        merged = Concatenate()([lstm_out2, loc_dense])
        
        # ===== FINAL LAYERS =====
        dense = Dense(64, activation='relu')(merged)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.2)(dense)
        
        dense2 = Dense(32, activation='relu')(dense)
        dense2 = Dropout(0.1)(dense2)
        
        # Output with ReLU to ensure non-negative
        output = Dense(1, activation='relu', name='energy_output')(dense2)
        
        # ===== COMPILE =====
        model = Model(inputs=[weather_input, location_input], outputs=output)
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss=non_negative_mse,
            metrics=['mae', 'mse']
        )
        
        logger.info("✓ Built Multi-Location LSTM with Attention")
        logger.info(f"  Weather input: {weather_input_shape}")
        logger.info(f"  Location input: (5,)")
        logger.info(f"  Total params: {model.count_params():,}")
        
        return model
    
    def prepare_sequences(self, X: np.ndarray, loc_features: np.ndarray, 
                         y: np.ndarray = None) -> Tuple:
        """
        Prepare sequences for multi-input LSTM
        
        Args:
            X: Weather feature array (n_samples, n_features)
            loc_features: Location features (n_samples, 5)
            y: Target array (optional)
            
        Returns:
            Tuple of (X_weather_seq, X_loc_seq, y_seq) or (X_weather_seq, X_loc_seq)
        """
        X_weather_seq = []
        X_loc_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(len(X) - self.sequence_length):
            X_weather_seq.append(X[i:i + self.sequence_length])
            X_loc_seq.append(loc_features[i + self.sequence_length])  # Use location at prediction time
            if y is not None:
                y_seq.append(y[i + self.sequence_length])
        
        X_weather_seq = np.array(X_weather_seq)
        X_loc_seq = np.array(X_loc_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_weather_seq, X_loc_seq, y_seq
        
        return X_weather_seq, X_loc_seq
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None) -> Dict:
        """
        Train the multi-location model
        
        Args:
            train_data: Training data with 'latitude' and 'longitude' columns
            val_data: Validation data (optional)
            
        Returns:
            Training history
        """
        logger.info("Training Multi-Location Forecaster V2...")
        
        # Check for location columns
        if 'latitude' not in train_data.columns or 'longitude' not in train_data.columns:
            raise ValueError("Training data must include 'latitude' and 'longitude' columns")
        
        # Engineer features for each unique location
        # (In production, this would be done in batches)
        logger.info("Engineering features with location awareness...")
        
        # For simplicity, assume single location in this batch
        # In production, you'd iterate over location groups
        lat = train_data['latitude'].iloc[0]
        lon = train_data['longitude'].iloc[0]
        
        train_data_eng = self.engineer_features(train_data, lat, lon)
        
        if val_data is not None:
            val_lat = val_data['latitude'].iloc[0]
            val_lon = val_data['longitude'].iloc[0]
            val_data_eng = self.engineer_features(val_data, val_lat, val_lon)
        
        # Select features
        exclude_cols = ['timestamp', 'energy_output_kWh', 'latitude', 'longitude']
        self.feature_cols = [col for col in train_data_eng.columns if col not in exclude_cols]
        
        # Separate location features
        loc_feature_names = ['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 'abs_lat']
        weather_feature_names = [col for col in self.feature_cols if col not in loc_feature_names]
        
        logger.info(f"Weather features: {len(weather_feature_names)}")
        logger.info(f"Location features: {len(loc_feature_names)}")
        
        # Prepare data
        X_weather_train = train_data_eng[weather_feature_names].values
        X_loc_train = train_data_eng[loc_feature_names].values
        y_train = train_data_eng['energy_output_kWh'].values
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_weather_train_scaled = self.scaler_X.fit_transform(X_weather_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X_weather_seq, X_loc_seq, y_seq = self.prepare_sequences(
            X_weather_train_scaled, X_loc_train, y_train_scaled
        )
        
        logger.info(f"Training sequences: Weather={X_weather_seq.shape}, Location={X_loc_seq.shape}")
        
        # Validation data
        validation_data = None
        if val_data is not None:
            X_weather_val = val_data_eng[weather_feature_names].values
            X_loc_val = val_data_eng[loc_feature_names].values
            y_val = val_data_eng['energy_output_kWh'].values
            
            X_weather_val_scaled = self.scaler_X.transform(X_weather_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            
            X_weather_val_seq, X_loc_val_seq, y_val_seq = self.prepare_sequences(
                X_weather_val_scaled, X_loc_val, y_val_scaled
            )
            
            validation_data = ([X_weather_val_seq, X_loc_val_seq], y_val_seq)
        
        # Build model
        weather_input_shape = (X_weather_seq.shape[1], X_weather_seq.shape[2])
        self.model = self.build_model(weather_input_shape)
        
        # Callbacks
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
            [X_weather_seq, X_loc_seq], y_seq,
            epochs=100,
            batch_size=32,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate bias correction
        y_train_pred_scaled = self.model.predict([X_weather_seq, X_loc_seq], verbose=0)
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled).flatten()
        y_train_actual = y_train[self.sequence_length:]
        
        self.bias_correction = np.mean(y_train_actual - y_train_pred)
        logger.info(f"Bias correction: {self.bias_correction:.4f}")
        
        logger.info("✓ Training complete")
        
        return {
            'history': history.history,
            'bias_correction': self.bias_correction
        }
    
    def save(self, path: str):
        """Save model and metadata"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        model_path = path if path.endswith('.keras') else f"{path}.keras"
        self.model.save(model_path)
        
        # Save metadata
        import joblib
        metadata_path = path.replace('.keras', '_metadata.pkl')
        joblib.dump({
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'bias_correction': self.bias_correction,
            'sequence_length': self.sequence_length,
            'feature_cols': self.feature_cols
        }, metadata_path)
        
        logger.info(f"✓ Model saved to {model_path}")
        logger.info(f"✓ Metadata saved to {metadata_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would be replaced with real data from PostgreSQL
    print("Multi-Location Forecaster V2")
    print("This is a prototype for the V2 roadmap")
    print("Requires real telemetry data from user installations")
