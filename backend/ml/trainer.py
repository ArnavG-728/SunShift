"""
Model Trainer
Orchestrates data collection, model training, and evaluation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

from .data_collector import SolarDataCollector
from .solar_forecaster import SolarForecasterML

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates the complete ML training pipeline:
    1. Data collection
    2. Model training
    3. Evaluation
    4. Model saving
    """
    
    def __init__(self, model_dir: str = None, cache_dir: str = None):
        """
        Initialize the trainer.
        
        Args:
            model_dir: Directory to save trained models
            cache_dir: Directory to cache training data
        """
        self.base_dir = Path(__file__).parent.parent
        self.model_dir = Path(model_dir) if model_dir else self.base_dir / "models" / "ml_saved"
        self.cache_dir = Path(cache_dir) if cache_dir else self.base_dir / "data" / "cache"
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_collector = SolarDataCollector(str(self.cache_dir))
        self.forecaster = None
        self.training_history = []
    
    def train_model(
        self,
        lat: float = 28.6139,
        lon: float = 77.2090,
        days: int = 365,
        epochs: int = 100,
        batch_size: int = 32,
        sequence_length: int = 24,
        force_data_refresh: bool = False,
        model_name: str = "solar_forecaster"
    ) -> Dict:
        """
        Train a new solar forecasting model.
        
        Args:
            lat: Latitude for data collection
            lon: Longitude for data collection
            days: Days of historical data to collect (default: 365)
            epochs: Training epochs (default: 100)
            batch_size: Training batch size (default: 32)
            sequence_length: LSTM sequence length (default: 24)
            force_data_refresh: If True, re-collect data even if cached
            model_name: Name for saving the model
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Solar Forecaster ML Training Pipeline")
        logger.info("=" * 60)
        logger.info(f"Location: ({lat}, {lon})")
        logger.info(f"Training data: {days} days")
        logger.info(f"Model config: epochs={epochs}, batch_size={batch_size}, seq_len={sequence_length}")
        
        results = {
            'status': 'in_progress',
            'location': {'lat': lat, 'lon': lon},
            'started_at': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Collect training data
            logger.info("\n[Step 1/4] Collecting training data...")
            
            if force_data_refresh:
                # Clear cached data for this location
                cache_pattern = f"solar_data_{lat:.4f}_{lon:.4f}_*"
                for cache_file in self.cache_dir.glob(cache_pattern):
                    cache_file.unlink()
                    logger.info(f"Cleared cache: {cache_file}")
            
            train_data = self.data_collector.collect_training_data(
                lat=lat, 
                lon=lon, 
                days=days
            )
            
            if len(train_data) < 100:
                raise ValueError(f"Insufficient training data: {len(train_data)} samples")
            
            results['data_collection'] = {
                'samples': len(train_data),
                'date_range': {
                    'start': str(train_data['timestamp'].min()),
                    'end': str(train_data['timestamp'].max())
                },
                'features': len(train_data.columns),
                'target_stats': {
                    'mean': float(train_data['energy_output_kWh'].mean()),
                    'std': float(train_data['energy_output_kWh'].std()),
                    'min': float(train_data['energy_output_kWh'].min()),
                    'max': float(train_data['energy_output_kWh'].max())
                }
            }
            
            logger.info(f"✓ Collected {len(train_data)} samples")
            logger.info(f"  Date range: {train_data['timestamp'].min()} to {train_data['timestamp'].max()}")
            
            # Step 2: Split data
            logger.info("\n[Step 2/4] Splitting data...")
            
            split_idx = int(len(train_data) * 0.8)
            train_subset = train_data.iloc[:split_idx].copy()
            val_subset = train_data.iloc[split_idx:].copy()
            
            logger.info(f"✓ Train: {len(train_subset)} samples, Validation: {len(val_subset)} samples")
            
            # Step 3: Train model
            logger.info("\n[Step 3/4] Training model...")
            
            self.forecaster = SolarForecasterML(
                sequence_length=sequence_length,
                model_dir=str(self.model_dir)
            )
            
            training_result = self.forecaster.train(
                train_data=train_subset,
                val_data=val_subset,
                epochs=epochs,
                batch_size=batch_size,
                location={'lat': lat, 'lon': lon}
            )
            
            results['training'] = {
                'epochs_completed': len(training_result['history']['loss']),
                'final_loss': float(training_result['history']['loss'][-1]),
                'final_val_loss': float(training_result['history']['val_loss'][-1]),
                'mae': float(training_result['mae']),
                'rmse': float(training_result['rmse']),
                'mape': float(training_result['mape']),
                'bias_correction': float(training_result['bias_correction'])
            }
            
            logger.info(f"✓ Training complete:")
            logger.info(f"  MAE: {training_result['mae']:.4f} kWh")
            logger.info(f"  RMSE: {training_result['rmse']:.4f} kWh")
            logger.info(f"  MAPE: {training_result['mape']:.2f}%")
            
            # Step 4: Evaluate and save
            logger.info("\n[Step 4/4] Evaluating and saving model...")
            
            # Test predictions on validation data
            val_predictions = self.forecaster.predict(val_subset)
            val_actual = val_subset['energy_output_kWh'].values[self.forecaster.sequence_length:]
            
            # Daytime vs nighttime accuracy
            val_timestamps = val_subset['timestamp'].values[self.forecaster.sequence_length:]
            val_hours = pd.to_datetime(val_timestamps).hour
            
            daytime_mask = (val_hours >= 6) & (val_hours <= 18)
            if daytime_mask.sum() > 0:
                daytime_mae = np.mean(np.abs(val_actual[daytime_mask] - val_predictions[daytime_mask]))
            else:
                daytime_mae = 0.0
            
            nighttime_mask = ~daytime_mask
            if nighttime_mask.sum() > 0:
                nighttime_mae = np.mean(np.abs(val_actual[nighttime_mask] - val_predictions[nighttime_mask]))
            else:
                nighttime_mae = 0.0
            
            results['evaluation'] = {
                'overall_mae': float(training_result['mae']),
                'daytime_mae': float(daytime_mae),
                'nighttime_mae': float(nighttime_mae),
                'prediction_range': {
                    'min': float(val_predictions.min()),
                    'max': float(val_predictions.max()),
                    'mean': float(val_predictions.mean())
                }
            }
            
            # Save model
            self.forecaster.save(model_name)
            
            results['model'] = {
                'name': model_name,
                'path': str(self.model_dir / f"{model_name}.keras"),
                'info': self.forecaster.get_model_info()
            }
            
            logger.info(f"✓ Model saved as '{model_name}'")
            
            # Update final results
            results['status'] = 'success'
            results['completed_at'] = datetime.now().isoformat()
            
            # Save training report
            report_path = self.model_dir / f"{model_name}_training_report.json"
            with open(report_path, 'w') as f:
                # Convert non-serializable types
                serializable_results = json.loads(
                    json.dumps(results, default=str)
                )
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"\n{'=' * 60}")
            logger.info("Training Pipeline Complete!")
            logger.info(f"Model: {model_name}")
            logger.info(f"MAE: {training_result['mae']:.4f} kWh")
            logger.info(f"Report saved: {report_path}")
            logger.info(f"{'=' * 60}")
            
            self.training_history.append(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            results['status'] = 'failed'
            results['error'] = str(e)
            results['completed_at'] = datetime.now().isoformat()
            return results
    
    def train_for_multiple_locations(
        self,
        locations: List[Dict],
        days: int = 365,
        epochs: int = 100,
        **kwargs
    ) -> Dict:
        """
        Train models for multiple locations.
        
        Args:
            locations: List of {'lat': float, 'lon': float, 'name': str}
            days: Days of training data
            epochs: Training epochs
            
        Returns:
            Dictionary with results for each location
        """
        results = {}
        
        for loc in locations:
            name = loc.get('name', f"{loc['lat']}_{loc['lon']}")
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training model for: {name}")
            logger.info(f"{'=' * 60}")
            
            model_name = f"solar_forecaster_{name.replace(' ', '_').lower()}"
            
            result = self.train_model(
                lat=loc['lat'],
                lon=loc['lon'],
                days=days,
                epochs=epochs,
                model_name=model_name,
                **kwargs
            )
            
            results[name] = result
        
        return results
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available trained models."""
        models = []
        
        for model_file in self.model_dir.glob("*.keras"):
            model_name = model_file.stem
            artifacts_file = self.model_dir / f"{model_name}_artifacts.pkl"
            report_file = self.model_dir / f"{model_name}_training_report.json"
            
            model_info = {
                'name': model_name,
                'path': str(model_file),
                'size_mb': model_file.stat().st_size / (1024 * 1024),
                'has_artifacts': artifacts_file.exists(),
                'has_report': report_file.exists()
            }
            
            if report_file.exists():
                try:
                    with open(report_file, 'r') as f:
                        report = json.load(f)
                    model_info['training_date'] = report.get('completed_at')
                    model_info['location'] = report.get('location')
                    if 'training' in report:
                        model_info['mae'] = report['training'].get('mae')
                        model_info['rmse'] = report['training'].get('rmse')
                except:
                    pass
            
            models.append(model_info)
        
        return sorted(models, key=lambda x: x.get('training_date', ''), reverse=True)
    
    def load_model(self, model_name: str = "solar_forecaster") -> Optional[SolarForecasterML]:
        """Load a trained model."""
        forecaster = SolarForecasterML(model_dir=str(self.model_dir))
        
        if forecaster.load(model_name):
            self.forecaster = forecaster
            return forecaster
        
        return None


def quick_train(
    lat: float = 28.6139,
    lon: float = 77.2090,
    days: int = 90,
    epochs: int = 50
) -> Dict:
    """
    Quick convenience function to train a model.
    
    Args:
        lat: Latitude
        lon: Longitude
        days: Days of training data
        epochs: Training epochs
        
    Returns:
        Training results
    """
    trainer = ModelTrainer()
    return trainer.train_model(
        lat=lat, 
        lon=lon, 
        days=days, 
        epochs=epochs
    )


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Train Solar Forecaster ML Model")
    parser.add_argument('--lat', type=float, default=28.6139, help='Latitude')
    parser.add_argument('--lon', type=float, default=77.2090, help='Longitude')
    parser.add_argument('--days', type=int, default=365, help='Days of training data')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=24, help='Sequence length')
    parser.add_argument('--name', type=str, default='solar_forecaster', help='Model name')
    parser.add_argument('--refresh', action='store_true', help='Force data refresh')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    result = trainer.train_model(
        lat=args.lat,
        lon=args.lon,
        days=args.days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        model_name=args.name,
        force_data_refresh=args.refresh
    )
    
    print(f"\n{'=' * 60}")
    print("TRAINING RESULT:")
    print(f"{'=' * 60}")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"MAE: {result['training']['mae']:.4f} kWh")
        print(f"RMSE: {result['training']['rmse']:.4f} kWh")
        print(f"MAPE: {result['training']['mape']:.2f}%")
        print(f"Model saved: {result['model']['path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
