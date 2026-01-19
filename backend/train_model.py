"""
Train Solar Forecaster ML Model
Run this script to train the ML model for solar energy forecasting.

Usage:
    python train_model.py
    python train_model.py --lat 28.6139 --lon 77.2090 --days 365 --epochs 100
    
Arguments:
    --lat: Latitude (default: 28.6139)
    --lon: Longitude (default: 77.2090)
    --days: Days of training data to collect (default: 365)
    --epochs: Training epochs (default: 100)
    --batch-size: Batch size (default: 32)
    --name: Model name (default: solar_forecaster)
    --refresh: Force refresh training data cache
"""
import argparse
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train Solar Forecaster ML Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py
      → Train with default settings (Delhi location, 365 days, 100 epochs)
  
  python train_model.py --lat 40.7128 --lon -74.0060 --days 180
      → Train for New York with 180 days of data
  
  python train_model.py --quick
      → Quick training with 90 days and 50 epochs
        """
    )
    
    parser.add_argument('--lat', type=float, default=28.6139, 
                        help='Latitude (default: 28.6139 - Delhi)')
    parser.add_argument('--lon', type=float, default=77.2090, 
                        help='Longitude (default: 77.2090 - Delhi)')
    parser.add_argument('--days', type=int, default=365, 
                        help='Days of historical data (default: 365)')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size (default: 32)')
    parser.add_argument('--seq-len', type=int, default=24, 
                        help='Sequence length for LSTM (default: 24)')
    parser.add_argument('--name', type=str, default='solar_forecaster', 
                        help='Model name (default: solar_forecaster)')
    parser.add_argument('--refresh', action='store_true', 
                        help='Force refresh training data cache')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick training (90 days, 50 epochs)')
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "=" * 60)
    print("       🌞 SunShift ML Model Training Pipeline 🌞")
    print("=" * 60)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # Apply quick settings if requested
    if args.quick:
        args.days = 90
        args.epochs = 50
        logger.info("Quick training mode enabled (90 days, 50 epochs)")
    
    logger.info(f"Training Configuration:")
    logger.info(f"  Location: ({args.lat}, {args.lon})")
    logger.info(f"  Data: {args.days} days of historical data")
    logger.info(f"  Model: {args.epochs} epochs, batch size {args.batch_size}")
    logger.info(f"  Sequence length: {args.seq_len}")
    logger.info(f"  Model name: {args.name}")
    
    try:
        from ml.trainer import ModelTrainer
        
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
        
        print("\n" + "=" * 60)
        print("                  TRAINING COMPLETE")
        print("=" * 60)
        
        if result['status'] == 'success':
            print(f"  ✓ Status: SUCCESS")
            print(f"  ✓ Training Samples: {result['data_collection']['samples']}")
            print(f"  ✓ MAE: {result['training']['mae']:.4f} kWh")
            print(f"  ✓ RMSE: {result['training']['rmse']:.4f} kWh")
            print(f"  ✓ MAPE: {result['training']['mape']:.2f}%")
            print(f"  ✓ Daytime MAE: {result['evaluation']['daytime_mae']:.4f} kWh")
            print(f"  ✓ Model saved: {result['model']['path']}")
            print("=" * 60 + "\n")
            return 0
        else:
            print(f"  ✗ Status: FAILED")
            print(f"  ✗ Error: {result.get('error', 'Unknown error')}")
            print("=" * 60 + "\n")
            return 1
            
    except ImportError as e:
        logger.error(f"Import error - make sure you're in the backend directory: {e}")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
