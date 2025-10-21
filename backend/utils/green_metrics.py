"""
Green AI Metrics Tracker
Monitors compute resources, energy consumption, and carbon emissions
"""
import time
import psutil
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class GreenMetricsTracker:
    """
    Track energy consumption and carbon emissions for AI operations
    Implements Green AI best practices
    """
    
    # Carbon intensity factors (gCO2/kWh) by region
    CARBON_INTENSITY = {
        'us-east': 415,      # US East Coast
        'us-west': 250,      # US West Coast (more renewable)
        'eu-west': 300,      # EU West
        'eu-north': 50,      # Nordic (very low - hydro/wind)
        'asia-pacific': 600, # Asia Pacific average
        'default': 400       # Global average
    }
    
    # Average power consumption (Watts)
    POWER_CONSUMPTION = {
        'cpu_per_core': 15,      # Watts per CPU core
        'gpu_nvidia_t4': 70,     # Watts for T4 GPU
        'gpu_nvidia_v100': 300,  # Watts for V100 GPU
        'gpu_nvidia_a100': 400,  # Watts for A100 GPU
        'ram_per_gb': 0.375,     # Watts per GB RAM
        'storage_ssd': 2,        # Watts for SSD
        'network': 5             # Watts for network
    }
    
    def __init__(self, region: str = 'default', log_file: Optional[Path] = None):
        """
        Initialize green metrics tracker
        
        Args:
            region: Cloud region for carbon intensity
            log_file: Path to log file for metrics
        """
        self.region = region
        self.carbon_intensity = self.CARBON_INTENSITY.get(region, self.CARBON_INTENSITY['default'])
        self.log_file = log_file
        
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        
    def start_tracking(self, operation_name: str):
        """Start tracking an operation"""
        self.operation_name = operation_name
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=0.1)
        self.start_memory = psutil.virtual_memory().used / (1024**3)  # GB
        
        logger.info(f"🌱 Started tracking: {operation_name}")
        
    def stop_tracking(self) -> Dict:
        """
        Stop tracking and calculate metrics
        
        Returns:
            Dictionary of green metrics
        """
        if not self.start_time:
            logger.warning("Tracking not started")
            return {}
        
        self.end_time = time.time()
        duration_hours = (self.end_time - self.start_time) / 3600
        
        # Get resource usage
        cpu_count = psutil.cpu_count()
        avg_cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_used_gb = psutil.virtual_memory().used / (1024**3)
        
        # Estimate power consumption
        cpu_power = (cpu_count * self.POWER_CONSUMPTION['cpu_per_core'] * 
                    (avg_cpu_percent / 100))
        memory_power = memory_used_gb * self.POWER_CONSUMPTION['ram_per_gb']
        storage_power = self.POWER_CONSUMPTION['storage_ssd']
        network_power = self.POWER_CONSUMPTION['network']
        
        # Total power in Watts
        total_power_watts = cpu_power + memory_power + storage_power + network_power
        
        # Energy consumption in kWh
        energy_kwh = (total_power_watts / 1000) * duration_hours
        
        # Carbon emissions in gCO2
        carbon_g = energy_kwh * self.carbon_intensity
        carbon_kg = carbon_g / 1000
        
        # Calculate efficiency metrics
        duration_seconds = self.end_time - self.start_time
        
        self.metrics = {
            'operation': self.operation_name,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration_seconds,
            'duration_hours': duration_hours,
            'cpu_cores_used': cpu_count,
            'avg_cpu_percent': avg_cpu_percent,
            'memory_gb': memory_used_gb,
            'power_watts': total_power_watts,
            'energy_kwh': energy_kwh,
            'carbon_g': carbon_g,
            'carbon_kg': carbon_kg,
            'carbon_intensity_region': self.region,
            'carbon_intensity_gco2_kwh': self.carbon_intensity
        }
        
        # Log metrics
        logger.info(f"🌱 Green Metrics for {self.operation_name}:")
        logger.info(f"   ⚡ Energy: {energy_kwh:.6f} kWh")
        logger.info(f"   🌍 Carbon: {carbon_g:.2f} gCO2 ({carbon_kg:.6f} kgCO2)")
        logger.info(f"   ⏱️  Duration: {duration_seconds:.2f}s")
        logger.info(f"   💻 CPU: {avg_cpu_percent:.1f}% ({cpu_count} cores)")
        logger.info(f"   🧠 Memory: {memory_used_gb:.2f} GB")
        
        # Save to log file
        if self.log_file:
            self._save_to_log()
        
        return self.metrics
    
    def _save_to_log(self):
        """Save metrics to JSON log file"""
        try:
            # Load existing logs
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Append new metrics
            logs.append(self.metrics)
            
            # Save
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
            logger.debug(f"Saved metrics to {self.log_file}")
        except Exception as e:
            logger.error(f"Error saving metrics to log: {e}")
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics from log file
        
        Returns:
            Aggregated statistics
        """
        if not self.log_file or not self.log_file.exists():
            return {}
        
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            if not logs:
                return {}
            
            total_energy = sum(log.get('energy_kwh', 0) for log in logs)
            total_carbon = sum(log.get('carbon_g', 0) for log in logs)
            total_operations = len(logs)
            avg_duration = sum(log.get('duration_seconds', 0) for log in logs) / total_operations
            
            return {
                'total_operations': total_operations,
                'total_energy_kwh': total_energy,
                'total_carbon_g': total_carbon,
                'total_carbon_kg': total_carbon / 1000,
                'avg_energy_per_operation_kwh': total_energy / total_operations,
                'avg_carbon_per_operation_g': total_carbon / total_operations,
                'avg_duration_seconds': avg_duration,
                'equivalent_tree_months': total_carbon / 21000,  # 1 tree absorbs ~21kg CO2/year
                'equivalent_km_driven': total_carbon / 120  # Average car emits ~120g CO2/km
            }
        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {}
    
    @staticmethod
    def estimate_training_cost(epochs: int, batch_size: int, dataset_size: int,
                              model_params: int, region: str = 'default') -> Dict:
        """
        Estimate energy cost for model training
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            dataset_size: Number of training samples
            model_params: Number of model parameters (millions)
            region: Cloud region
            
        Returns:
            Estimated costs
        """
        # Estimate training time (rough heuristic)
        iterations_per_epoch = dataset_size / batch_size
        total_iterations = iterations_per_epoch * epochs
        
        # Time per iteration (ms) - scales with model size
        time_per_iteration_ms = 10 + (model_params * 0.5)
        total_time_hours = (total_iterations * time_per_iteration_ms / 1000) / 3600
        
        # Power consumption (assume GPU training)
        gpu_power_watts = GreenMetricsTracker.POWER_CONSUMPTION['gpu_nvidia_t4']
        cpu_power_watts = 4 * GreenMetricsTracker.POWER_CONSUMPTION['cpu_per_core']
        total_power_watts = gpu_power_watts + cpu_power_watts
        
        # Energy and carbon
        energy_kwh = (total_power_watts / 1000) * total_time_hours
        carbon_intensity = GreenMetricsTracker.CARBON_INTENSITY.get(
            region, GreenMetricsTracker.CARBON_INTENSITY['default']
        )
        carbon_g = energy_kwh * carbon_intensity
        
        return {
            'estimated_duration_hours': total_time_hours,
            'estimated_energy_kwh': energy_kwh,
            'estimated_carbon_g': carbon_g,
            'estimated_carbon_kg': carbon_g / 1000,
            'region': region,
            'carbon_intensity': carbon_intensity
        }


def track_operation(operation_name: str, region: str = 'default', log_file: Optional[Path] = None):
    """
    Decorator to track green metrics for a function
    
    Usage:
        @track_operation("model_training")
        def train_model():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = GreenMetricsTracker(region=region, log_file=log_file)
            tracker.start_tracking(operation_name)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                metrics = tracker.stop_tracking()
                # Attach metrics to result if it's a dict
                if isinstance(result, dict):
                    result['green_metrics'] = metrics
        
        return wrapper
    return decorator
