"""
YOLO Training Script for Helmet Detection
Optimized for Puffer GPU cluster
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
from utils.logger import setup_logger, log
from utils.config_loader import config

class YOLOTrainer:
    """
    YOLO training pipeline for helmet detection
    """
    
    def __init__(self, data_yaml_path: str, output_dir: str = None):
        """
        Initialize trainer
        
        Args:
            data_yaml_path: Path to data.yaml configuration
            output_dir: Output directory for models (default: ../models/yolo)
        """
        self.data_yaml = Path(data_yaml_path)
        self.output_dir = Path(output_dir) if output_dir else Path("../models/yolo")
        
        # Load training config
        self.train_config = config.get('training.yolo', {})
        
        log.info(f"YOLOTrainer initialized")
        log.info(f"Data config: {self.data_yaml}")
        log.info(f"Output dir: {self.output_dir}")
    
    def check_environment(self):
        """Check training environment"""
        log.info("="*70)
        log.info("ENVIRONMENT CHECK")
        log.info("="*70)
        
        log.info(f"PyTorch version: {torch.__version__}")
        log.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            log.info(f"CUDA version: {torch.version.cuda}")
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            log.warning("No GPU detected! Training will be very slow.")
        
        log.info("="*70)
    
    def verify_data(self):
        """Verify dataset configuration"""
        log.info("\nVERIFYING DATASET...")
        
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {self.data_yaml}")
        
        with open(self.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        log.info(f"Classes: {data_config['nc']}")
        log.info(f"Names: {data_config['names']}")
        
        # Check paths
        base_path = self.data_yaml.parent
        train_path = base_path / data_config['train']
        val_path = base_path / data_config['val']
        
        # Count images
        train_images = list(train_path.glob('*.[jp][pn][g]'))
        val_images = list(val_path.glob('*.[jp][pn][g]'))
        
        log.info(f"\nTraining images: {len(train_images)}")
        log.info(f"Validation images: {len(val_images)}")
        
        if len(train_images) == 0:
            raise Exception("No training images found!")
        
        if len(val_images) == 0:
            log.warning("No validation images found!")
        
        log.info("✓ Dataset verified!\n")
        
        return data_config
    
    def train(self):
        """Run training"""
        # Environment check
        self.check_environment()
        
        # Verify data
        data_config = self.verify_data()
        
        # Load model
        model_size = self.train_config.get('model_size', 'n')
        model_name = f"yolov8{model_size}.pt"
        
        log.info(f"\nLoading pre-trained model: {model_name}")
        model = YOLO(model_name)
        
        # Training parameters
        epochs = self.train_config.get('epochs', 100)
        batch_size = self.train_config.get('batch_size', 16)
        img_size = self.train_config.get('img_size', 640)
        device = self.train_config.get('device', '0')
        patience = self.train_config.get('patience', 15)
        workers = self.train_config.get('workers', 4)
        
        # Augmentation
        aug = config.get('training.augmentation', {})
        
        log.info("\n" + "="*70)
        log.info("TRAINING CONFIGURATION")
        log.info("="*70)
        log.info(f"Model: {model_name}")
        log.info(f"Epochs: {epochs}")
        log.info(f"Batch size: {batch_size}")
        log.info(f"Image size: {img_size}")
        log.info(f"Device: GPU {device}")
        log.info(f"Patience: {patience}")
        log.info(f"Workers: {workers}")
        log.info("="*70 + "\n")
        
        # Create experiment name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"helmet_detection_{timestamp}"
        
        log.info(f"Starting training: {experiment_name}")
        log.info("This may take 1-3 hours depending on GPU and dataset size...")
        log.info("You can monitor progress in real-time\n")
        
        # Train!
        results = model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            patience=patience,
            workers=workers,
            
            # Output settings
            project=str(self.output_dir),
            name=experiment_name,
            exist_ok=True,
            
            # Saving
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            
            # Augmentation
            hsv_h=aug.get('hsv_h', 0.015),
            hsv_s=aug.get('hsv_s', 0.7),
            hsv_v=aug.get('hsv_v', 0.4),
            degrees=aug.get('degrees', 0.0),
            translate=aug.get('translate', 0.1),
            scale=aug.get('scale', 0.5),
            fliplr=aug.get('fliplr', 0.5),
            mosaic=aug.get('mosaic', 1.0),
            
            # Visualization
            plots=True,
            
            # Performance
            amp=True,  # Automatic Mixed Precision
            val=True,
        )
        
        log.info("\n" + "="*70)
        log.info("TRAINING COMPLETE!")
        log.info("="*70)
        
        return results, experiment_name
    
    def evaluate(self, experiment_name: str):
        """Evaluate trained model"""
        log.info("\nEVALUATING MODEL...")
        
        # Load best model
        best_model_path = self.output_dir / experiment_name / "weights" / "best.pt"
        
        if not best_model_path.exists():
            log.error(f"Best model not found: {best_model_path}")
            return None
        
        model = YOLO(str(best_model_path))
        
        # Validate
        metrics = model.val(data=str(self.data_yaml))
        
        log.info("\n" + "="*70)
        log.info("EVALUATION RESULTS")
        log.info("="*70)
        log.info(f"mAP@0.5: {metrics.box.map50:.3f}")
        log.info(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
        log.info(f"Precision: {metrics.box.mp:.3f}")
        log.info(f"Recall: {metrics.box.mr:.3f}")
        
        # Per-class metrics
        if hasattr(metrics.box, 'maps'):
            log.info("\nPER-CLASS mAP@0.5:")
            class_names = ['with_helmet', 'without_helmet']
            for name, map_val in zip(class_names, metrics.box.maps):
                log.info(f"  {name}: {map_val:.3f}")
        
        log.info("="*70)
        
        return metrics
    
    def export_model(self, experiment_name: str, format: str = 'onnx'):
        """Export model to different format"""
        log.info(f"\nExporting model to {format.upper()}...")
        
        best_model_path = self.output_dir / experiment_name / "weights" / "best.pt"
        model = YOLO(str(best_model_path))
        
        model.export(format=format)
        
        log.info(f"✓ Model exported successfully")


def main():
    """Main training function"""
    # Setup logger
    setup_logger(
        log_file="../results/logs/training.log",
        level="INFO"
    )
    
    log.info("="*70)
    log.info("HELMET DETECTION TRAINING PIPELINE")
    log.info("="*70)
    
    # Initialize trainer
    trainer = YOLOTrainer(
        data_yaml_path="../data/helmet_dataset/data.yaml"
    )
    
    # Train
    results, experiment_name = trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate(experiment_name)
    
    # Print final summary
    log.info("\n" + "="*70)
    log.info("TRAINING SUMMARY")
    log.info("="*70)
    log.info(f"Experiment: {experiment_name}")
    log.info(f"Best model: {trainer.output_dir / experiment_name / 'weights' / 'best.pt'}")
    
    if metrics:
        log.info(f"Final mAP@0.5: {metrics.box.map50:.3f}")
        log.info(f"Final Precision: {metrics.box.mp:.3f}")
        log.info(f"Final Recall: {metrics.box.mr:.3f}")
    
    log.info("\n✓ Training pipeline complete!")
    log.info("="*70)


if __name__ == "__main__":
    main()
