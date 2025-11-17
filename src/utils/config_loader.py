"""
Configuration loader for project settings
"""
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Load and manage project configuration"""
    
    def __init__(self, config_path: str = "../../configs/config.yaml"):
        self.config_path = Path(__file__).parent / config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        Example: config.get('training.yolo.epochs')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @property
    def paths(self):
        """Get all paths"""
        return self.config.get('paths', {})
    
    @property
    def training_config(self):
        """Get training configuration"""
        return self.config.get('training', {})
    
    @property
    def detection_config(self):
        """Get detection configuration"""
        return self.config.get('detection', {})

# Global config instance
config = Config()