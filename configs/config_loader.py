
import yaml
from pathlib import Path
import os

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return str(self.__dict__)

def load_config(config_path):
    """
    Load configuration from YAML file and return a Config object (dot-notation access).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    return Config(config_dict)
