"""
Configuration loading utilities.
Supports YAML configs with nested attribute access.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from dataclasses import dataclass, field


@dataclass
class Config:
    """Nested configuration object with dot notation access."""
    _data: Dict[str, Any] = field(default_factory=dict)
    
    def __getattr__(self, key: str) -> Any:
        if key in self._data:
            value = self._data[key]
            if isinstance(value, dict):
                return Config(value)
            elif isinstance(value, list):
                return [Config(v) if isinstance(v, dict) else v for v in value]
            return value
        raise AttributeError(f"'Config' object has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    # --- Mapping protocol (so nested configs behave like dicts) ---
    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self):
        return self._data.keys()

    def values(self):
        return [getattr(self, k) for k in self._data.keys()]

    def items(self):
        return [(k, getattr(self, k)) for k in self._data.keys()]
    
    def get(self, key: str, default: Any = None) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        def convert(obj):
            if isinstance(obj, Config):
                return {k: convert(v) for k, v in obj._data.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        return convert(self)
    
    def update(self, other: Union[Dict, "Config"]) -> None:
        if isinstance(other, Config):
            other = other.to_dict()
        self._data.update(other)


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    
    return Config(data or {})


def save_config(config: Config, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Config, override: Config) -> Config:
    """Merge two configs, with override taking precedence."""
    result = Config(base.to_dict())
    result.update(override.to_dict())
    return result