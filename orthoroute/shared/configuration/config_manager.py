"""Configuration manager for loading, saving, and managing application settings."""
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import asdict

from .settings import ApplicationSettings

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration manager for OrthoRoute."""
    
    DEFAULT_CONFIG_PATHS = [
        "orthoroute.json",
        "config/orthoroute.json",
        "~/.orthoroute/config.json",
        "~/.config/orthoroute/config.json"
    ]
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file.
                        If None, will search default locations.
        """
        self.config_path: Optional[Path] = None
        self.settings: ApplicationSettings = ApplicationSettings()
        
        # Find or create config file
        if config_path:
            self.config_path = Path(config_path).expanduser().resolve()
        else:
            self.config_path = self._find_config_file()
        
        # Load existing configuration
        if self.config_path and self.config_path.exists():
            self.load()
        else:
            # Create default config file
            self._create_default_config()
    
    def _find_config_file(self) -> Optional[Path]:
        """Find existing configuration file in default locations."""
        for path_str in self.DEFAULT_CONFIG_PATHS:
            path = Path(path_str).expanduser().resolve()
            if path.exists():
                logger.info(f"Found existing config file: {path}")
                return path
        
        # No existing config found, use first default location
        default_path = Path(self.DEFAULT_CONFIG_PATHS[0]).expanduser().resolve()
        logger.info(f"No existing config found, will create: {default_path}")
        return default_path
    
    def _create_default_config(self):
        """Create default configuration file."""
        if self.config_path:
            try:
                # Ensure directory exists
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save default settings
                self.save()
                logger.info(f"Created default configuration file: {self.config_path}")
                
            except Exception as e:
                logger.error(f"Failed to create default config file: {e}")
    
    def load(self, config_path: Optional[Union[str, Path]] = None) -> bool:
        """Load configuration from file.
        
        Args:
            config_path: Optional path to load from. Uses instance path if None.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        path = Path(config_path).expanduser().resolve() if config_path else self.config_path
        
        if not path or not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return False
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update settings from loaded data
            self._update_settings_from_dict(config_data)
            
            # Validate loaded settings
            errors = self.validate()
            if any(error_list for error_list in errors.values()):
                logger.warning("Configuration validation errors found:")
                for category, error_list in errors.items():
                    for error in error_list:
                        logger.warning(f"  {category}: {error}")
            
            logger.info(f"Configuration loaded from: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            return False
    
    def save(self, config_path: Optional[Union[str, Path]] = None) -> bool:
        """Save configuration to file.
        
        Args:
            config_path: Optional path to save to. Uses instance path if None.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        path = Path(config_path).expanduser().resolve() if config_path else self.config_path
        
        if not path:
            logger.error("No configuration path specified")
            return False
        
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert settings to dictionary
            config_data = asdict(self.settings)
            
            # Write to file with pretty formatting
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {path}: {e}")
            return False
    
    def _update_settings_from_dict(self, config_data: Dict[str, Any]):
        """Update settings from dictionary data."""
        # Helper function to update nested dataclass
        def update_dataclass(obj, data):
            if not isinstance(data, dict):
                return
            
            for key, value in data.items():
                if hasattr(obj, key):
                    attr = getattr(obj, key)
                    if hasattr(attr, '__dict__'):  # It's a dataclass
                        update_dataclass(attr, value)
                    else:
                        setattr(obj, key, value)
        
        update_dataclass(self.settings, config_data)
    
    def get_settings(self) -> ApplicationSettings:
        """Get current application settings."""
        return self.settings
    
    def update_routing_settings(self, **kwargs):
        """Update routing settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings.routing, key):
                setattr(self.settings.routing, key, value)
            else:
                logger.warning(f"Unknown routing setting: {key}")
    
    def update_display_settings(self, **kwargs):
        """Update display settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings.display, key):
                setattr(self.settings.display, key, value)
            else:
                logger.warning(f"Unknown display setting: {key}")
    
    def update_gpu_settings(self, **kwargs):
        """Update GPU settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings.gpu, key):
                setattr(self.settings.gpu, key, value)
            else:
                logger.warning(f"Unknown GPU setting: {key}")
    
    def update_kicad_settings(self, **kwargs):
        """Update KiCad settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings.kicad, key):
                setattr(self.settings.kicad, key, value)
            else:
                logger.warning(f"Unknown KiCad setting: {key}")
    
    def update_logging_settings(self, **kwargs):
        """Update logging settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings.logging, key):
                setattr(self.settings.logging, key, value)
            else:
                logger.warning(f"Unknown logging setting: {key}")
    
    def validate(self) -> Dict[str, Any]:
        """Validate current settings."""
        return self.settings.validate()
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.settings = ApplicationSettings()
        logger.info("Settings reset to defaults")
    
    def reset_category_to_defaults(self, category: str):
        """Reset a specific category to defaults."""
        if category == "routing":
            self.settings.routing = ApplicationSettings().routing
        elif category == "display":
            self.settings.display = ApplicationSettings().display
        elif category == "gpu":
            self.settings.gpu = ApplicationSettings().gpu
        elif category == "kicad":
            self.settings.kicad = ApplicationSettings().kicad
        elif category == "logging":
            self.settings.logging = ApplicationSettings().logging
        else:
            logger.warning(f"Unknown settings category: {category}")
            return
        
        logger.info(f"Reset {category} settings to defaults")
    
    def backup_config(self, suffix: str = None) -> Optional[Path]:
        """Create a backup of current configuration.
        
        Args:
            suffix: Optional suffix for backup filename
            
        Returns:
            Path to backup file if successful, None otherwise
        """
        if not self.config_path:
            logger.error("No configuration path to backup")
            return None
        
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix_str = f"_{suffix}" if suffix else ""
            backup_name = f"{self.config_path.stem}_backup_{timestamp}{suffix_str}.json"
            backup_path = self.config_path.parent / backup_name
            
            # Copy current config to backup location
            import shutil
            shutil.copy2(self.config_path, backup_path)
            
            logger.info(f"Configuration backed up to: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to backup configuration: {e}")
            return None
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about current configuration."""
        return {
            "config_path": str(self.config_path) if self.config_path else None,
            "config_exists": self.config_path.exists() if self.config_path else False,
            "version": self.settings.version,
            "config_version": self.settings.config_version,
            "validation_errors": self.validate()
        }


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def initialize_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Initialize the global configuration manager with optional custom path."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager