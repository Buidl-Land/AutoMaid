import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Manages application configuration, loading from a JSON file and overriding
    with environment variables for flexibility.

    Environment variables can override config values. The expected format for
    environment variables is `APP_SECTION_KEY`, where `SECTION` is the top-level
    JSON key and `KEY` is the nested key.

    For example, to override `embedding_model_name` in the `memory_protocol`
    section, you would set the environment variable:
    `APP_MEMORY_PROTOCOL_EMBEDDING_MODEL_NAME="your-model-name"`
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the ConfigManager and loads the configuration.

        Args:
            config_path: The path to the JSON configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads configuration from the JSON file and merges overrides from
        environment variables.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        # Apply overrides from environment variables
        for section, settings in config.items():
            if isinstance(settings, dict):
                for key, value in settings.items():
                    env_var_name = f"APP_{section.upper()}_{key.upper()}"
                    override_value = os.getenv(env_var_name)
                    if override_value:
                        # Attempt to parse the override value to its original type
                        original_type = type(value)
                        try:
                            if original_type == bool:
                                config[section][key] = override_value.lower() in ['true', '1', 'yes']
                            elif original_type == int:
                                config[section][key] = int(override_value)
                            elif original_type == float:
                                config[section][key] = float(override_value)
                            else:
                                config[section][key] = override_value
                        except (ValueError, TypeError):
                            config[section][key] = override_value # Fallback to string

        return config

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the entire configuration dictionary.

        Returns:
            The fully resolved configuration.
        """
        return self.config

    def get_section(self, section_name: str) -> Optional[Dict[str, Any]]:
        """
        Returns a specific section of the configuration.

        Args:
            section_name: The name of the configuration section to retrieve.

        Returns:
            A dictionary representing the configuration section, or None if not found.
        """
        return self.config.get(section_name)

# Global instance for easy access
# This allows other modules to import and use the same config manager instance.
config_manager = ConfigManager()
