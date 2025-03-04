import json
import logging
import os
import re
from typing import Dict, List, Pattern, Any

from cognee.modules.engine.models.EntityType import EntityType

logger = logging.getLogger("regex_entity_config")


class RegexEntityConfig:
    """Class to load and process regex entity extraction configuration."""

    def __init__(self, config_path: str = None):
        """Initialize the regex entity configuration with an optional custom config path."""
        if config_path is None:
            # Use default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "regex_entity_config.json")

        self.config_path = config_path
        self.entity_configs = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load and process the configuration from the JSON file."""
        try:
            with open(self.config_path, "r") as f:
                config_list = json.load(f)

            # Process each entity configuration
            for config in config_list:
                entity_name = config["entity_name"]

                # Create EntityType instance
                entity_type = EntityType(name=entity_name, description=config["entity_description"])

                # Compile regex pattern
                compiled_pattern = re.compile(config["regex"])

                # Store in dictionary
                self.entity_configs[entity_name] = {
                    "entity_type": entity_type,
                    "regex": config["regex"],
                    "compiled_pattern": compiled_pattern,
                    "description_template": config["description_template"],
                }

            logger.info(
                f"Loaded {len(self.entity_configs)} entity configurations from {self.config_path}"
            )

        except Exception as e:
            logger.error(f"Failed to load entity configuration from {self.config_path}: {str(e)}")
            raise

    def get_entity_names(self) -> List[str]:
        """Get a list of all entity names in the configuration."""
        return list(self.entity_configs.keys())

    def get_entity_config(self, entity_name: str) -> Dict[str, Any]:
        """Get the configuration dictionary for a specific entity name."""
        if entity_name not in self.entity_configs:
            raise KeyError(f"Entity name '{entity_name}' not found in configuration")

        return self.entity_configs[entity_name]

    def get_entity_type(self, entity_name: str) -> EntityType:
        """Get the EntityType instance for a specific entity name."""
        return self.get_entity_config(entity_name)["entity_type"]

    def get_compiled_pattern(self, entity_name: str) -> Pattern:
        """Get the compiled regex pattern for a specific entity name."""
        return self.get_entity_config(entity_name)["compiled_pattern"]

    def get_description_template(self, entity_name: str) -> str:
        """Get the description template for a specific entity name."""
        return self.get_entity_config(entity_name)["description_template"]
