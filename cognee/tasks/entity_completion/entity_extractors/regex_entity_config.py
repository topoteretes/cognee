import json
from cognee.shared.logging_utils import get_logger
import os
import re
from typing import Dict, List, Pattern, Any

from cognee.modules.engine.models.EntityType import EntityType
from cognee.root_dir import get_absolute_path

logger = get_logger("regex_entity_config")


class RegexEntityConfig:
    """Class to load and process regex entity extraction configuration."""

    def __init__(self, config_path: str):
        """Initialize the regex entity configuration with the config path."""
        self.config_path = config_path
        self.entity_configs = {}
        self._load_config()

    def _validate_config_fields(self, config: Dict[str, Any]) -> None:
        """Validate that all required fields are present in the configuration."""
        required_fields = ["entity_name", "entity_description", "regex", "description_template"]
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            raise ValueError(
                f"Missing required fields in entity configuration: {', '.join(missing_fields)}"
            )

    def _compile_regex(self, pattern: str, entity_name: str) -> Pattern:
        """Compile a regex pattern safely, with error handling."""
        try:
            return re.compile(pattern)
        except re.error as e:
            logger.error(f"Invalid regex pattern for entity '{entity_name}': {str(e)}")
            raise ValueError(f"Invalid regex pattern for entity '{entity_name}': {str(e)}")

    def _load_config(self) -> None:
        """Load and process the configuration from the JSON file."""
        try:
            with open(self.config_path, "r") as f:
                config_list = json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {self.config_path}: {str(e)}")
            raise ValueError(f"Invalid JSON in config file: {str(e)}")

        for config in config_list:
            self._validate_config_fields(config)
            entity_name = config["entity_name"]

            entity_type = EntityType(name=entity_name, description=config["entity_description"])

            compiled_pattern = self._compile_regex(config["regex"], entity_name)

            self.entity_configs[entity_name] = {
                "entity_type": entity_type,
                "regex": config["regex"],
                "compiled_pattern": compiled_pattern,
                "description_template": config["description_template"],
            }

        logger.info(
            f"Loaded {len(self.entity_configs)} entity configurations from {self.config_path}"
        )

    def get_entity_names(self) -> List[str]:
        """Return a list of all configured entity names."""
        return list(self.entity_configs.keys())

    def get_entity_config(self, entity_name: str) -> Dict[str, Any]:
        """Get the configuration for a specific entity type."""
        if entity_name not in self.entity_configs:
            raise KeyError(f"Unknown entity type: {entity_name}")
        return self.entity_configs[entity_name]

    def get_entity_type(self, entity_name: str) -> EntityType:
        """Get the EntityType object for a specific entity type."""
        return self.get_entity_config(entity_name)["entity_type"]

    def get_compiled_pattern(self, entity_name: str) -> Pattern:
        """Get the compiled regex pattern for a specific entity type."""
        return self.get_entity_config(entity_name)["compiled_pattern"]

    def get_description_template(self, entity_name: str) -> str:
        """Get the description template for a specific entity type."""
        return self.get_entity_config(entity_name)["description_template"]
