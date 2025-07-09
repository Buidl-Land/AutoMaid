"""
Trigger factory for creating trigger instances.

This module provides a factory pattern implementation for creating different
types of triggers based on configuration.
"""

from typing import Dict, Any, Type
import logging

from .base_trigger import BaseTrigger
from ..core.exceptions import ConfigurationError, ValidationError
from ..core.config_validator import ConfigValidator


class TriggerFactory:
    """Factory for creating trigger instances."""
    
    # Registry of available trigger types
    _trigger_types: Dict[str, Type[BaseTrigger]] = {}
    
    @classmethod
    def register_trigger_type(cls, trigger_type: str, trigger_class: Type[BaseTrigger]) -> None:
        """
        Register a new trigger type.
        
        Args:
            trigger_type: String identifier for the trigger type
            trigger_class: Class that implements BaseTrigger
        """
        if not issubclass(trigger_class, BaseTrigger):
            raise ValueError(f"Trigger class must inherit from BaseTrigger")
        
        cls._trigger_types[trigger_type] = trigger_class
        logging.getLogger(__name__).info(f"Registered trigger type: {trigger_type}")
    
    @classmethod
    def unregister_trigger_type(cls, trigger_type: str) -> None:
        """
        Unregister a trigger type.
        
        Args:
            trigger_type: String identifier for the trigger type
        """
        if trigger_type in cls._trigger_types:
            del cls._trigger_types[trigger_type]
            logging.getLogger(__name__).info(f"Unregistered trigger type: {trigger_type}")
    
    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of available trigger types."""
        return list(cls._trigger_types.keys())
    
    @classmethod
    def create_trigger(cls, trigger_id: str, trigger_type: str, config: Dict[str, Any]) -> BaseTrigger:
        """
        Create a trigger instance.
        
        Args:
            trigger_id: Unique identifier for the trigger
            trigger_type: Type of trigger to create
            config: Configuration dictionary for the trigger
            
        Returns:
            Configured trigger instance
            
        Raises:
            ConfigurationError: If trigger type is not supported or config is invalid
            ValidationError: If configuration validation fails
        """
        logger = logging.getLogger(__name__)
        
        # Validate trigger type
        if trigger_type not in cls._trigger_types:
            available_types = ", ".join(cls.get_available_types())
            raise ConfigurationError(
                f"Unsupported trigger type: {trigger_type}. "
                f"Available types: {available_types}"
            )
        
        # Validate configuration
        try:
            ConfigValidator.validate_trigger_config(trigger_type, config)
        except ValidationError as e:
            raise ConfigurationError(f"Invalid configuration for {trigger_type} trigger: {e}")
        
        # Validate agent mapping if present
        if "agent_mapping" in config:
            try:
                ConfigValidator.validate_agent_mapping(config["agent_mapping"])
            except ValidationError as e:
                raise ConfigurationError(f"Invalid agent mapping for {trigger_type} trigger: {e}")
        
        # Create trigger instance
        trigger_class = cls._trigger_types[trigger_type]
        try:
            trigger = trigger_class(trigger_id, config)
            logger.info(f"Created {trigger_type} trigger with ID: {trigger_id}")
            return trigger
        except Exception as e:
            raise ConfigurationError(f"Failed to create {trigger_type} trigger: {e}")
    
    @classmethod
    def create_triggers_from_config(cls, triggers_config: Dict[str, Dict[str, Any]]) -> Dict[str, BaseTrigger]:
        """
        Create multiple triggers from configuration.
        
        Args:
            triggers_config: Dictionary mapping trigger IDs to their configurations
            
        Returns:
            Dictionary mapping trigger IDs to trigger instances
            
        Raises:
            ConfigurationError: If any trigger configuration is invalid
        """
        triggers = {}
        logger = logging.getLogger(__name__)
        
        for trigger_id, trigger_config in triggers_config.items():
            try:
                # Extract trigger type and configuration
                if "type" not in trigger_config:
                    raise ConfigurationError(f"Trigger {trigger_id} missing 'type' field")
                
                trigger_type = trigger_config["type"]
                enabled = trigger_config.get("enabled", True)
                
                if not enabled:
                    logger.info(f"Skipping disabled trigger: {trigger_id}")
                    continue
                
                config = trigger_config.get("config", {})
                
                # Add agent mapping to config if present at trigger level
                if "agent_mapping" in trigger_config:
                    config["agent_mapping"] = trigger_config["agent_mapping"]
                
                # Create trigger
                trigger = cls.create_trigger(trigger_id, trigger_type, config)
                triggers[trigger_id] = trigger
                
            except Exception as e:
                logger.error(f"Failed to create trigger {trigger_id}: {e}")
                raise ConfigurationError(f"Failed to create trigger {trigger_id}: {e}")
        
        logger.info(f"Created {len(triggers)} triggers")
        return triggers


# Auto-register built-in trigger types when module is imported
def _register_builtin_triggers():
    """Register built-in trigger types."""
    try:
        # Import and register Solana wallet trigger
        from .solana_wallet_trigger import SolanaWalletTrigger
        TriggerFactory.register_trigger_type("solana_wallet", SolanaWalletTrigger)
    except ImportError:
        logging.getLogger(__name__).warning("Solana wallet trigger not available")
    
    # Future trigger types can be registered here
    # try:
    #     from .price_alert_trigger import PriceAlertTrigger
    #     TriggerFactory.register_trigger_type("price_alert", PriceAlertTrigger)
    # except ImportError:
    #     pass


# Register built-in triggers when module is loaded
_register_builtin_triggers()
