"""
Client factory for creating messaging client instances.

This module provides a factory pattern implementation for creating different
types of messaging clients based on configuration.
"""

from typing import Dict, Any, Type
import logging

from .base_client import BaseClient
from ..core.exceptions import ConfigurationError, ValidationError
from ..core.config_validator import ConfigValidator


class ClientFactory:
    """Factory for creating messaging client instances."""
    
    # Registry of available client types
    _client_types: Dict[str, Type[BaseClient]] = {}
    
    @classmethod
    def register_client_type(cls, client_type: str, client_class: Type[BaseClient]) -> None:
        """
        Register a new client type.
        
        Args:
            client_type: String identifier for the client type
            client_class: Class that implements BaseClient
        """
        if not issubclass(client_class, BaseClient):
            raise ValueError(f"Client class must inherit from BaseClient")
        
        cls._client_types[client_type] = client_class
        logging.getLogger(__name__).info(f"Registered client type: {client_type}")
    
    @classmethod
    def unregister_client_type(cls, client_type: str) -> None:
        """
        Unregister a client type.
        
        Args:
            client_type: String identifier for the client type
        """
        if client_type in cls._client_types:
            del cls._client_types[client_type]
            logging.getLogger(__name__).info(f"Unregistered client type: {client_type}")
    
    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of available client types."""
        return list(cls._client_types.keys())
    
    @classmethod
    def create_client(cls, client_id: str, client_type: str, config: Dict[str, Any]) -> BaseClient:
        """
        Create a client instance.
        
        Args:
            client_id: Unique identifier for the client
            client_type: Type of client to create
            config: Configuration dictionary for the client
            
        Returns:
            Configured client instance
            
        Raises:
            ConfigurationError: If client type is not supported or config is invalid
            ValidationError: If configuration validation fails
        """
        logger = logging.getLogger(__name__)
        
        # Validate client type
        if client_type not in cls._client_types:
            available_types = ", ".join(cls.get_available_types())
            raise ConfigurationError(
                f"Unsupported client type: {client_type}. "
                f"Available types: {available_types}"
            )
        
        # Validate configuration
        try:
            ConfigValidator.validate_client_config(client_type, config)
        except ValidationError as e:
            raise ConfigurationError(f"Invalid configuration for {client_type} client: {e}")
        
        # Create client instance
        client_class = cls._client_types[client_type]
        try:
            client = client_class(client_id, config)
            logger.info(f"Created {client_type} client with ID: {client_id}")
            return client
        except Exception as e:
            raise ConfigurationError(f"Failed to create {client_type} client: {e}")
    
    @classmethod
    def create_clients_from_config(cls, clients_config: Dict[str, Dict[str, Any]]) -> Dict[str, BaseClient]:
        """
        Create multiple clients from configuration.
        
        Args:
            clients_config: Dictionary mapping client IDs to their configurations
            
        Returns:
            Dictionary mapping client IDs to client instances
            
        Raises:
            ConfigurationError: If any client configuration is invalid
        """
        clients = {}
        logger = logging.getLogger(__name__)
        
        for client_id, client_config in clients_config.items():
            try:
                # Extract client type and configuration
                if "type" not in client_config:
                    raise ConfigurationError(f"Client {client_id} missing 'type' field")
                
                client_type = client_config["type"]
                enabled = client_config.get("enabled", True)
                
                if not enabled:
                    logger.info(f"Skipping disabled client: {client_id}")
                    continue
                
                config = client_config.get("config", {})
                
                # Create client
                client = cls.create_client(client_id, client_type, config)
                clients[client_id] = client
                
            except Exception as e:
                logger.error(f"Failed to create client {client_id}: {e}")
                raise ConfigurationError(f"Failed to create client {client_id}: {e}")
        
        logger.info(f"Created {len(clients)} clients")
        return clients


# Auto-register built-in client types when module is imported
def _register_builtin_clients():
    """Register built-in client types."""
    try:
        # Import and register Telegram client
        from .telegram_client import TelegramClient
        ClientFactory.register_client_type("telegram", TelegramClient)
    except ImportError:
        logging.getLogger(__name__).warning("Telegram client not available")
    
    # Future client types can be registered here
    # try:
    #     from .discord_client import DiscordClient
    #     ClientFactory.register_client_type("discord", DiscordClient)
    # except ImportError:
    #     pass


# Register built-in clients when module is loaded
_register_builtin_clients()
