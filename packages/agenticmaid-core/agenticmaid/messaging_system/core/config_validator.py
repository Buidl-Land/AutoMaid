"""
Configuration validation for the messaging system.

This module provides validation utilities for client and trigger configurations
to ensure proper setup and prevent runtime errors.
"""

from typing import Dict, Any, List, Optional
import re
from .exceptions import ValidationError, ConfigurationError


class ConfigValidator:
    """Validates configuration for messaging system components."""
    
    # Required fields for different component types
    CLIENT_REQUIRED_FIELDS = {
        "telegram": ["bot_token"],
        "discord": ["bot_token"],
        "slack": ["bot_token", "app_token"],
        "websocket": ["endpoint"]
    }
    
    TRIGGER_REQUIRED_FIELDS = {
        "solana_wallet": ["wallet_address", "rpc_endpoint"],
        "price_alert": ["symbol", "threshold"],
        "smart_contract": ["contract_address", "rpc_endpoint"],
        "scheduled": ["cron_expression"]
    }
    
    @classmethod
    def validate_client_config(cls, client_type: str, config: Dict[str, Any]) -> None:
        """
        Validate client configuration.
        
        Args:
            client_type: Type of client (e.g., "telegram", "discord")
            config: Configuration dictionary
            
        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValidationError(f"Client config must be a dictionary, got {type(config)}")
        
        # Check required fields
        required_fields = cls.CLIENT_REQUIRED_FIELDS.get(client_type, [])
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required field '{field}' for {client_type} client")
            
            value = config[field]
            if not value or (isinstance(value, str) and not value.strip()):
                raise ValidationError(f"Field '{field}' cannot be empty for {client_type} client")
        
        # Type-specific validation
        if client_type == "telegram":
            cls._validate_telegram_config(config)
        elif client_type == "discord":
            cls._validate_discord_config(config)
        elif client_type == "slack":
            cls._validate_slack_config(config)
        elif client_type == "websocket":
            cls._validate_websocket_config(config)
    
    @classmethod
    def validate_trigger_config(cls, trigger_type: str, config: Dict[str, Any]) -> None:
        """
        Validate trigger configuration.
        
        Args:
            trigger_type: Type of trigger (e.g., "solana_wallet", "price_alert")
            config: Configuration dictionary
            
        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValidationError(f"Trigger config must be a dictionary, got {type(config)}")
        
        # Check required fields
        required_fields = cls.TRIGGER_REQUIRED_FIELDS.get(trigger_type, [])
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required field '{field}' for {trigger_type} trigger")
            
            value = config[field]
            if not value or (isinstance(value, str) and not value.strip()):
                raise ValidationError(f"Field '{field}' cannot be empty for {trigger_type} trigger")
        
        # Type-specific validation
        if trigger_type == "solana_wallet":
            cls._validate_solana_wallet_config(config)
        elif trigger_type == "price_alert":
            cls._validate_price_alert_config(config)
        elif trigger_type == "smart_contract":
            cls._validate_smart_contract_config(config)
        elif trigger_type == "scheduled":
            cls._validate_scheduled_config(config)
    
    @classmethod
    def validate_agent_mapping(cls, agent_mapping: Dict[str, str]) -> None:
        """
        Validate agent mapping configuration.
        
        Args:
            agent_mapping: Dictionary mapping event types to agent IDs
            
        Raises:
            ValidationError: If agent mapping is invalid
        """
        if not isinstance(agent_mapping, dict):
            raise ValidationError(f"Agent mapping must be a dictionary, got {type(agent_mapping)}")
        
        for event_type, agent_id in agent_mapping.items():
            if not isinstance(event_type, str) or not event_type.strip():
                raise ValidationError("Event type must be a non-empty string")
            
            if not isinstance(agent_id, str) or not agent_id.strip():
                raise ValidationError(f"Agent ID for event '{event_type}' must be a non-empty string")
    
    @classmethod
    def _validate_telegram_config(cls, config: Dict[str, Any]) -> None:
        """Validate Telegram-specific configuration."""
        bot_token = config["bot_token"]
        
        # Basic token format validation (Telegram bot tokens have a specific format)
        if not re.match(r'^\d+:[A-Za-z0-9_-]+$', bot_token):
            raise ValidationError("Invalid Telegram bot token format")
        
        # Validate optional fields
        if "allowed_users" in config:
            allowed_users = config["allowed_users"]
            if not isinstance(allowed_users, list):
                raise ValidationError("allowed_users must be a list")
            
            for user in allowed_users:
                if not isinstance(user, (str, int)):
                    raise ValidationError("Each allowed user must be a string or integer")
        
        if "max_connections" in config:
            max_conn = config["max_connections"]
            if not isinstance(max_conn, int) or max_conn <= 0:
                raise ValidationError("max_connections must be a positive integer")
        
        if "timeout" in config:
            timeout = config["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValidationError("timeout must be a positive number")
    
    @classmethod
    def _validate_discord_config(cls, config: Dict[str, Any]) -> None:
        """Validate Discord-specific configuration."""
        bot_token = config["bot_token"]
        
        # Discord bot tokens are typically longer and have a different format
        if len(bot_token) < 50:
            raise ValidationError("Discord bot token appears to be too short")
        
        if "guild_id" in config:
            guild_id = config["guild_id"]
            if not isinstance(guild_id, (str, int)):
                raise ValidationError("guild_id must be a string or integer")
    
    @classmethod
    def _validate_slack_config(cls, config: Dict[str, Any]) -> None:
        """Validate Slack-specific configuration."""
        bot_token = config["bot_token"]
        app_token = config["app_token"]
        
        # Slack bot tokens start with "xoxb-"
        if not bot_token.startswith("xoxb-"):
            raise ValidationError("Slack bot token must start with 'xoxb-'")
        
        # Slack app tokens start with "xapp-"
        if not app_token.startswith("xapp-"):
            raise ValidationError("Slack app token must start with 'xapp-'")
    
    @classmethod
    def _validate_websocket_config(cls, config: Dict[str, Any]) -> None:
        """Validate WebSocket-specific configuration."""
        endpoint = config["endpoint"]
        
        # Basic URL validation
        if not (endpoint.startswith("ws://") or endpoint.startswith("wss://")):
            raise ValidationError("WebSocket endpoint must start with 'ws://' or 'wss://'")
    
    @classmethod
    def _validate_solana_wallet_config(cls, config: Dict[str, Any]) -> None:
        """Validate Solana wallet trigger configuration."""
        wallet_address = config["wallet_address"]
        rpc_endpoint = config["rpc_endpoint"]
        
        # Basic Solana address validation (base58, 32-44 characters)
        if not re.match(r'^[1-9A-HJ-NP-Za-km-z]{32,44}$', wallet_address):
            raise ValidationError("Invalid Solana wallet address format")
        
        # Basic RPC endpoint validation
        if not (rpc_endpoint.startswith("http://") or rpc_endpoint.startswith("https://")):
            raise ValidationError("Solana RPC endpoint must be a valid HTTP/HTTPS URL")
        
        # Validate optional fields
        if "check_interval" in config:
            interval = config["check_interval"]
            if not isinstance(interval, (int, float)) or interval <= 0:
                raise ValidationError("check_interval must be a positive number")
        
        if "min_transaction_amount" in config:
            min_amount = config["min_transaction_amount"]
            if not isinstance(min_amount, (int, float)) or min_amount < 0:
                raise ValidationError("min_transaction_amount must be a non-negative number")
    
    @classmethod
    def _validate_price_alert_config(cls, config: Dict[str, Any]) -> None:
        """Validate price alert trigger configuration."""
        symbol = config["symbol"]
        threshold = config["threshold"]
        
        # Basic symbol validation
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValidationError("Symbol must be a non-empty string")
        
        # Threshold validation
        if not isinstance(threshold, (int, float)):
            raise ValidationError("Threshold must be a number")
        
        if "operator" in config:
            operator = config["operator"]
            valid_operators = [">=", "<=", ">", "<", "==", "!="]
            if operator not in valid_operators:
                raise ValidationError(f"Operator must be one of: {valid_operators}")
    
    @classmethod
    def _validate_smart_contract_config(cls, config: Dict[str, Any]) -> None:
        """Validate smart contract trigger configuration."""
        contract_address = config["contract_address"]
        rpc_endpoint = config["rpc_endpoint"]
        
        # Basic contract address validation (Ethereum-style)
        if not re.match(r'^0x[a-fA-F0-9]{40}$', contract_address):
            raise ValidationError("Invalid contract address format")
        
        # Basic RPC endpoint validation
        if not (rpc_endpoint.startswith("http://") or rpc_endpoint.startswith("https://")):
            raise ValidationError("RPC endpoint must be a valid HTTP/HTTPS URL")
    
    @classmethod
    def _validate_scheduled_config(cls, config: Dict[str, Any]) -> None:
        """Validate scheduled trigger configuration."""
        cron_expression = config["cron_expression"]
        
        # Basic cron expression validation (5 or 6 fields)
        fields = cron_expression.split()
        if len(fields) not in [5, 6]:
            raise ValidationError("Cron expression must have 5 or 6 fields")
        
        # More detailed cron validation could be added here
        # For now, we'll do basic field count validation
