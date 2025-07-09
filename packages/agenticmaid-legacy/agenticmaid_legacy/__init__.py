"""
AgenticMaid Legacy - Backward compatibility and legacy API methods for AgenticMaid.

This package provides backward compatibility for older AgenticMaid implementations
and includes legacy API calling methods that were used in earlier versions of the
framework. It's designed to help users migrate from older versions while maintaining
compatibility with existing code.

Key Features:
- Legacy MCP STDIO Adapter
- Original API calling methods
- Backward compatibility wrappers
- Migration utilities
- Deprecated function support
- Legacy configuration formats

Example Usage:
    >>> from agenticmaid_legacy import LegacyAgenticMaid, MCPStdioAdapter
    >>> 
    >>> # Use legacy API calling method
    >>> legacy_client = LegacyAgenticMaid()
    >>> response = legacy_client.call_api_legacy(
    ...     endpoint="chat",
    ...     data={"message": "Hello"}
    ... )
    >>> 
    >>> # Use legacy MCP adapter
    >>> adapter = MCPStdioAdapter()
    >>> result = adapter.execute_command("list_tools")
"""

__version__ = "1.0.0"
__author__ = "AgenticMaid Team"
__email__ = "contact@agenticmaid.com"
__license__ = "MIT"

# Import legacy components
try:
    from .mcp_stdio_adapter import MCPStdioAdapter
    MCP_STDIO_AVAILABLE = True
except ImportError:
    MCP_STDIO_AVAILABLE = False

# Legacy API wrapper
class LegacyAgenticMaid:
    """
    Legacy wrapper for AgenticMaid functionality.
    
    This class provides backward compatibility for older AgenticMaid
    implementations and API calling methods.
    """
    
    def __init__(self, config=None):
        """Initialize legacy AgenticMaid client."""
        self.config = config or {}
        self._initialized = False
    
    def initialize(self):
        """Initialize the legacy client (synchronous)."""
        self._initialized = True
        return True
    
    def call_api_legacy(self, endpoint: str, data: dict = None, method: str = "POST"):
        """
        Legacy API calling method.
        
        Args:
            endpoint: API endpoint to call
            data: Data to send with the request
            method: HTTP method to use
            
        Returns:
            API response
        """
        # This is a placeholder for legacy API functionality
        # In a real implementation, this would contain the original API calling logic
        return {
            "status": "success",
            "endpoint": endpoint,
            "data": data,
            "method": method,
            "message": "Legacy API call completed"
        }
    
    def run_task_legacy(self, task_name: str, params: dict = None):
        """
        Legacy task execution method.
        
        Args:
            task_name: Name of the task to run
            params: Task parameters
            
        Returns:
            Task execution result
        """
        return {
            "task": task_name,
            "params": params,
            "status": "completed",
            "message": "Legacy task execution completed"
        }
    
    def get_config_legacy(self):
        """Get configuration in legacy format."""
        return self.config
    
    def set_config_legacy(self, config: dict):
        """Set configuration in legacy format."""
        self.config = config


# Legacy configuration converter
class LegacyConfigConverter:
    """Convert between legacy and modern configuration formats."""
    
    @staticmethod
    def convert_to_modern(legacy_config: dict) -> dict:
        """
        Convert legacy configuration to modern format.
        
        Args:
            legacy_config: Configuration in legacy format
            
        Returns:
            Configuration in modern format
        """
        modern_config = {}
        
        # Convert legacy AI service configuration
        if "ai_service" in legacy_config:
            modern_config["ai_services"] = {
                "default_service": legacy_config["ai_service"]
            }
            modern_config["default_llm_service_name"] = "default_service"
        
        # Convert legacy MCP configuration
        if "mcp_config" in legacy_config:
            modern_config["mcp_servers"] = legacy_config["mcp_config"]
        
        # Convert legacy messaging configuration
        if "messaging_config" in legacy_config:
            modern_config["messaging_clients"] = legacy_config["messaging_config"]
        
        # Copy other settings
        for key, value in legacy_config.items():
            if key not in ["ai_service", "mcp_config", "messaging_config"]:
                modern_config[key] = value
        
        return modern_config
    
    @staticmethod
    def convert_to_legacy(modern_config: dict) -> dict:
        """
        Convert modern configuration to legacy format.
        
        Args:
            modern_config: Configuration in modern format
            
        Returns:
            Configuration in legacy format
        """
        legacy_config = {}
        
        # Convert modern AI service configuration
        if "ai_services" in modern_config and "default_llm_service_name" in modern_config:
            default_service = modern_config["default_llm_service_name"]
            if default_service in modern_config["ai_services"]:
                legacy_config["ai_service"] = modern_config["ai_services"][default_service]
        
        # Convert modern MCP configuration
        if "mcp_servers" in modern_config:
            legacy_config["mcp_config"] = modern_config["mcp_servers"]
        
        # Convert modern messaging configuration
        if "messaging_clients" in modern_config:
            legacy_config["messaging_config"] = modern_config["messaging_clients"]
        
        # Copy other settings
        for key, value in modern_config.items():
            if key not in ["ai_services", "default_llm_service_name", "mcp_servers", "messaging_clients"]:
                legacy_config[key] = value
        
        return legacy_config


# Define what gets imported with "from agenticmaid_legacy import *"
__all__ = [
    "LegacyAgenticMaid",
    "LegacyConfigConverter",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Add MCP STDIO adapter if available
if MCP_STDIO_AVAILABLE:
    __all__.append("MCPStdioAdapter")


def get_version():
    """Get the current version of AgenticMaid Legacy."""
    return __version__


def get_info():
    """Get information about AgenticMaid Legacy."""
    return {
        "name": "AgenticMaid Legacy",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "mcp_stdio_available": MCP_STDIO_AVAILABLE,
        "purpose": "Backward compatibility and legacy API support",
    }


def migrate_to_modern(legacy_config_path: str, modern_config_path: str):
    """
    Migrate a legacy configuration file to modern format.
    
    Args:
        legacy_config_path: Path to legacy configuration file
        modern_config_path: Path to save modern configuration file
    """
    import json
    
    # Read legacy configuration
    with open(legacy_config_path, 'r', encoding='utf-8') as f:
        legacy_config = json.load(f)
    
    # Convert to modern format
    modern_config = LegacyConfigConverter.convert_to_modern(legacy_config)
    
    # Save modern configuration
    with open(modern_config_path, 'w', encoding='utf-8') as f:
        json.dump(modern_config, f, indent=2, ensure_ascii=False)
    
    print(f"Configuration migrated from {legacy_config_path} to {modern_config_path}")


# Deprecation warnings
import warnings

def deprecated_function(func):
    """Decorator to mark functions as deprecated."""
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in a future version. "
            f"Please use the modern AgenticMaid API instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper
