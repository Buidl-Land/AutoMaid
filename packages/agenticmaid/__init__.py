"""
AgenticMaid - Complete ecosystem meta-package.

This package provides convenient imports for the entire AgenticMaid ecosystem.
It automatically imports from all available sub-packages and provides a unified interface.
"""

__version__ = "1.0.0"
__author__ = "AgenticMaid Team"
__email__ = "contact@agenticmaid.com"
__license__ = "MIT"

# Import core components (always available)
try:
    from agenticmaid_core import AgenticMaid, ConfigManager, get_conversation_logger
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Import client components (optional)
try:
    from agenticmaid_clients import (
        TelegramClient, Message, MessageType, TriggerEvent,
        create_client as create_messaging_client,
        get_available_clients
    )
    CLIENTS_AVAILABLE = True
except ImportError:
    CLIENTS_AVAILABLE = False

# Import trigger components (optional)
try:
    from agenticmaid_triggers import (
        SolanaWalletTrigger, TimeTrigger, FileWatcherTrigger,
        create_trigger, get_available_triggers
    )
    TRIGGERS_AVAILABLE = True
except ImportError:
    TRIGGERS_AVAILABLE = False

# Import legacy components (optional)
try:
    from agenticmaid_legacy import (
        LegacyAgenticMaid, LegacyConfigConverter, migrate_to_modern
    )
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

# Define what gets imported with "from agenticmaid import *"
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Add core components if available
if CORE_AVAILABLE:
    __all__.extend([
        "AgenticMaid",
        "ConfigManager",
        "get_conversation_logger",
    ])

# Add client components if available
if CLIENTS_AVAILABLE:
    __all__.extend([
        "TelegramClient",
        "Message",
        "MessageType",
        "TriggerEvent",
        "create_messaging_client",
        "get_available_clients",
    ])

# Add trigger components if available
if TRIGGERS_AVAILABLE:
    __all__.extend([
        "SolanaWalletTrigger",
        "TimeTrigger",
        "FileWatcherTrigger",
        "create_trigger",
        "get_available_triggers",
    ])

# Add legacy components if available
if LEGACY_AVAILABLE:
    __all__.extend([
        "LegacyAgenticMaid",
        "LegacyConfigConverter",
        "migrate_to_modern",
    ])


def get_version():
    """Get the current version of AgenticMaid."""
    return __version__


def get_ecosystem_info():
    """Get information about the AgenticMaid ecosystem."""
    info = {
        "name": "AgenticMaid Ecosystem",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "components": {
            "core": CORE_AVAILABLE,
            "clients": CLIENTS_AVAILABLE,
            "triggers": TRIGGERS_AVAILABLE,
            "legacy": LEGACY_AVAILABLE,
        }
    }
    
    if CLIENTS_AVAILABLE:
        info["available_clients"] = get_available_clients()
    
    if TRIGGERS_AVAILABLE:
        info["available_triggers"] = get_available_triggers()
    
    return info


def check_installation():
    """Check which AgenticMaid components are installed."""
    components = {
        "agenticmaid-core": CORE_AVAILABLE,
        "agenticmaid-clients": CLIENTS_AVAILABLE,
        "agenticmaid-triggers": TRIGGERS_AVAILABLE,
        "agenticmaid-legacy": LEGACY_AVAILABLE,
    }
    
    print("AgenticMaid Installation Status:")
    print("=" * 40)
    
    for component, available in components.items():
        status = "✅ Installed" if available else "❌ Not installed"
        print(f"{component:<25} {status}")
    
    if not CORE_AVAILABLE:
        print("\n⚠️  Core package not found. Install with: pip install agenticmaid-core")
    
    if not any([CLIENTS_AVAILABLE, TRIGGERS_AVAILABLE]):
        print("\n💡 Consider installing additional components:")
        if not CLIENTS_AVAILABLE:
            print("   - Messaging clients: pip install agenticmaid-clients[telegram]")
        if not TRIGGERS_AVAILABLE:
            print("   - Event triggers: pip install agenticmaid-triggers[solana]")
    
    print(f"\nTotal components installed: {sum(components.values())}/4")
    return components


def create_agent(config=None, **kwargs):
    """
    Convenience function to create an AgenticMaid agent.
    
    Args:
        config: Configuration dictionary or file path
        **kwargs: Additional configuration options
        
    Returns:
        AgenticMaid instance
        
    Raises:
        ImportError: If core package is not installed
    """
    if not CORE_AVAILABLE:
        raise ImportError(
            "AgenticMaid core package is not installed. "
            "Install with: pip install agenticmaid-core"
        )
    
    if config is None:
        config = {}
    
    # Merge kwargs into config
    if kwargs:
        if isinstance(config, dict):
            config.update(kwargs)
        else:
            # If config is a file path, we can't merge kwargs
            raise ValueError(
                "Cannot merge kwargs when config is a file path. "
                "Use a dictionary config instead."
            )
    
    return AgenticMaid(config)


# Convenience aliases
create_client = create_messaging_client if CLIENTS_AVAILABLE else None

# Version compatibility
def get_info():
    """Alias for get_ecosystem_info() for backward compatibility."""
    return get_ecosystem_info()


# Auto-check installation on import (optional, can be disabled)
import os
if os.environ.get("AGENTICMAID_SILENT_IMPORT", "").lower() not in ("1", "true", "yes"):
    if not CORE_AVAILABLE:
        print("⚠️  AgenticMaid core package not found. Install with: pip install agenticmaid-core")


# Export main classes for easy access
if CORE_AVAILABLE:
    # Re-export main classes at package level
    AgenticMaid = AgenticMaid
    ConfigManager = ConfigManager

if CLIENTS_AVAILABLE:
    TelegramClient = TelegramClient
    Message = Message
    MessageType = MessageType

if TRIGGERS_AVAILABLE:
    SolanaWalletTrigger = SolanaWalletTrigger
    TimeTrigger = TimeTrigger

if LEGACY_AVAILABLE:
    LegacyAgenticMaid = LegacyAgenticMaid
