"""
AgenticMaid Triggers - Event trigger implementations for AgenticMaid.

This package provides various trigger implementations that can be used
with the AgenticMaid framework to enable event-driven automation based on
external events such as blockchain transactions, time schedules, file changes,
API webhooks, and more.

Key Features:
- Solana Wallet Monitoring
- Time-based Triggers (Cron-like scheduling)
- File System Watchers
- HTTP Webhook Triggers
- Custom Trigger Interface for extensions
- Event filtering and processing
- Async event handling

Example Usage:
    >>> from agenticmaid_triggers import SolanaWalletTrigger, TriggerEvent
    >>> 
    >>> # Initialize Solana wallet monitor
    >>> config = {
    ...     "wallet_addresses": ["wallet_address_1", "wallet_address_2"],
    ...     "rpc_endpoint": "https://api.mainnet-beta.solana.com",
    ...     "check_interval": 30,
    ...     "min_sol_amount": 0.1
    ... }
    >>> 
    >>> trigger = SolanaWalletTrigger()
    >>> await trigger.initialize(config)
    >>> 
    >>> # Start monitoring
    >>> await trigger.start_monitoring()
"""

__version__ = "1.0.0"
__author__ = "AgenticMaid Team"
__email__ = "contact@agenticmaid.com"
__license__ = "MIT"

# Import core trigger components
from .core.trigger_interface import TriggerInterface
from .core.trigger_event import TriggerEvent
from .core.message import Message, MessageType

# Import available triggers
try:
    from .triggers.solana_wallet_trigger import SolanaWalletTrigger
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False

try:
    from .triggers.time_trigger import TimeTrigger
    TIME_AVAILABLE = True
except ImportError:
    TIME_AVAILABLE = False

try:
    from .triggers.file_watcher_trigger import FileWatcherTrigger
    FILE_WATCHER_AVAILABLE = True
except ImportError:
    FILE_WATCHER_AVAILABLE = False

try:
    from .triggers.webhook_trigger import WebhookTrigger
    WEBHOOK_AVAILABLE = True
except ImportError:
    WEBHOOK_AVAILABLE = False

# Define what gets imported with "from agenticmaid_triggers import *"
__all__ = [
    "TriggerInterface",
    "TriggerEvent",
    "Message",
    "MessageType",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Add available triggers to __all__
if SOLANA_AVAILABLE:
    __all__.append("SolanaWalletTrigger")

if TIME_AVAILABLE:
    __all__.append("TimeTrigger")

if FILE_WATCHER_AVAILABLE:
    __all__.append("FileWatcherTrigger")

if WEBHOOK_AVAILABLE:
    __all__.append("WebhookTrigger")


def get_version():
    """Get the current version of AgenticMaid Triggers."""
    return __version__


def get_available_triggers():
    """Get a list of available trigger implementations."""
    triggers = []
    
    if SOLANA_AVAILABLE:
        triggers.append("SolanaWalletTrigger")
    
    if TIME_AVAILABLE:
        triggers.append("TimeTrigger")
    
    if FILE_WATCHER_AVAILABLE:
        triggers.append("FileWatcherTrigger")
    
    if WEBHOOK_AVAILABLE:
        triggers.append("WebhookTrigger")
    
    return triggers


def get_info():
    """Get information about AgenticMaid Triggers."""
    return {
        "name": "AgenticMaid Triggers",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "available_triggers": get_available_triggers(),
        "solana_available": SOLANA_AVAILABLE,
        "time_available": TIME_AVAILABLE,
        "file_watcher_available": FILE_WATCHER_AVAILABLE,
        "webhook_available": WEBHOOK_AVAILABLE,
    }


# Trigger factory function
def create_trigger(trigger_type: str, config: dict = None):
    """
    Factory function to create trigger instances.
    
    Args:
        trigger_type: Type of trigger to create ('solana_wallet', 'time', 'file_watcher', 'webhook')
        config: Configuration dictionary for the trigger
        
    Returns:
        Trigger instance
        
    Raises:
        ValueError: If trigger type is not available or unknown
    """
    trigger_type = trigger_type.lower()
    
    if trigger_type == "solana_wallet":
        if not SOLANA_AVAILABLE:
            raise ValueError("Solana wallet trigger is not available. Install with: pip install agenticmaid-triggers[solana]")
        return SolanaWalletTrigger()
    
    elif trigger_type == "time":
        if not TIME_AVAILABLE:
            raise ValueError("Time trigger is not available.")
        return TimeTrigger()
    
    elif trigger_type == "file_watcher":
        if not FILE_WATCHER_AVAILABLE:
            raise ValueError("File watcher trigger is not available. Install with: pip install agenticmaid-triggers[file_watcher]")
        return FileWatcherTrigger()
    
    elif trigger_type == "webhook":
        if not WEBHOOK_AVAILABLE:
            raise ValueError("Webhook trigger is not available. Install with: pip install agenticmaid-triggers[webhook]")
        return WebhookTrigger()
    
    else:
        available = ", ".join(get_available_triggers())
        raise ValueError(f"Unknown trigger type '{trigger_type}'. Available triggers: {available}")
