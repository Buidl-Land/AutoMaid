"""
AgenticMaid Clients - Messaging client implementations for AgenticMaid.

This package provides various messaging client implementations that can be used
with the AgenticMaid framework to enable communication through different platforms
such as Telegram, Discord, Slack, and more.

Key Features:
- Telegram Bot Client
- Discord Bot Client (planned)
- Slack Bot Client (planned)
- WebSocket Client
- HTTP Webhook Client
- Custom Client Interface for extensions

Example Usage:
    >>> from agenticmaid_clients import TelegramClient
    >>> from agenticmaid_clients.core import Message, MessageType
    >>> 
    >>> # Initialize Telegram client
    >>> config = {
    ...     "bot_token": "your_telegram_bot_token",
    ...     "allowed_users": [],
    ...     "polling_interval": 1.0
    ... }
    >>> 
    >>> client = TelegramClient()
    >>> await client.initialize(config)
    >>> 
    >>> # Send a message
    >>> message = Message(
    ...     content="Hello from AgenticMaid!",
    ...     message_type=MessageType.TEXT,
    ...     sender_id="system"
    ... )
    >>> await client.send_message(message)
"""

__version__ = "1.0.0"
__author__ = "AgenticMaid Team"
__email__ = "contact@agenticmaid.com"
__license__ = "MIT"

# Import core messaging components
from .core.client_interface import ClientInterface
from .core.message import Message, MessageType
from .core.trigger_event import TriggerEvent

# Import available clients
try:
    from .clients.telegram_client import TelegramClient
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

try:
    from .clients.discord_client import DiscordClient
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

try:
    from .clients.webhook_client import WebhookClient
    WEBHOOK_AVAILABLE = True
except ImportError:
    WEBHOOK_AVAILABLE = False

# Define what gets imported with "from agenticmaid_clients import *"
__all__ = [
    "ClientInterface",
    "Message",
    "MessageType",
    "TriggerEvent",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Add available clients to __all__
if TELEGRAM_AVAILABLE:
    __all__.append("TelegramClient")

if DISCORD_AVAILABLE:
    __all__.append("DiscordClient")

if WEBHOOK_AVAILABLE:
    __all__.append("WebhookClient")


def get_version():
    """Get the current version of AgenticMaid Clients."""
    return __version__


def get_available_clients():
    """Get a list of available client implementations."""
    clients = []
    
    if TELEGRAM_AVAILABLE:
        clients.append("TelegramClient")
    
    if DISCORD_AVAILABLE:
        clients.append("DiscordClient")
    
    if WEBHOOK_AVAILABLE:
        clients.append("WebhookClient")
    
    return clients


def get_info():
    """Get information about AgenticMaid Clients."""
    return {
        "name": "AgenticMaid Clients",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "available_clients": get_available_clients(),
        "telegram_available": TELEGRAM_AVAILABLE,
        "discord_available": DISCORD_AVAILABLE,
        "webhook_available": WEBHOOK_AVAILABLE,
    }


# Client factory function
def create_client(client_type: str, config: dict = None):
    """
    Factory function to create client instances.
    
    Args:
        client_type: Type of client to create ('telegram', 'discord', 'webhook')
        config: Configuration dictionary for the client
        
    Returns:
        Client instance
        
    Raises:
        ValueError: If client type is not available or unknown
    """
    client_type = client_type.lower()
    
    if client_type == "telegram":
        if not TELEGRAM_AVAILABLE:
            raise ValueError("Telegram client is not available. Install with: pip install agenticmaid-clients[telegram]")
        return TelegramClient()
    
    elif client_type == "discord":
        if not DISCORD_AVAILABLE:
            raise ValueError("Discord client is not available. Install with: pip install agenticmaid-clients[discord]")
        return DiscordClient()
    
    elif client_type == "webhook":
        if not WEBHOOK_AVAILABLE:
            raise ValueError("Webhook client is not available.")
        return WebhookClient()
    
    else:
        available = ", ".join(get_available_clients())
        raise ValueError(f"Unknown client type '{client_type}'. Available clients: {available}")
