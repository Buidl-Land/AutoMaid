"""
AgenticMaid - A Python framework for building reactive, multi-agent systems.

AgenticMaid enables dynamic interaction with Multi-Capability Protocol (MCP) servers,
allowing agents to discover and use tools at runtime. It features a robust configuration
system, scheduled task execution, chat service integration, and a powerful Memory Protocol
for giving agents long-term memory.

Key Features:
- Multi-Server MCP Interaction
- Dynamic Tool Discovery
- Configurable Memory Protocol
- Flexible Configuration
- Multi-Agent Dispatch
- Scheduled & Ad-Hoc Tasks
- Chat Service Integration
- Extensible AI Model Support
- Concurrency & Resource Management

Example Usage:
    >>> import asyncio
    >>> from agenticmaid import AgenticMaid
    >>> 
    >>> async def main():
    ...     config = {
    ...         "ai_services": {
    ...             "default_service": {
    ...                 "provider": "Google",
    ...                 "model": "gemini-2.5-pro",
    ...                 "api_key": "your_api_key"
    ...             }
    ...         }
    ...     }
    ...     
    ...     maid = AgenticMaid(config)
    ...     await maid.async_initialize()
    ...     
    ...     response = await maid.run_mcp_interaction(
    ...         messages=[{"role": "user", "content": "Hello!"}],
    ...         llm_service_name="default_service"
    ...     )
    ...     print(response)
    >>> 
    >>> asyncio.run(main())
"""

__version__ = "1.0.0"
__author__ = "AgenticMaid Team"
__email__ = "contact@agenticmaid.com"
__license__ = "MIT"

# Import main classes for easy access
from .client import AgenticMaid
from .config_manager import ConfigManager
from .conversation_logger import get_conversation_logger

# Import messaging system components
try:
    from .messaging_system.integration import extend_agentic_maid_with_messaging
    from .messaging_system.core.message import Message, MessageType
    from .messaging_system.core.trigger_event import TriggerEvent
    MESSAGING_AVAILABLE = True
except ImportError:
    MESSAGING_AVAILABLE = False

# Import memory protocol
try:
    from .memory_protocol import MemoryProtocol
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# Define what gets imported with "from agenticmaid import *"
__all__ = [
    "AgenticMaid",
    "ConfigManager", 
    "get_conversation_logger",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Add messaging components if available
if MESSAGING_AVAILABLE:
    __all__.extend([
        "extend_agentic_maid_with_messaging",
        "Message",
        "MessageType", 
        "TriggerEvent",
    ])

# Add memory components if available
if MEMORY_AVAILABLE:
    __all__.extend([
        "MemoryProtocol",
    ])


def get_version():
    """Get the current version of AgenticMaid."""
    return __version__


def get_info():
    """Get information about AgenticMaid."""
    return {
        "name": "AgenticMaid",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "messaging_available": MESSAGING_AVAILABLE,
        "memory_available": MEMORY_AVAILABLE,
    }
