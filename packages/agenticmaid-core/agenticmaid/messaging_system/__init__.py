"""
Multi-Component Messaging System for AgenticMaid

This package provides a comprehensive messaging system with standardized client interfaces
and trigger systems for event-driven agent activation.

Components:
- clients: Standardized messaging client implementations
- triggers: Event-driven trigger system for agent activation
- core: Core data structures and utilities
- utils: Logging, monitoring, and utility functions
"""

__version__ = "1.0.0"
__author__ = "AgenticMaid Team"

from .core.message import Message, MessageType
from .core.trigger_event import TriggerEvent, TriggerEventType
from .core.exceptions import (
    MessagingSystemError,
    ClientError,
    TriggerError,
    ConfigurationError,
    ConnectionError,
    AuthenticationError
)

__all__ = [
    "Message",
    "MessageType", 
    "TriggerEvent",
    "TriggerEventType",
    "MessagingSystemError",
    "ClientError",
    "TriggerError",
    "ConfigurationError",
    "ConnectionError",
    "AuthenticationError"
]
