"""
Core components for the messaging system.

This module contains the fundamental data structures, exceptions, and utilities
used throughout the messaging system.
"""

from .message import Message, MessageType
from .trigger_event import TriggerEvent, TriggerEventType
from .exceptions import (
    MessagingSystemError,
    ClientError,
    TriggerError,
    ConfigurationError,
    ConnectionError,
    AuthenticationError
)
from .config_validator import ConfigValidator

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
    "AuthenticationError",
    "ConfigValidator"
]
