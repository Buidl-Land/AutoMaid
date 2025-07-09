"""
Client components for the messaging system.

This module contains the client interfaces and implementations for various
messaging platforms and protocols.
"""

from .base_client import BaseClient, ClientInfo, ClientStatus
from .client_factory import ClientFactory

__all__ = [
    "BaseClient",
    "ClientInfo", 
    "ClientStatus",
    "ClientFactory"
]
