"""
Trigger components for the messaging system.

This module contains the trigger interfaces and implementations for various
event detection and agent activation scenarios.
"""

from .base_trigger import BaseTrigger, TriggerInfo, TriggerStatus
from .trigger_factory import TriggerFactory

__all__ = [
    "BaseTrigger",
    "TriggerInfo",
    "TriggerStatus", 
    "TriggerFactory"
]
