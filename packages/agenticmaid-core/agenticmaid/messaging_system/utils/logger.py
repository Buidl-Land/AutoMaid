"""
Logging utilities for the messaging system.

This module provides standardized logging configuration and utilities
for the messaging system components.
"""

import logging
import sys
from typing import Optional
from datetime import datetime


def setup_messaging_logger(
    name: str = "messaging_system",
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    include_thread: bool = False
) -> logging.Logger:
    """
    Set up a logger for the messaging system.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
        include_timestamp: Whether to include timestamp in logs
        include_thread: Whether to include thread information
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    
    # Create formatter
    if format_string is None:
        format_parts = []
        
        if include_timestamp:
            format_parts.append("%(asctime)s")
        
        format_parts.extend([
            "%(name)s",
            "%(levelname)s"
        ])
        
        if include_thread:
            format_parts.append("%(threadName)s")
        
        format_parts.append("%(message)s")
        
        format_string = " - ".join(format_parts)
    
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_messaging_logger(name: str) -> logging.Logger:
    """
    Get a logger for a messaging system component.
    
    Args:
        name: Component name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"messaging_system.{name}")


class MessagingLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds messaging system context to log records.
    """
    
    def __init__(self, logger: logging.Logger, extra: dict):
        super().__init__(logger, extra)
    
    def process(self, msg, kwargs):
        """Process the logging record to add extra context."""
        # Add timestamp if not present
        if "timestamp" not in self.extra:
            self.extra["timestamp"] = datetime.utcnow().isoformat()
        
        return super().process(msg, kwargs)


def create_component_logger(
    component_type: str,
    component_id: str,
    extra_context: Optional[dict] = None
) -> MessagingLoggerAdapter:
    """
    Create a logger adapter for a specific component.
    
    Args:
        component_type: Type of component (client, trigger, etc.)
        component_id: Unique identifier for the component
        extra_context: Additional context to include in logs
        
    Returns:
        Logger adapter with component context
    """
    logger_name = f"messaging_system.{component_type}.{component_id}"
    logger = logging.getLogger(logger_name)
    
    context = {
        "component_type": component_type,
        "component_id": component_id
    }
    
    if extra_context:
        context.update(extra_context)
    
    return MessagingLoggerAdapter(logger, context)
