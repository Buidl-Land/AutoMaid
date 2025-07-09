"""
Exception classes for the messaging system.

This module defines all custom exceptions used throughout the messaging system
to provide clear error handling and debugging capabilities.
"""

class MessagingSystemError(Exception):
    """Base exception for all messaging system errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ClientError(MessagingSystemError):
    """Exception raised for client-related errors."""
    pass


class ConnectionError(ClientError):
    """Exception raised for connection-related errors."""
    pass


class AuthenticationError(ClientError):
    """Exception raised for authentication-related errors."""
    pass


class TriggerError(MessagingSystemError):
    """Exception raised for trigger-related errors."""
    pass


class ConfigurationError(MessagingSystemError):
    """Exception raised for configuration-related errors."""
    pass


class MessageDeliveryError(ClientError):
    """Exception raised when message delivery fails."""
    pass


class EventProcessingError(TriggerError):
    """Exception raised when event processing fails."""
    pass


class AgentActivationError(TriggerError):
    """Exception raised when agent activation fails."""
    pass


class RateLimitError(ClientError):
    """Exception raised when rate limits are exceeded."""
    pass


class ValidationError(ConfigurationError):
    """Exception raised when validation fails."""
    pass
