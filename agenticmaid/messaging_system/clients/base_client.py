"""
Base client interface for the messaging system.

This module defines the abstract base class that all messaging clients must implement,
providing a standardized interface for communication with different platforms.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass
import asyncio
import logging
import threading
from queue import Queue, Empty

from ..core.message import Message, MessageType
from ..core.exceptions import ClientError, ConnectionError, AuthenticationError


class ClientStatus(Enum):
    """Enumeration of client connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ClientInfo:
    """Information about a client instance."""
    client_id: str
    client_type: str
    status: ClientStatus
    connected_since: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    message_count: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseClient(ABC):
    """
    Abstract base class for all messaging clients.
    
    This class defines the standard interface that all messaging clients must implement
    to provide consistent communication capabilities across different platforms.
    """
    
    def __init__(self, client_id: str, config: Dict[str, Any]):
        """
        Initialize the base client.
        
        Args:
            client_id: Unique identifier for this client instance
            config: Configuration dictionary for the client
        """
        self.client_id = client_id
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{client_id}")
        
        # Connection state
        self._status = ClientStatus.DISCONNECTED
        self._connected_since: Optional[datetime] = None
        self._last_activity: Optional[datetime] = None
        
        # Statistics
        self._message_count = 0
        self._error_count = 0
        
        # Message handling
        self._message_queue: Queue[Message] = Queue()
        self._message_handlers: List[Callable[[Message], Awaitable[None]]] = []
        
        # Threading and async support
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._background_tasks: List[asyncio.Task] = []
        
        # Configuration
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5.0)
        self.timeout = config.get("timeout", 30.0)
        self.max_message_size = config.get("max_message_size", 4096)
    
    @property
    def status(self) -> ClientStatus:
        """Get the current client status."""
        with self._lock:
            return self._status
    
    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self.status == ClientStatus.CONNECTED
    
    @property
    def client_info(self) -> ClientInfo:
        """Get information about this client."""
        with self._lock:
            return ClientInfo(
                client_id=self.client_id,
                client_type=self.get_client_type(),
                status=self._status,
                connected_since=self._connected_since,
                last_activity=self._last_activity,
                message_count=self._message_count,
                error_count=self._error_count,
                metadata=self.get_metadata()
            )
    
    @abstractmethod
    def get_client_type(self) -> str:
        """Get the type identifier for this client."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the messaging platform.
        
        Returns:
            True if connection was successful, False otherwise
            
        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the messaging platform.
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def send_message(self, message: Message) -> bool:
        """
        Send a message through the client.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was sent successfully, False otherwise
            
        Raises:
            ClientError: If sending fails
        """
        pass
    
    @abstractmethod
    async def receive_message(self, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive a message from the platform.
        
        Args:
            timeout: Maximum time to wait for a message
            
        Returns:
            Received message or None if timeout
            
        Raises:
            ClientError: If receiving fails
        """
        pass
    
    def add_message_handler(self, handler: Callable[[Message], Awaitable[None]]) -> None:
        """
        Add a message handler function.
        
        Args:
            handler: Async function to handle incoming messages
        """
        self._message_handlers.append(handler)
    
    def remove_message_handler(self, handler: Callable[[Message], Awaitable[None]]) -> None:
        """
        Remove a message handler function.
        
        Args:
            handler: Handler function to remove
        """
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)
    
    async def start_listening(self) -> None:
        """Start listening for incoming messages."""
        if not self.is_connected:
            raise ClientError("Client must be connected before starting to listen")
        
        self.logger.info(f"Starting message listener for client {self.client_id}")
        
        try:
            while not self._shutdown_event.is_set() and self.is_connected:
                try:
                    message = await self.receive_message(timeout=1.0)
                    if message:
                        await self._handle_incoming_message(message)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in message listener: {e}")
                    await asyncio.sleep(1.0)
        except Exception as e:
            self.logger.error(f"Fatal error in message listener: {e}")
            self._set_status(ClientStatus.ERROR)
    
    async def stop_listening(self) -> None:
        """Stop listening for incoming messages."""
        self.logger.info(f"Stopping message listener for client {self.client_id}")
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
    
    async def _handle_incoming_message(self, message: Message) -> None:
        """Handle an incoming message by calling all registered handlers."""
        self._update_activity()
        self._message_count += 1
        
        self.logger.debug(f"Handling incoming message: {message.id}")
        
        # Call all registered handlers
        for handler in self._message_handlers:
            try:
                await handler(message)
            except Exception as e:
                self.logger.error(f"Error in message handler: {e}")
                self._error_count += 1
    
    def _set_status(self, status: ClientStatus) -> None:
        """Set the client status."""
        with self._lock:
            old_status = self._status
            self._status = status
            
            if status == ClientStatus.CONNECTED and old_status != ClientStatus.CONNECTED:
                self._connected_since = datetime.utcnow()
            elif status != ClientStatus.CONNECTED:
                self._connected_since = None
            
            self.logger.debug(f"Client status changed: {old_status} -> {status}")
    
    def _update_activity(self) -> None:
        """Update the last activity timestamp."""
        with self._lock:
            self._last_activity = datetime.utcnow()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get client-specific metadata. Override in subclasses."""
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "max_message_size": self.max_message_size
        }
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the client.
        
        Returns:
            True if client is healthy, False otherwise
        """
        try:
            if not self.is_connected:
                return False
            
            # Subclasses can override this to perform platform-specific health checks
            return await self._perform_health_check()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def _perform_health_check(self) -> bool:
        """Perform platform-specific health check. Override in subclasses."""
        return True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.client_id}, status={self.status.value})"
    
    def __repr__(self) -> str:
        return self.__str__()
