"""
Unit tests for the BaseClient abstract class and related functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from messaging_system.clients.base_client import BaseClient, ClientStatus, ClientInfo
from messaging_system.core.message import Message, MessageType
from messaging_system.core.exceptions import ClientError


class MockClient(BaseClient):
    """Mock implementation of BaseClient for testing."""
    
    def __init__(self, client_id: str, config: dict):
        super().__init__(client_id, config)
        self.connect_called = False
        self.disconnect_called = False
        self.send_message_called = False
        self.receive_message_called = False
        self.should_fail_connect = False
        self.should_fail_send = False
        self.mock_messages = []
    
    def get_client_type(self) -> str:
        return "mock"
    
    async def connect(self) -> bool:
        self.connect_called = True
        if self.should_fail_connect:
            self._set_status(ClientStatus.ERROR)
            return False
        self._set_status(ClientStatus.CONNECTED)
        return True
    
    async def disconnect(self) -> bool:
        self.disconnect_called = True
        self._set_status(ClientStatus.DISCONNECTED)
        return True
    
    async def send_message(self, message: Message) -> bool:
        self.send_message_called = True
        if self.should_fail_send:
            return False
        message.mark_delivered()
        return True
    
    async def receive_message(self, timeout=None) -> Message:
        self.receive_message_called = True
        if self.mock_messages:
            return self.mock_messages.pop(0)
        return None


class TestBaseClient:
    """Test cases for BaseClient functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        config = {
            "max_retries": 3,
            "retry_delay": 1.0,
            "timeout": 30.0,
            "max_message_size": 4096
        }
        return MockClient("test_client", config)
    
    @pytest.fixture
    def sample_message(self):
        """Create a sample message for testing."""
        return Message(
            content="Test message",
            sender="test_sender",
            recipient="test_recipient",
            message_type=MessageType.TEXT
        )
    
    def test_client_initialization(self, mock_client):
        """Test client initialization."""
        assert mock_client.client_id == "test_client"
        assert mock_client.status == ClientStatus.DISCONNECTED
        assert not mock_client.is_connected
        assert mock_client.max_retries == 3
        assert mock_client.retry_delay == 1.0
        assert mock_client.timeout == 30.0
        assert mock_client.max_message_size == 4096
    
    def test_client_info(self, mock_client):
        """Test client info property."""
        info = mock_client.client_info
        assert isinstance(info, ClientInfo)
        assert info.client_id == "test_client"
        assert info.client_type == "mock"
        assert info.status == ClientStatus.DISCONNECTED
        assert info.message_count == 0
        assert info.error_count == 0
    
    @pytest.mark.asyncio
    async def test_successful_connection(self, mock_client):
        """Test successful client connection."""
        result = await mock_client.connect()
        assert result is True
        assert mock_client.connect_called
        assert mock_client.status == ClientStatus.CONNECTED
        assert mock_client.is_connected
    
    @pytest.mark.asyncio
    async def test_failed_connection(self, mock_client):
        """Test failed client connection."""
        mock_client.should_fail_connect = True
        result = await mock_client.connect()
        assert result is False
        assert mock_client.connect_called
        assert mock_client.status == ClientStatus.ERROR
        assert not mock_client.is_connected
    
    @pytest.mark.asyncio
    async def test_disconnection(self, mock_client):
        """Test client disconnection."""
        await mock_client.connect()
        result = await mock_client.disconnect()
        assert result is True
        assert mock_client.disconnect_called
        assert mock_client.status == ClientStatus.DISCONNECTED
        assert not mock_client.is_connected
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_client, sample_message):
        """Test successful message sending."""
        await mock_client.connect()
        result = await mock_client.send_message(sample_message)
        assert result is True
        assert mock_client.send_message_called
        assert sample_message.delivered
        assert sample_message.delivery_timestamp is not None
    
    @pytest.mark.asyncio
    async def test_send_message_failure(self, mock_client, sample_message):
        """Test failed message sending."""
        await mock_client.connect()
        mock_client.should_fail_send = True
        result = await mock_client.send_message(sample_message)
        assert result is False
        assert mock_client.send_message_called
        assert not sample_message.delivered
    
    @pytest.mark.asyncio
    async def test_send_message_not_connected(self, mock_client, sample_message):
        """Test sending message when not connected."""
        with pytest.raises(ClientError):
            await mock_client.send_message(sample_message)
    
    @pytest.mark.asyncio
    async def test_receive_message(self, mock_client):
        """Test message receiving."""
        test_message = Message(
            content="Received message",
            sender="sender",
            recipient="recipient"
        )
        mock_client.mock_messages.append(test_message)
        
        await mock_client.connect()
        received = await mock_client.receive_message()
        
        assert received is not None
        assert received.content == "Received message"
        assert mock_client.receive_message_called
    
    def test_message_handler_management(self, mock_client):
        """Test adding and removing message handlers."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        # Add handlers
        mock_client.add_message_handler(handler1)
        mock_client.add_message_handler(handler2)
        assert len(mock_client._message_handlers) == 2
        
        # Remove handler
        mock_client.remove_message_handler(handler1)
        assert len(mock_client._message_handlers) == 1
        assert handler2 in mock_client._message_handlers
    
    @pytest.mark.asyncio
    async def test_message_handler_execution(self, mock_client, sample_message):
        """Test that message handlers are called."""
        handler = AsyncMock()
        mock_client.add_message_handler(handler)
        
        await mock_client._handle_incoming_message(sample_message)
        
        handler.assert_called_once_with(sample_message)
        assert mock_client._message_count == 1
    
    @pytest.mark.asyncio
    async def test_message_handler_error_handling(self, mock_client, sample_message):
        """Test error handling in message handlers."""
        handler = AsyncMock(side_effect=Exception("Handler error"))
        mock_client.add_message_handler(handler)
        
        # Should not raise exception
        await mock_client._handle_incoming_message(sample_message)
        
        assert mock_client._error_count == 1
    
    @pytest.mark.asyncio
    async def test_start_listening_not_connected(self, mock_client):
        """Test starting listener when not connected."""
        with pytest.raises(ClientError):
            await mock_client.start_listening()
    
    @pytest.mark.asyncio
    async def test_health_check_connected(self, mock_client):
        """Test health check when connected."""
        await mock_client.connect()
        result = await mock_client.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, mock_client):
        """Test health check when not connected."""
        result = await mock_client.health_check()
        assert result is False
    
    def test_metadata(self, mock_client):
        """Test client metadata."""
        metadata = mock_client.get_metadata()
        assert isinstance(metadata, dict)
        assert "max_retries" in metadata
        assert "retry_delay" in metadata
        assert "timeout" in metadata
        assert "max_message_size" in metadata
    
    def test_string_representation(self, mock_client):
        """Test string representation of client."""
        str_repr = str(mock_client)
        assert "MockClient" in str_repr
        assert "test_client" in str_repr
        assert "disconnected" in str_repr.lower()


class TestClientInfo:
    """Test cases for ClientInfo data class."""
    
    def test_client_info_creation(self):
        """Test ClientInfo creation."""
        info = ClientInfo(
            client_id="test_client",
            client_type="test_type",
            status=ClientStatus.CONNECTED,
            message_count=10,
            error_count=2
        )
        
        assert info.client_id == "test_client"
        assert info.client_type == "test_type"
        assert info.status == ClientStatus.CONNECTED
        assert info.message_count == 10
        assert info.error_count == 2
        assert info.metadata == {}
    
    def test_client_info_with_metadata(self):
        """Test ClientInfo with metadata."""
        metadata = {"custom_field": "value"}
        info = ClientInfo(
            client_id="test_client",
            client_type="test_type",
            status=ClientStatus.CONNECTED,
            metadata=metadata
        )
        
        assert info.metadata == metadata


if __name__ == "__main__":
    pytest.main([__file__])
