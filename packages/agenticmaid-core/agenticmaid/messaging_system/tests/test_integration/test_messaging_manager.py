"""
Integration tests for the MessagingSystemManager.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json

from messaging_system.messaging_manager import MessagingSystemManager
from messaging_system.core.message import Message, MessageType
from messaging_system.core.trigger_event import TriggerEvent, TriggerEventType
from messaging_system.core.exceptions import ConfigurationError


class MockAgenticMaid:
    """Mock AgenticMaid instance for testing."""
    
    def __init__(self):
        self.run_agent_called = False
        self.process_message_called = False
        self.last_agent_id = None
        self.last_prompt = None
        self.last_context = None
        self.last_message = None
        self.should_fail = False
    
    async def run_agent(self, agent_id: str, prompt: str, context: dict = None):
        self.run_agent_called = True
        self.last_agent_id = agent_id
        self.last_prompt = prompt
        self.last_context = context
        
        if self.should_fail:
            raise Exception("Mock agent failure")
        
        return {"output": f"Agent {agent_id} processed: {prompt}"}
    
    async def process_message(self, message: Message):
        self.process_message_called = True
        self.last_message = message
        
        if self.should_fail:
            raise Exception("Mock message processing failure")


class TestMessagingSystemManager:
    """Integration tests for MessagingSystemManager."""
    
    @pytest.fixture
    def mock_agentic_maid(self):
        """Create a mock AgenticMaid instance."""
        return MockAgenticMaid()
    
    @pytest.fixture
    def basic_config(self):
        """Create a basic configuration for testing."""
        return {
            "messaging_clients": {
                "test_client": {
                    "type": "telegram",
                    "enabled": True,
                    "config": {
                        "bot_token": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
                    }
                }
            },
            "trigger_systems": {
                "test_trigger": {
                    "type": "solana_wallet",
                    "enabled": True,
                    "config": {
                        "wallet_address": "11111111111111111111111111111112",
                        "rpc_endpoint": "https://api.mainnet-beta.solana.com"
                    },
                    "agent_mapping": {
                        "transaction_detected": "test_agent"
                    }
                }
            },
            "log_level": "DEBUG"
        }
    
    @pytest.fixture
    def manager(self, basic_config, mock_agentic_maid):
        """Create a MessagingSystemManager instance."""
        return MessagingSystemManager(basic_config, mock_agentic_maid)
    
    def test_manager_initialization(self, manager, mock_agentic_maid):
        """Test manager initialization."""
        assert manager.config is not None
        assert manager.agentic_maid is mock_agentic_maid
        assert not manager._initialized
        assert not manager._running
        assert len(manager.clients) == 0
        assert len(manager.triggers) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, manager):
        """Test successful manager initialization."""
        with patch('messaging_system.clients.ClientFactory.create_clients_from_config') as mock_create_clients:
            with patch('messaging_system.triggers.TriggerFactory.create_triggers_from_config') as mock_create_triggers:
                mock_create_clients.return_value = {}
                mock_create_triggers.return_value = {}
                
                result = await manager.initialize()
                
                assert result is True
                assert manager._initialized
                assert mock_create_clients.called
                assert mock_create_triggers.called
    
    @pytest.mark.asyncio
    async def test_initialize_validation_error(self, manager):
        """Test initialization with validation error."""
        # Create invalid config
        manager.config["messaging_clients"]["test_client"]["config"] = {}  # Missing bot_token
        
        result = await manager.initialize()
        assert result is False
        assert not manager._initialized
    
    @pytest.mark.asyncio
    async def test_start_stop_cycle(self, manager):
        """Test start and stop cycle."""
        # Mock clients and triggers
        mock_client = AsyncMock()
        mock_client.connect.return_value = True
        mock_client.start_listening = AsyncMock()
        mock_client.stop_listening = AsyncMock()
        mock_client.disconnect = AsyncMock()
        
        mock_trigger = AsyncMock()
        mock_trigger.start_monitoring.return_value = True
        mock_trigger.stop_monitoring = AsyncMock()
        
        manager.clients["test_client"] = mock_client
        manager.triggers["test_trigger"] = mock_trigger
        manager._initialized = True
        
        # Test start
        result = await manager.start()
        assert result is True
        assert manager._running
        mock_client.connect.assert_called_once()
        mock_client.start_listening.assert_called_once()
        mock_trigger.start_monitoring.assert_called_once()
        
        # Test stop
        result = await manager.stop()
        assert result is True
        assert not manager._running
        mock_client.stop_listening.assert_called_once()
        mock_client.disconnect.assert_called_once()
        mock_trigger.stop_monitoring.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, manager):
        """Test successful message sending."""
        mock_client = AsyncMock()
        mock_client.send_message.return_value = True
        manager.clients["test_client"] = mock_client
        
        message = Message(
            content="Test message",
            sender="test_sender",
            recipient="test_recipient"
        )
        
        result = await manager.send_message("test_client", message)
        assert result is True
        mock_client.send_message.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_send_message_client_not_found(self, manager):
        """Test sending message with non-existent client."""
        message = Message(
            content="Test message",
            sender="test_sender",
            recipient="test_recipient"
        )
        
        result = await manager.send_message("nonexistent_client", message)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_message_failure(self, manager):
        """Test message sending failure."""
        mock_client = AsyncMock()
        mock_client.send_message.return_value = False
        manager.clients["test_client"] = mock_client
        
        message = Message(
            content="Test message",
            sender="test_sender",
            recipient="test_recipient"
        )
        
        result = await manager.send_message("test_client", message)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_handle_incoming_message(self, manager, mock_agentic_maid):
        """Test handling incoming messages."""
        message = Message(
            content="Test message",
            sender="test_sender",
            recipient="test_recipient"
        )
        
        # Add custom handler
        custom_handler = AsyncMock()
        manager.add_message_handler(custom_handler)
        
        await manager._handle_incoming_message(message)
        
        # Check that custom handler was called
        custom_handler.assert_called_once_with(message)
        
        # Check that AgenticMaid was called (if it has process_message method)
        # Note: Our mock doesn't have this method, so it won't be called
    
    @pytest.mark.asyncio
    async def test_handle_trigger_event(self, manager):
        """Test handling trigger events."""
        event = TriggerEvent(
            trigger_type=TriggerEventType.WALLET_TRANSACTION,
            event_data={"test": "data"},
            source="test_trigger"
        )
        
        # Add custom handler
        custom_handler = AsyncMock()
        manager.add_event_handler(custom_handler)
        
        await manager._handle_trigger_event(event)
        
        # Check that custom handler was called
        custom_handler.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_handle_agent_activation(self, manager, mock_agentic_maid):
        """Test agent activation from trigger events."""
        event = TriggerEvent(
            trigger_type=TriggerEventType.WALLET_TRANSACTION,
            event_data={"test": "data"},
            source="test_trigger",
            agent_id="test_agent",
            agent_prompt="Test prompt"
        )
        
        await manager._handle_agent_activation(event)
        
        # Check that agent was called
        assert mock_agentic_maid.run_agent_called
        assert mock_agentic_maid.last_agent_id == "test_agent"
        assert mock_agentic_maid.last_prompt == "Test prompt"
        assert event.status.value == "completed"
    
    @pytest.mark.asyncio
    async def test_handle_agent_activation_failure(self, manager, mock_agentic_maid):
        """Test agent activation failure handling."""
        mock_agentic_maid.should_fail = True
        
        event = TriggerEvent(
            trigger_type=TriggerEventType.WALLET_TRANSACTION,
            event_data={"test": "data"},
            source="test_trigger",
            agent_id="test_agent",
            agent_prompt="Test prompt"
        )
        
        await manager._handle_agent_activation(event)
        
        # Check that event was marked as failed
        assert event.status.value == "failed"
        assert "Agent activation failed" in event.last_error
    
    def test_get_system_status(self, manager):
        """Test getting system status."""
        # Add mock components
        mock_client = Mock()
        mock_client.get_client_type.return_value = "telegram"
        mock_client.status.value = "connected"
        mock_client.is_connected = True
        
        mock_trigger = Mock()
        mock_trigger.get_trigger_type.return_value = "solana_wallet"
        mock_trigger.status.value = "running"
        mock_trigger.is_running = True
        
        manager.clients["test_client"] = mock_client
        manager.triggers["test_trigger"] = mock_trigger
        manager._initialized = True
        manager._running = True
        
        status = manager.get_system_status()
        
        assert status["initialized"] is True
        assert status["running"] is True
        assert "test_client" in status["clients"]
        assert "test_trigger" in status["triggers"]
        assert status["clients"]["test_client"]["type"] == "telegram"
        assert status["triggers"]["test_trigger"]["type"] == "solana_wallet"
        assert "health" in status
        assert "timestamp" in status
    
    def test_get_client_and_trigger(self, manager):
        """Test getting clients and triggers."""
        mock_client = Mock()
        mock_trigger = Mock()
        
        manager.clients["test_client"] = mock_client
        manager.triggers["test_trigger"] = mock_trigger
        
        # Test getting existing components
        assert manager.get_client("test_client") is mock_client
        assert manager.get_trigger("test_trigger") is mock_trigger
        
        # Test getting non-existent components
        assert manager.get_client("nonexistent") is None
        assert manager.get_trigger("nonexistent") is None
    
    def test_handler_management(self, manager):
        """Test adding and managing handlers."""
        message_handler = AsyncMock()
        event_handler = AsyncMock()
        
        # Add handlers
        manager.add_message_handler(message_handler)
        manager.add_event_handler(event_handler)
        
        assert message_handler in manager._message_handlers
        assert event_handler in manager._event_handlers


if __name__ == "__main__":
    pytest.main([__file__])
