"""
Unit tests for the BaseTrigger abstract class and related functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from messaging_system.triggers.base_trigger import BaseTrigger, TriggerStatus, TriggerInfo
from messaging_system.core.trigger_event import TriggerEvent, TriggerEventType
from messaging_system.core.exceptions import TriggerError


class MockTrigger(BaseTrigger):
    """Mock implementation of BaseTrigger for testing."""
    
    def __init__(self, trigger_id: str, config: dict):
        super().__init__(trigger_id, config)
        self.check_conditions_called = False
        self.initialize_called = False
        self.cleanup_called = False
        self.should_fail_check = False
        self.mock_events = []
    
    def get_trigger_type(self) -> str:
        return "mock"
    
    async def check_conditions(self):
        self.check_conditions_called = True
        if self.should_fail_check:
            raise TriggerError("Mock check failure")
        return self.mock_events.copy()
    
    async def _initialize(self):
        self.initialize_called = True
    
    async def _cleanup(self):
        self.cleanup_called = True


class TestBaseTrigger:
    """Test cases for BaseTrigger functionality."""
    
    @pytest.fixture
    def mock_trigger(self):
        """Create a mock trigger for testing."""
        config = {
            "check_interval": 5.0,
            "max_retries": 3,
            "retry_delay": 1.0,
            "event_timeout": 300.0,
            "agent_mapping": {
                "test_event": "test_agent"
            }
        }
        return MockTrigger("test_trigger", config)
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample trigger event for testing."""
        return TriggerEvent(
            trigger_type=TriggerEventType.WALLET_TRANSACTION,
            event_data={"test": "data"},
            source="test_trigger"
        )
    
    def test_trigger_initialization(self, mock_trigger):
        """Test trigger initialization."""
        assert mock_trigger.trigger_id == "test_trigger"
        assert mock_trigger.status == TriggerStatus.STOPPED
        assert not mock_trigger.is_running
        assert mock_trigger.check_interval == 5.0
        assert mock_trigger.max_retries == 3
        assert mock_trigger.retry_delay == 1.0
        assert mock_trigger.event_timeout == 300.0
        assert mock_trigger.agent_mapping == {"test_event": "test_agent"}
    
    def test_trigger_info(self, mock_trigger):
        """Test trigger info property."""
        info = mock_trigger.trigger_info
        assert isinstance(info, TriggerInfo)
        assert info.trigger_id == "test_trigger"
        assert info.trigger_type == "mock"
        assert info.status == TriggerStatus.STOPPED
        assert info.events_generated == 0
        assert info.events_processed == 0
        assert info.error_count == 0
    
    @pytest.mark.asyncio
    async def test_start_monitoring_success(self, mock_trigger):
        """Test successful trigger monitoring start."""
        result = await mock_trigger.start_monitoring()
        assert result is True
        assert mock_trigger.status == TriggerStatus.RUNNING
        assert mock_trigger.is_running
        assert mock_trigger.initialize_called
        assert mock_trigger._started_at is not None
        
        # Clean up
        await mock_trigger.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_start_monitoring_already_running(self, mock_trigger):
        """Test starting monitoring when already running."""
        await mock_trigger.start_monitoring()
        result = await mock_trigger.start_monitoring()
        assert result is True
        
        # Clean up
        await mock_trigger.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_stop_monitoring_success(self, mock_trigger):
        """Test successful trigger monitoring stop."""
        await mock_trigger.start_monitoring()
        result = await mock_trigger.stop_monitoring()
        assert result is True
        assert mock_trigger.status == TriggerStatus.STOPPED
        assert not mock_trigger.is_running
        assert mock_trigger.cleanup_called
        assert mock_trigger._started_at is None
    
    @pytest.mark.asyncio
    async def test_stop_monitoring_not_running(self, mock_trigger):
        """Test stopping monitoring when not running."""
        result = await mock_trigger.stop_monitoring()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_pause_and_resume_monitoring(self, mock_trigger):
        """Test pausing and resuming monitoring."""
        await mock_trigger.start_monitoring()
        
        # Pause
        result = await mock_trigger.pause_monitoring()
        assert result is True
        assert mock_trigger.status == TriggerStatus.PAUSED
        
        # Resume
        result = await mock_trigger.resume_monitoring()
        assert result is True
        assert mock_trigger.status == TriggerStatus.RUNNING
        
        # Clean up
        await mock_trigger.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_pause_when_not_running(self, mock_trigger):
        """Test pausing when not running."""
        result = await mock_trigger.pause_monitoring()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_resume_when_not_paused(self, mock_trigger):
        """Test resuming when not paused."""
        result = await mock_trigger.resume_monitoring()
        assert result is False
    
    def test_event_handler_management(self, mock_trigger):
        """Test adding and removing event handlers."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        # Add handlers
        mock_trigger.add_event_handler(handler1)
        mock_trigger.add_event_handler(handler2)
        assert len(mock_trigger._event_handlers) == 2
        
        # Remove handler
        mock_trigger.remove_event_handler(handler1)
        assert len(mock_trigger._event_handlers) == 1
        assert handler2 in mock_trigger._event_handlers
    
    @pytest.mark.asyncio
    async def test_event_handler_execution(self, mock_trigger, sample_event):
        """Test that event handlers are called."""
        handler = AsyncMock()
        mock_trigger.add_event_handler(handler)
        
        await mock_trigger._process_event(sample_event)
        
        handler.assert_called_once_with(sample_event)
        assert sample_event.status.value == "completed"
    
    @pytest.mark.asyncio
    async def test_event_handler_error_handling(self, mock_trigger, sample_event):
        """Test error handling in event handlers."""
        handler = AsyncMock(side_effect=Exception("Handler error"))
        mock_trigger.add_event_handler(handler)
        
        # Should not raise exception
        await mock_trigger._process_event(sample_event)
        
        # Event should still be marked as completed
        assert sample_event.status.value == "completed"
    
    def test_condition_management(self, mock_trigger):
        """Test adding and removing conditions."""
        from messaging_system.core.trigger_event import TriggerCondition
        
        condition1 = TriggerCondition(
            name="test_condition",
            operator=">=",
            value=100,
            field_path="amount"
        )
        
        condition2 = TriggerCondition(
            name="another_condition",
            operator="==",
            value="active",
            field_path="status"
        )
        
        # Add conditions
        mock_trigger.add_condition(condition1)
        mock_trigger.add_condition(condition2)
        assert len(mock_trigger._conditions) == 2
        
        # Remove condition
        mock_trigger.remove_condition("test_condition")
        assert len(mock_trigger._conditions) == 1
        assert mock_trigger._conditions[0].name == "another_condition"
    
    @pytest.mark.asyncio
    async def test_health_check_running(self, mock_trigger):
        """Test health check when running."""
        await mock_trigger.start_monitoring()
        result = await mock_trigger.health_check()
        assert result is True
        
        # Clean up
        await mock_trigger.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_health_check_not_running(self, mock_trigger):
        """Test health check when not running."""
        result = await mock_trigger.health_check()
        assert result is False
    
    def test_metadata(self, mock_trigger):
        """Test trigger metadata."""
        metadata = mock_trigger.get_metadata()
        assert isinstance(metadata, dict)
        assert "check_interval" in metadata
        assert "max_retries" in metadata
        assert "retry_delay" in metadata
        assert "event_timeout" in metadata
        assert "conditions_count" in metadata
        assert "agent_mapping" in metadata
    
    def test_string_representation(self, mock_trigger):
        """Test string representation of trigger."""
        str_repr = str(mock_trigger)
        assert "MockTrigger" in str_repr
        assert "test_trigger" in str_repr
        assert "stopped" in str_repr.lower()
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_with_events(self, mock_trigger):
        """Test monitoring loop generates and processes events."""
        # Add a mock event
        test_event = TriggerEvent(
            trigger_type=TriggerEventType.WALLET_TRANSACTION,
            event_data={"test": "data"},
            source="test_trigger"
        )
        mock_trigger.mock_events.append(test_event)
        
        # Start monitoring
        await mock_trigger.start_monitoring()
        
        # Wait a bit for the monitoring loop to run
        await asyncio.sleep(0.1)
        
        # Check that conditions were checked
        assert mock_trigger.check_conditions_called
        
        # Clean up
        await mock_trigger.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_error_handling(self, mock_trigger):
        """Test monitoring loop handles errors gracefully."""
        mock_trigger.should_fail_check = True
        
        # Start monitoring
        await mock_trigger.start_monitoring()
        
        # Wait a bit for the monitoring loop to run
        await asyncio.sleep(0.1)
        
        # Should still be running despite errors
        assert mock_trigger.is_running
        assert mock_trigger._error_count > 0
        
        # Clean up
        await mock_trigger.stop_monitoring()


class TestTriggerInfo:
    """Test cases for TriggerInfo data class."""
    
    def test_trigger_info_creation(self):
        """Test TriggerInfo creation."""
        info = TriggerInfo(
            trigger_id="test_trigger",
            trigger_type="test_type",
            status=TriggerStatus.RUNNING,
            events_generated=5,
            events_processed=3,
            error_count=1
        )
        
        assert info.trigger_id == "test_trigger"
        assert info.trigger_type == "test_type"
        assert info.status == TriggerStatus.RUNNING
        assert info.events_generated == 5
        assert info.events_processed == 3
        assert info.error_count == 1
        assert info.metadata == {}
    
    def test_trigger_info_with_metadata(self):
        """Test TriggerInfo with metadata."""
        metadata = {"custom_field": "value"}
        info = TriggerInfo(
            trigger_id="test_trigger",
            trigger_type="test_type",
            status=TriggerStatus.RUNNING,
            metadata=metadata
        )
        
        assert info.metadata == metadata


if __name__ == "__main__":
    pytest.main([__file__])
