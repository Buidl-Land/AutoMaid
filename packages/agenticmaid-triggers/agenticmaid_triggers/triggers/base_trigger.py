"""
Base trigger interface for the messaging system.

This module defines the abstract base class that all triggers must implement,
providing a standardized interface for event detection and agent activation.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass
import asyncio
import logging
import threading
from queue import Queue, Empty

from ..core.trigger_event import TriggerEvent, TriggerEventType, TriggerEventStatus, TriggerCondition
from ..core.exceptions import TriggerError, EventProcessingError, AgentActivationError


class TriggerStatus(Enum):
    """Enumeration of trigger status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    PAUSED = "paused"


@dataclass
class TriggerInfo:
    """Information about a trigger instance."""
    trigger_id: str
    trigger_type: str
    status: TriggerStatus
    started_at: Optional[datetime] = None
    last_check: Optional[datetime] = None
    events_generated: int = 0
    events_processed: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTrigger(ABC):
    """
    Abstract base class for all triggers.
    
    This class defines the standard interface that all triggers must implement
    to provide consistent event detection and agent activation capabilities.
    """
    
    def __init__(self, trigger_id: str, config: Dict[str, Any]):
        """
        Initialize the base trigger.
        
        Args:
            trigger_id: Unique identifier for this trigger instance
            config: Configuration dictionary for the trigger
        """
        self.trigger_id = trigger_id
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{trigger_id}")
        
        # Status and timing
        self._status = TriggerStatus.STOPPED
        self._started_at: Optional[datetime] = None
        self._last_check: Optional[datetime] = None
        
        # Statistics
        self._events_generated = 0
        self._events_processed = 0
        self._error_count = 0
        
        # Event handling
        self._event_queue: Queue[TriggerEvent] = Queue()
        self._event_handlers: List[Callable[[TriggerEvent], Awaitable[None]]] = []
        self._conditions: List[TriggerCondition] = []
        
        # Threading and async support
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._processing_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.check_interval = config.get("check_interval", 30.0)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5.0)
        self.event_timeout = config.get("event_timeout", 300.0)
        self.agent_mapping = config.get("agent_mapping", {})
        
        # Load conditions from config
        self._load_conditions()
    
    @property
    def status(self) -> TriggerStatus:
        """Get the current trigger status."""
        with self._lock:
            return self._status
    
    @property
    def is_running(self) -> bool:
        """Check if the trigger is running."""
        return self.status == TriggerStatus.RUNNING
    
    @property
    def trigger_info(self) -> TriggerInfo:
        """Get information about this trigger."""
        with self._lock:
            return TriggerInfo(
                trigger_id=self.trigger_id,
                trigger_type=self.get_trigger_type(),
                status=self._status,
                started_at=self._started_at,
                last_check=self._last_check,
                events_generated=self._events_generated,
                events_processed=self._events_processed,
                error_count=self._error_count,
                metadata=self.get_metadata()
            )
    
    @abstractmethod
    def get_trigger_type(self) -> str:
        """Get the type identifier for this trigger."""
        pass
    
    @abstractmethod
    async def check_conditions(self) -> List[TriggerEvent]:
        """
        Check for trigger conditions and generate events.
        
        Returns:
            List of trigger events generated
            
        Raises:
            TriggerError: If condition checking fails
        """
        pass
    
    async def start_monitoring(self) -> bool:
        """
        Start monitoring for trigger conditions.
        
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if self.is_running:
            self.logger.warning(f"Trigger {self.trigger_id} is already running")
            return True
        
        try:
            self._set_status(TriggerStatus.STARTING)
            self.logger.info(f"Starting trigger monitoring: {self.trigger_id}")
            
            # Initialize trigger-specific resources
            await self._initialize()
            
            # Start monitoring and processing tasks
            self._shutdown_event.clear()
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._processing_task = asyncio.create_task(self._processing_loop())
            
            self._set_status(TriggerStatus.RUNNING)
            self._started_at = datetime.utcnow()
            
            self.logger.info(f"Trigger monitoring started: {self.trigger_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start trigger monitoring: {e}")
            self._set_status(TriggerStatus.ERROR)
            return False
    
    async def stop_monitoring(self) -> bool:
        """
        Stop monitoring for trigger conditions.
        
        Returns:
            True if monitoring stopped successfully, False otherwise
        """
        if not self.is_running:
            self.logger.warning(f"Trigger {self.trigger_id} is not running")
            return True
        
        try:
            self._set_status(TriggerStatus.STOPPING)
            self.logger.info(f"Stopping trigger monitoring: {self.trigger_id}")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel and wait for tasks
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
            
            # Wait for tasks to complete
            tasks = [t for t in [self._monitoring_task, self._processing_task] if t]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Cleanup trigger-specific resources
            await self._cleanup()
            
            self._set_status(TriggerStatus.STOPPED)
            self._started_at = None
            
            self.logger.info(f"Trigger monitoring stopped: {self.trigger_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop trigger monitoring: {e}")
            self._set_status(TriggerStatus.ERROR)
            return False
    
    async def pause_monitoring(self) -> bool:
        """Pause monitoring without stopping completely."""
        if self.status != TriggerStatus.RUNNING:
            return False
        
        self._set_status(TriggerStatus.PAUSED)
        self.logger.info(f"Trigger monitoring paused: {self.trigger_id}")
        return True
    
    async def resume_monitoring(self) -> bool:
        """Resume monitoring from paused state."""
        if self.status != TriggerStatus.PAUSED:
            return False
        
        self._set_status(TriggerStatus.RUNNING)
        self.logger.info(f"Trigger monitoring resumed: {self.trigger_id}")
        return True
    
    def add_event_handler(self, handler: Callable[[TriggerEvent], Awaitable[None]]) -> None:
        """
        Add an event handler function.
        
        Args:
            handler: Async function to handle trigger events
        """
        self._event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[TriggerEvent], Awaitable[None]]) -> None:
        """
        Remove an event handler function.
        
        Args:
            handler: Handler function to remove
        """
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)
    
    def add_condition(self, condition: TriggerCondition) -> None:
        """Add a trigger condition."""
        self._conditions.append(condition)
    
    def remove_condition(self, condition_name: str) -> None:
        """Remove a trigger condition by name."""
        self._conditions = [c for c in self._conditions if c.name != condition_name]
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that checks conditions periodically."""
        self.logger.info(f"Starting monitoring loop for trigger: {self.trigger_id}")
        
        while not self._shutdown_event.is_set():
            try:
                if self.status == TriggerStatus.RUNNING:
                    # Check conditions and generate events
                    events = await self.check_conditions()
                    
                    # Queue events for processing
                    for event in events:
                        self._event_queue.put(event)
                        self._events_generated += 1
                    
                    self._last_check = datetime.utcnow()
                
                # Wait for next check interval
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._error_count += 1
                await asyncio.sleep(self.retry_delay)
    
    async def _processing_loop(self) -> None:
        """Main processing loop that handles generated events."""
        self.logger.info(f"Starting processing loop for trigger: {self.trigger_id}")
        
        while not self._shutdown_event.is_set():
            try:
                # Get event from queue with timeout
                try:
                    event = self._event_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Process the event
                await self._process_event(event)
                self._events_processed += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                self._error_count += 1
                await asyncio.sleep(self.retry_delay)
    
    async def _process_event(self, event: TriggerEvent) -> None:
        """Process a single trigger event."""
        try:
            event.mark_processing()
            self.logger.debug(f"Processing event: {event.id}")
            
            # Call all registered handlers
            for handler in self._event_handlers:
                try:
                    await handler(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")
            
            event.mark_completed()
            
        except Exception as e:
            self.logger.error(f"Failed to process event {event.id}: {e}")
            event.mark_failed(str(e))
    
    def _load_conditions(self) -> None:
        """Load trigger conditions from configuration."""
        conditions_config = self.config.get("conditions", [])
        for condition_config in conditions_config:
            condition = TriggerCondition(
                name=condition_config["name"],
                operator=condition_config["operator"],
                value=condition_config["value"],
                field_path=condition_config["field_path"],
                description=condition_config.get("description")
            )
            self._conditions.append(condition)
    
    def _set_status(self, status: TriggerStatus) -> None:
        """Set the trigger status."""
        with self._lock:
            old_status = self._status
            self._status = status
            self.logger.debug(f"Trigger status changed: {old_status} -> {status}")
    
    async def _initialize(self) -> None:
        """Initialize trigger-specific resources. Override in subclasses."""
        pass
    
    async def _cleanup(self) -> None:
        """Cleanup trigger-specific resources. Override in subclasses."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get trigger-specific metadata. Override in subclasses."""
        return {
            "check_interval": self.check_interval,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "event_timeout": self.event_timeout,
            "conditions_count": len(self._conditions),
            "agent_mapping": self.agent_mapping
        }
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the trigger.
        
        Returns:
            True if trigger is healthy, False otherwise
        """
        try:
            if not self.is_running:
                return False
            
            # Check if monitoring tasks are running
            if self._monitoring_task and self._monitoring_task.done():
                return False
            if self._processing_task and self._processing_task.done():
                return False
            
            # Subclasses can override this to perform specific health checks
            return await self._perform_health_check()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def _perform_health_check(self) -> bool:
        """Perform trigger-specific health check. Override in subclasses."""
        return True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.trigger_id}, status={self.status.value})"
    
    def __repr__(self) -> str:
        return self.__str__()
