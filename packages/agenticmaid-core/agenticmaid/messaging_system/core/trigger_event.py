"""
Trigger event data structures for the messaging system.

This module defines the core TriggerEvent class and related enums used for
event-driven agent activation and communication.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import uuid


class TriggerEventType(Enum):
    """Enumeration of trigger event types."""
    WALLET_TRANSACTION = "wallet_transaction"
    BALANCE_CHANGE = "balance_change"
    PRICE_ALERT = "price_alert"
    SMART_CONTRACT_EVENT = "smart_contract_event"
    SCHEDULED_EVENT = "scheduled_event"
    FILE_CHANGE = "file_change"
    API_RESPONSE = "api_response"
    SYSTEM_EVENT = "system_event"
    CUSTOM_EVENT = "custom_event"


class TriggerEventPriority(Enum):
    """Enumeration of trigger event priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TriggerEventStatus(Enum):
    """Enumeration of trigger event processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TriggerCondition:
    """Represents a condition that must be met for trigger activation."""
    name: str
    operator: str  # e.g., ">=", "<=", "==", "!=", "contains", "matches"
    value: Any
    field_path: str  # dot notation path to the field in event data
    description: Optional[str] = None


@dataclass
class TriggerEvent:
    """
    Represents an event in the trigger system.
    
    This class encapsulates all information about a trigger event including
    event data, conditions, agent mapping, and processing status.
    """
    
    # Core event data
    trigger_type: TriggerEventType
    event_data: Dict[str, Any]
    source: str  # identifier of the trigger that generated this event
    
    # Identifiers and timestamps
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Event properties
    priority: TriggerEventPriority = TriggerEventPriority.NORMAL
    conditions_met: List[str] = field(default_factory=list)
    
    # Agent activation
    agent_id: Optional[str] = None
    agent_prompt: Optional[str] = None
    agent_context: Dict[str, Any] = field(default_factory=dict)
    
    # Processing status
    status: TriggerEventStatus = TriggerEventStatus.PENDING
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    
    # Error handling
    processing_attempts: int = 0
    last_error: Optional[str] = None
    retry_after: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def mark_processing(self) -> None:
        """Mark the event as being processed."""
        self.status = TriggerEventStatus.PROCESSING
        self.processing_started = datetime.utcnow()
        self.processing_attempts += 1
    
    def mark_completed(self) -> None:
        """Mark the event as completed."""
        self.status = TriggerEventStatus.COMPLETED
        self.processing_completed = datetime.utcnow()
    
    def mark_failed(self, error_message: str, retry_after: Optional[datetime] = None) -> None:
        """Mark the event as failed with error details."""
        self.status = TriggerEventStatus.FAILED
        self.last_error = error_message
        self.retry_after = retry_after
        self.processing_completed = datetime.utcnow()
    
    def mark_cancelled(self) -> None:
        """Mark the event as cancelled."""
        self.status = TriggerEventStatus.CANCELLED
        self.processing_completed = datetime.utcnow()
    
    def add_condition_met(self, condition_name: str) -> None:
        """Add a condition that was met for this event."""
        if condition_name not in self.conditions_met:
            self.conditions_met.append(condition_name)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the event."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if the event has expired based on age."""
        age = datetime.utcnow() - self.timestamp
        return age.total_seconds() > (max_age_hours * 3600)
    
    def can_retry(self, max_attempts: int = 3) -> bool:
        """Check if the event can be retried."""
        if self.processing_attempts >= max_attempts:
            return False
        if self.retry_after and datetime.utcnow() < self.retry_after:
            return False
        return self.status == TriggerEventStatus.FAILED
    
    def get_processing_duration(self) -> Optional[float]:
        """Get the processing duration in seconds."""
        if self.processing_started and self.processing_completed:
            return (self.processing_completed - self.processing_started).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "id": self.id,
            "trigger_type": self.trigger_type.value,
            "event_data": self.event_data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "conditions_met": self.conditions_met,
            "agent_id": self.agent_id,
            "agent_prompt": self.agent_prompt,
            "agent_context": self.agent_context,
            "status": self.status.value,
            "processing_started": self.processing_started.isoformat() if self.processing_started else None,
            "processing_completed": self.processing_completed.isoformat() if self.processing_completed else None,
            "processing_attempts": self.processing_attempts,
            "last_error": self.last_error,
            "retry_after": self.retry_after.isoformat() if self.retry_after else None,
            "metadata": self.metadata,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TriggerEvent':
        """Create event from dictionary representation."""
        # Parse timestamps
        timestamp = datetime.fromisoformat(data["timestamp"])
        processing_started = None
        if data.get("processing_started"):
            processing_started = datetime.fromisoformat(data["processing_started"])
        processing_completed = None
        if data.get("processing_completed"):
            processing_completed = datetime.fromisoformat(data["processing_completed"])
        retry_after = None
        if data.get("retry_after"):
            retry_after = datetime.fromisoformat(data["retry_after"])
        
        return cls(
            id=data["id"],
            trigger_type=TriggerEventType(data["trigger_type"]),
            event_data=data["event_data"],
            source=data["source"],
            timestamp=timestamp,
            priority=TriggerEventPriority(data["priority"]),
            conditions_met=data.get("conditions_met", []),
            agent_id=data.get("agent_id"),
            agent_prompt=data.get("agent_prompt"),
            agent_context=data.get("agent_context", {}),
            status=TriggerEventStatus(data["status"]),
            processing_started=processing_started,
            processing_completed=processing_completed,
            processing_attempts=data.get("processing_attempts", 0),
            last_error=data.get("last_error"),
            retry_after=retry_after,
            metadata=data.get("metadata", {}),
            tags=data.get("tags", [])
        )
