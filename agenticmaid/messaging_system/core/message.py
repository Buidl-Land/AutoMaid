"""
Message data structures for the messaging system.

This module defines the core Message class and related enums used for
communication between clients and agents.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import uuid


class MessageType(Enum):
    """Enumeration of message types."""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    COMMAND = "command"
    SYSTEM = "system"
    ERROR = "error"
    NOTIFICATION = "notification"


class MessagePriority(Enum):
    """Enumeration of message priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class MessageAttachment:
    """Represents a file attachment in a message."""
    filename: str
    content_type: str
    size: int
    url: Optional[str] = None
    data: Optional[bytes] = None


@dataclass
class Message:
    """
    Represents a message in the messaging system.
    
    This class encapsulates all information about a message including
    content, metadata, routing information, and delivery status.
    """
    
    # Core message data
    content: str
    sender: str
    recipient: str
    message_type: MessageType = MessageType.TEXT
    
    # Identifiers and timestamps
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Message properties
    priority: MessagePriority = MessagePriority.NORMAL
    reply_to: Optional[str] = None
    thread_id: Optional[str] = None
    
    # Metadata and attachments
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: list[MessageAttachment] = field(default_factory=list)
    
    # Delivery tracking
    delivered: bool = False
    delivery_timestamp: Optional[datetime] = None
    read: bool = False
    read_timestamp: Optional[datetime] = None
    
    # Error handling
    delivery_attempts: int = 0
    last_error: Optional[str] = None
    
    def mark_delivered(self) -> None:
        """Mark the message as delivered."""
        self.delivered = True
        self.delivery_timestamp = datetime.utcnow()
    
    def mark_read(self) -> None:
        """Mark the message as read."""
        self.read = True
        self.read_timestamp = datetime.utcnow()
    
    def add_attachment(self, attachment: MessageAttachment) -> None:
        """Add an attachment to the message."""
        self.attachments.append(attachment)
    
    def set_error(self, error_message: str) -> None:
        """Set an error message for delivery failure."""
        self.last_error = error_message
        self.delivery_attempts += 1
    
    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if the message has expired based on age."""
        age = datetime.utcnow() - self.timestamp
        return age.total_seconds() > (max_age_hours * 3600)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "reply_to": self.reply_to,
            "thread_id": self.thread_id,
            "metadata": self.metadata,
            "attachments": [
                {
                    "filename": att.filename,
                    "content_type": att.content_type,
                    "size": att.size,
                    "url": att.url
                }
                for att in self.attachments
            ],
            "delivered": self.delivered,
            "delivery_timestamp": self.delivery_timestamp.isoformat() if self.delivery_timestamp else None,
            "read": self.read,
            "read_timestamp": self.read_timestamp.isoformat() if self.read_timestamp else None,
            "delivery_attempts": self.delivery_attempts,
            "last_error": self.last_error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary representation."""
        # Parse timestamps
        timestamp = datetime.fromisoformat(data["timestamp"])
        delivery_timestamp = None
        if data.get("delivery_timestamp"):
            delivery_timestamp = datetime.fromisoformat(data["delivery_timestamp"])
        read_timestamp = None
        if data.get("read_timestamp"):
            read_timestamp = datetime.fromisoformat(data["read_timestamp"])
        
        # Parse attachments
        attachments = []
        for att_data in data.get("attachments", []):
            attachments.append(MessageAttachment(
                filename=att_data["filename"],
                content_type=att_data["content_type"],
                size=att_data["size"],
                url=att_data.get("url")
            ))
        
        return cls(
            id=data["id"],
            content=data["content"],
            sender=data["sender"],
            recipient=data["recipient"],
            message_type=MessageType(data["message_type"]),
            priority=MessagePriority(data["priority"]),
            timestamp=timestamp,
            reply_to=data.get("reply_to"),
            thread_id=data.get("thread_id"),
            metadata=data.get("metadata", {}),
            attachments=attachments,
            delivered=data.get("delivered", False),
            delivery_timestamp=delivery_timestamp,
            read=data.get("read", False),
            read_timestamp=read_timestamp,
            delivery_attempts=data.get("delivery_attempts", 0),
            last_error=data.get("last_error")
        )
