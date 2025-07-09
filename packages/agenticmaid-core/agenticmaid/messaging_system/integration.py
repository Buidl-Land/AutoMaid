"""
Integration module for connecting the messaging system with AgenticMaid.

This module provides utilities and extensions to integrate the messaging system
seamlessly with the existing AgenticMaid framework.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .messaging_manager import MessagingSystemManager
from .core.message import Message, MessageType
from .core.trigger_event import TriggerEvent
from .core.exceptions import MessagingSystemError


class AgenticMaidMessagingIntegration:
    """
    Integration layer between AgenticMaid and the messaging system.
    
    This class extends AgenticMaid functionality to include messaging capabilities
    and provides a bridge between agents and external communication channels.
    """
    
    def __init__(self, agentic_maid_instance):
        """
        Initialize the integration.
        
        Args:
            agentic_maid_instance: Instance of AgenticMaid to extend
        """
        self.agentic_maid = agentic_maid_instance
        self.messaging_manager: Optional[MessagingSystemManager] = None
        self.logger = logging.getLogger(__name__)
        
        # Message routing configuration
        self.default_client_id: Optional[str] = None
        self.agent_client_mapping: Dict[str, str] = {}
        
        self.logger.info("AgenticMaid messaging integration initialized")
    
    async def initialize_messaging(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the messaging system with configuration.
        
        Args:
            config: Messaging system configuration
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Extract messaging configuration
            messaging_config = {
                "messaging_clients": config.get("messaging_clients", {}),
                "trigger_systems": config.get("trigger_systems", {}),
                "log_level": config.get("log_level", "INFO")
            }
            
            # Create messaging manager
            self.messaging_manager = MessagingSystemManager(
                messaging_config,
                self.agentic_maid
            )
            
            # Set up configuration
            self.default_client_id = config.get("default_messaging_client")
            self.agent_client_mapping = config.get("agent_client_mapping", {})
            
            # Initialize and start messaging system
            success = await self.messaging_manager.initialize()
            if success:
                success = await self.messaging_manager.start()
            
            if success:
                # Add custom message handlers
                self.messaging_manager.add_message_handler(self._handle_agent_message)
                self.messaging_manager.add_event_handler(self._handle_agent_trigger)
                
                self.logger.info("Messaging system initialized and started")
            else:
                self.logger.error("Failed to initialize messaging system")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error initializing messaging system: {e}")
            return False
    
    async def shutdown_messaging(self) -> bool:
        """
        Shutdown the messaging system.
        
        Returns:
            True if shutdown was successful, False otherwise
        """
        if not self.messaging_manager:
            return True
        
        try:
            success = await self.messaging_manager.stop()
            self.messaging_manager = None
            self.logger.info("Messaging system shutdown complete")
            return success
            
        except Exception as e:
            self.logger.error(f"Error shutting down messaging system: {e}")
            return False
    
    async def send_message_to_client(
        self,
        content: str,
        recipient: str,
        client_id: Optional[str] = None,
        message_type: MessageType = MessageType.TEXT,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a message through a messaging client.
        
        Args:
            content: Message content
            recipient: Message recipient
            client_id: ID of client to use (uses default if None)
            message_type: Type of message
            metadata: Additional message metadata
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.messaging_manager:
            self.logger.error("Messaging system not initialized")
            return False
        
        # Determine which client to use
        target_client_id = client_id or self.default_client_id
        if not target_client_id:
            self.logger.error("No client specified and no default client configured")
            return False
        
        # Create message
        message = Message(
            content=content,
            sender="agentic_maid",
            recipient=recipient,
            message_type=message_type,
            metadata=metadata or {}
        )
        
        # Send message
        return await self.messaging_manager.send_message(target_client_id, message)
    
    async def send_agent_response(
        self,
        agent_id: str,
        response: str,
        original_message: Optional[Message] = None,
        client_id: Optional[str] = None
    ) -> bool:
        """
        Send an agent response back through the appropriate client.
        
        Args:
            agent_id: ID of the responding agent
            response: Agent response content
            original_message: Original message that triggered the response
            client_id: Specific client to use for response
            
        Returns:
            True if response was sent successfully, False otherwise
        """
        # Determine recipient and client
        if original_message:
            recipient = original_message.sender
            target_client_id = client_id or self._get_client_for_message(original_message)
        else:
            self.logger.error("Cannot send agent response without original message context")
            return False
        
        # Determine client for agent if not specified
        if not target_client_id:
            target_client_id = self.agent_client_mapping.get(agent_id, self.default_client_id)
        
        if not target_client_id:
            self.logger.error(f"No client configured for agent {agent_id}")
            return False
        
        # Create response message
        response_metadata = {
            "agent_id": agent_id,
            "response_to": original_message.id if original_message else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_message_to_client(
            content=response,
            recipient=recipient,
            client_id=target_client_id,
            message_type=MessageType.TEXT,
            metadata=response_metadata
        )
    
    def get_messaging_status(self) -> Dict[str, Any]:
        """Get the status of the messaging system."""
        if not self.messaging_manager:
            return {"status": "not_initialized"}
        
        return self.messaging_manager.get_system_status()
    
    def get_messaging_metrics(self) -> Dict[str, Any]:
        """Get messaging system metrics."""
        if not self.messaging_manager:
            return {}
        
        return {
            "system_health": self.messaging_manager.monitor.get_system_health(),
            "component_metrics": self.messaging_manager.metrics_collector.get_all_component_metrics()
        }
    
    async def _handle_agent_message(self, message: Message) -> None:
        """Handle incoming messages for agent processing."""
        try:
            self.logger.debug(f"Processing message for agents: {message.id}")
            
            # Determine which agent should handle this message
            agent_id = self._determine_agent_for_message(message)
            
            if agent_id and hasattr(self.agentic_maid, 'run_agent'):
                # Prepare context for agent
                context = {
                    "message": message.to_dict(),
                    "sender": message.sender,
                    "message_type": message.message_type.value,
                    "timestamp": message.timestamp.isoformat()
                }
                
                # Run agent
                try:
                    result = await self.agentic_maid.run_agent(
                        agent_id=agent_id,
                        prompt=message.content,
                        context=context
                    )
                    
                    # Send agent response back
                    if result and "output" in result:
                        await self.send_agent_response(
                            agent_id=agent_id,
                            response=result["output"],
                            original_message=message
                        )
                    
                except Exception as e:
                    self.logger.error(f"Error running agent {agent_id} for message {message.id}: {e}")
                    
                    # Send error response
                    error_response = f"Sorry, I encountered an error processing your message: {str(e)}"
                    await self.send_agent_response(
                        agent_id=agent_id,
                        response=error_response,
                        original_message=message
                    )
            
        except Exception as e:
            self.logger.error(f"Error handling agent message: {e}")
    
    async def _handle_agent_trigger(self, event: TriggerEvent) -> None:
        """Handle trigger events for agent activation."""
        try:
            self.logger.debug(f"Processing trigger event for agents: {event.id}")
            
            if not event.agent_id:
                self.logger.debug(f"No agent specified for trigger event: {event.id}")
                return
            
            if hasattr(self.agentic_maid, 'run_agent'):
                # Prepare context for agent
                context = {
                    "trigger_event": event.to_dict(),
                    "event_source": event.source,
                    "event_type": event.trigger_type.value,
                    "conditions_met": event.conditions_met
                }
                context.update(event.agent_context)
                
                # Run agent
                try:
                    result = await self.agentic_maid.run_agent(
                        agent_id=event.agent_id,
                        prompt=event.agent_prompt or f"Process trigger event: {event.trigger_type.value}",
                        context=context
                    )
                    
                    event.mark_completed()
                    self.logger.info(f"Agent {event.agent_id} completed processing trigger event {event.id}")
                    
                    # Optionally send notification about trigger processing
                    if result and "output" in result and self.default_client_id:
                        notification_content = f"Trigger processed: {result['output']}"
                        # This could be sent to a configured notification channel
                        
                except Exception as e:
                    event.mark_failed(f"Agent processing failed: {e}")
                    self.logger.error(f"Error running agent {event.agent_id} for trigger {event.id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error handling agent trigger: {e}")
    
    def _determine_agent_for_message(self, message: Message) -> Optional[str]:
        """Determine which agent should handle a message."""
        # Simple routing logic - can be extended
        
        # Check if message is a command
        if message.message_type == MessageType.COMMAND:
            command = message.content.strip().lower()
            if command.startswith('/'):
                command = command[1:]  # Remove leading slash
                
                # Map commands to agents
                command_agent_mapping = {
                    "help": "help_agent",
                    "status": "status_agent",
                    "trade": "trading_agent",
                    "balance": "balance_agent"
                }
                
                return command_agent_mapping.get(command, "general_agent")
        
        # Default to general agent for text messages
        return "general_agent"
    
    def _get_client_for_message(self, message: Message) -> Optional[str]:
        """Determine which client should be used for responding to a message."""
        # Check message metadata for client information
        if "client_id" in message.metadata:
            return message.metadata["client_id"]
        
        # Check if sender has a preferred client
        sender_client_mapping = getattr(self, 'sender_client_mapping', {})
        if message.sender in sender_client_mapping:
            return sender_client_mapping[message.sender]
        
        # Use default client
        return self.default_client_id


def extend_agentic_maid_with_messaging(agentic_maid_instance):
    """
    Extend an AgenticMaid instance with messaging capabilities.
    
    Args:
        agentic_maid_instance: AgenticMaid instance to extend
        
    Returns:
        AgenticMaidMessagingIntegration instance
    """
    integration = AgenticMaidMessagingIntegration(agentic_maid_instance)
    
    # Add messaging methods to the AgenticMaid instance
    agentic_maid_instance.messaging = integration
    agentic_maid_instance.send_message = integration.send_message_to_client
    agentic_maid_instance.get_messaging_status = integration.get_messaging_status
    agentic_maid_instance.get_messaging_metrics = integration.get_messaging_metrics
    
    return integration
