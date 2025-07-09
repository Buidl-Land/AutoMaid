"""
Main messaging system manager for AgenticMaid integration.

This module provides the primary interface for integrating the messaging system
with the AgenticMaid framework, managing clients, triggers, and agent communication.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime

from .clients import BaseClient, ClientFactory
from .triggers import BaseTrigger, TriggerFactory
from .core.message import Message
from .core.trigger_event import TriggerEvent
from .core.exceptions import MessagingSystemError, ConfigurationError
from .core.config_validator import ConfigValidator
from .utils.logger import setup_messaging_logger, create_component_logger
from .utils.monitoring import MessagingSystemMonitor, MetricsCollector


class MessagingSystemManager:
    """
    Main manager for the messaging system.
    
    This class coordinates clients, triggers, and agent communication,
    providing a unified interface for the AgenticMaid framework.
    """
    
    def __init__(self, config: Dict[str, Any], agentic_maid_instance=None):
        """
        Initialize the messaging system manager.
        
        Args:
            config: Configuration dictionary
            agentic_maid_instance: Reference to AgenticMaid instance for agent communication
        """
        self.config = config
        self.agentic_maid = agentic_maid_instance
        
        # Set up logging
        self.logger = setup_messaging_logger(
            level=config.get("log_level", "INFO"),
            include_timestamp=True,
            include_thread=True
        )
        
        # Component storage
        self.clients: Dict[str, BaseClient] = {}
        self.triggers: Dict[str, BaseTrigger] = {}
        
        # Monitoring
        self.metrics_collector = MetricsCollector()
        self.monitor = MessagingSystemMonitor(self.metrics_collector)
        
        # Event handlers
        self._message_handlers: List[Callable[[Message], Awaitable[None]]] = []
        self._event_handlers: List[Callable[[TriggerEvent], Awaitable[None]]] = []
        
        # System state
        self._initialized = False
        self._running = False
        
        self.logger.info("Messaging system manager initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the messaging system.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self._initialized:
            self.logger.warning("Messaging system already initialized")
            return True
        
        try:
            self.logger.info("Initializing messaging system")
            
            # Validate configuration
            await self._validate_configuration()
            
            # Initialize clients
            await self._initialize_clients()
            
            # Initialize triggers
            await self._initialize_triggers()
            
            # Set up default handlers
            self._setup_default_handlers()
            
            # Start monitoring
            self.monitor.start_monitoring()
            
            self._initialized = True
            self.logger.info("Messaging system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize messaging system: {e}")
            return False
    
    async def start(self) -> bool:
        """
        Start the messaging system.
        
        Returns:
            True if started successfully, False otherwise
        """
        if not self._initialized:
            self.logger.error("Messaging system not initialized")
            return False
        
        if self._running:
            self.logger.warning("Messaging system already running")
            return True
        
        try:
            self.logger.info("Starting messaging system")
            
            # Connect all clients
            for client_id, client in self.clients.items():
                try:
                    success = await client.connect()
                    if success:
                        await client.start_listening()
                        self.logger.info(f"Started client: {client_id}")
                    else:
                        self.logger.error(f"Failed to connect client: {client_id}")
                except Exception as e:
                    self.logger.error(f"Error starting client {client_id}: {e}")
            
            # Start all triggers
            for trigger_id, trigger in self.triggers.items():
                try:
                    success = await trigger.start_monitoring()
                    if success:
                        self.logger.info(f"Started trigger: {trigger_id}")
                    else:
                        self.logger.error(f"Failed to start trigger: {trigger_id}")
                except Exception as e:
                    self.logger.error(f"Error starting trigger {trigger_id}: {e}")
            
            self._running = True
            self.logger.info("Messaging system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start messaging system: {e}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop the messaging system.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self._running:
            self.logger.warning("Messaging system not running")
            return True
        
        try:
            self.logger.info("Stopping messaging system")
            
            # Stop all triggers
            for trigger_id, trigger in self.triggers.items():
                try:
                    await trigger.stop_monitoring()
                    self.logger.info(f"Stopped trigger: {trigger_id}")
                except Exception as e:
                    self.logger.error(f"Error stopping trigger {trigger_id}: {e}")
            
            # Stop all clients
            for client_id, client in self.clients.items():
                try:
                    await client.stop_listening()
                    await client.disconnect()
                    self.logger.info(f"Stopped client: {client_id}")
                except Exception as e:
                    self.logger.error(f"Error stopping client {client_id}: {e}")
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            self._running = False
            self.logger.info("Messaging system stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop messaging system: {e}")
            return False
    
    async def send_message(self, client_id: str, message: Message) -> bool:
        """
        Send a message through a specific client.
        
        Args:
            client_id: ID of the client to use
            message: Message to send
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if client_id not in self.clients:
            self.logger.error(f"Client not found: {client_id}")
            return False
        
        client = self.clients[client_id]
        
        try:
            success = await client.send_message(message)
            if success:
                self.metrics_collector.record_metric("messages_sent", 1.0, client_id)
            else:
                self.metrics_collector.record_metric("message_send_failures", 1.0, client_id)
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send message via {client_id}: {e}")
            self.metrics_collector.record_metric("message_send_errors", 1.0, client_id)
            return False
    
    def add_message_handler(self, handler: Callable[[Message], Awaitable[None]]) -> None:
        """Add a global message handler."""
        self._message_handlers.append(handler)
    
    def add_event_handler(self, handler: Callable[[TriggerEvent], Awaitable[None]]) -> None:
        """Add a global event handler."""
        self._event_handlers.append(handler)
    
    def get_client(self, client_id: str) -> Optional[BaseClient]:
        """Get a client by ID."""
        return self.clients.get(client_id)
    
    def get_trigger(self, trigger_id: str) -> Optional[BaseTrigger]:
        """Get a trigger by ID."""
        return self.triggers.get(trigger_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "clients": {
                client_id: {
                    "type": client.get_client_type(),
                    "status": client.status.value,
                    "connected": client.is_connected
                }
                for client_id, client in self.clients.items()
            },
            "triggers": {
                trigger_id: {
                    "type": trigger.get_trigger_type(),
                    "status": trigger.status.value,
                    "running": trigger.is_running
                }
                for trigger_id, trigger in self.triggers.items()
            },
            "health": self.monitor.get_system_health(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _validate_configuration(self) -> None:
        """Validate the messaging system configuration."""
        # Validate clients configuration
        clients_config = self.config.get("messaging_clients", {})
        for client_id, client_config in clients_config.items():
            if not client_config.get("enabled", True):
                continue
            
            client_type = client_config.get("type")
            if not client_type:
                raise ConfigurationError(f"Client {client_id} missing type")
            
            ConfigValidator.validate_client_config(client_type, client_config.get("config", {}))
        
        # Validate triggers configuration
        triggers_config = self.config.get("trigger_systems", {})
        for trigger_id, trigger_config in triggers_config.items():
            if not trigger_config.get("enabled", True):
                continue
            
            trigger_type = trigger_config.get("type")
            if not trigger_type:
                raise ConfigurationError(f"Trigger {trigger_id} missing type")
            
            ConfigValidator.validate_trigger_config(trigger_type, trigger_config.get("config", {}))
            
            if "agent_mapping" in trigger_config:
                ConfigValidator.validate_agent_mapping(trigger_config["agent_mapping"])
    
    async def _initialize_clients(self) -> None:
        """Initialize all configured clients."""
        clients_config = self.config.get("messaging_clients", {})
        
        if clients_config:
            self.clients = ClientFactory.create_clients_from_config(clients_config)
            
            # Register clients with monitor and add handlers
            for client_id, client in self.clients.items():
                self.monitor.register_component(client_id, client)
                client.add_message_handler(self._handle_incoming_message)
    
    async def _initialize_triggers(self) -> None:
        """Initialize all configured triggers."""
        triggers_config = self.config.get("trigger_systems", {})
        
        if triggers_config:
            self.triggers = TriggerFactory.create_triggers_from_config(triggers_config)
            
            # Register triggers with monitor and add handlers
            for trigger_id, trigger in self.triggers.items():
                self.monitor.register_component(trigger_id, trigger)
                trigger.add_event_handler(self._handle_trigger_event)
    
    def _setup_default_handlers(self) -> None:
        """Set up default message and event handlers."""
        # Add default agent communication handler
        if self.agentic_maid:
            self.add_event_handler(self._handle_agent_activation)
    
    async def _handle_incoming_message(self, message: Message) -> None:
        """Handle incoming messages from clients."""
        try:
            self.logger.debug(f"Handling incoming message: {message.id}")
            
            # Record metrics
            self.metrics_collector.record_metric("messages_received", 1.0)
            
            # Call all registered handlers
            for handler in self._message_handlers:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"Error in message handler: {e}")
            
            # Default behavior: forward to agent if AgenticMaid is available
            if self.agentic_maid and hasattr(self.agentic_maid, 'process_message'):
                try:
                    await self.agentic_maid.process_message(message)
                except Exception as e:
                    self.logger.error(f"Error processing message with AgenticMaid: {e}")
            
        except Exception as e:
            self.logger.error(f"Error handling incoming message: {e}")
    
    async def _handle_trigger_event(self, event: TriggerEvent) -> None:
        """Handle trigger events."""
        try:
            self.logger.debug(f"Handling trigger event: {event.id}")
            
            # Record metrics
            self.metrics_collector.record_metric("events_received", 1.0)
            
            # Call all registered handlers
            for handler in self._event_handlers:
                try:
                    await handler(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")
            
        except Exception as e:
            self.logger.error(f"Error handling trigger event: {e}")
    
    async def _handle_agent_activation(self, event: TriggerEvent) -> None:
        """Handle agent activation from trigger events."""
        try:
            if not self.agentic_maid or not event.agent_id:
                return
            
            self.logger.info(f"Activating agent {event.agent_id} for event {event.id}")
            
            # Prepare agent context
            context = {
                "trigger_event": event.to_dict(),
                "event_source": event.source,
                "event_type": event.trigger_type.value,
                "timestamp": event.timestamp.isoformat()
            }
            context.update(event.agent_context)
            
            # Activate agent through AgenticMaid
            if hasattr(self.agentic_maid, 'run_agent'):
                try:
                    result = await self.agentic_maid.run_agent(
                        agent_id=event.agent_id,
                        prompt=event.agent_prompt or f"Process trigger event: {event.trigger_type.value}",
                        context=context
                    )
                    
                    event.mark_completed()
                    self.logger.info(f"Agent {event.agent_id} completed processing event {event.id}")
                    
                except Exception as e:
                    event.mark_failed(f"Agent activation failed: {e}")
                    self.logger.error(f"Agent activation failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Error in agent activation: {e}")
            if event:
                event.mark_failed(f"Agent activation error: {e}")
