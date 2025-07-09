# AgenticMaid Triggers

Event trigger implementations for the AgenticMaid framework. This package provides various trigger implementations that enable AgenticMaid agents to respond to external events and automate workflows based on real-world conditions.

## Overview

AgenticMaid Triggers is a modular package that provides event-driven trigger implementations for various scenarios. Each trigger monitors specific conditions and generates events that can be processed by AgenticMaid agents, enabling reactive and automated behavior.

## Features

- **Solana Wallet Monitoring**: Monitor Solana wallet transactions and balances
- **Time-based Triggers**: Cron-like scheduling for time-based automation
- **File System Watchers**: Monitor file and directory changes
- **HTTP Webhook Triggers**: Receive and process HTTP webhooks
- **Cryptocurrency Monitoring**: Track crypto prices and market events (planned)
- **Unified Interface**: Common interface for all trigger implementations
- **Async Support**: Full asynchronous operation support
- **Event Filtering**: Advanced filtering and processing capabilities

## Installation

### Basic Installation

```bash
pip install agenticmaid-triggers
```

### With Specific Feature Support

```bash
# Solana blockchain support
pip install agenticmaid-triggers[solana]

# File system monitoring
pip install agenticmaid-triggers[file_watcher]

# Webhook support
pip install agenticmaid-triggers[webhook]

# Cryptocurrency monitoring
pip install agenticmaid-triggers[crypto]

# All features
pip install agenticmaid-triggers[all]

# Development tools
pip install agenticmaid-triggers[dev]
```

## Quick Start

### Solana Wallet Monitoring

```python
import asyncio
from agenticmaid_triggers import SolanaWalletTrigger, TriggerEvent

async def handle_wallet_event(event: TriggerEvent):
    """Handle wallet transaction events."""
    print(f"Wallet event: {event.event_type}")
    print(f"Data: {event.data}")

async def main():
    # Initialize Solana wallet monitor
    config = {
        "wallet_addresses": [
            "wallet_address_1",
            "wallet_address_2"
        ],
        "rpc_endpoint": "https://api.mainnet-beta.solana.com",
        "check_interval": 30,  # Check every 30 seconds
        "min_sol_amount": 0.1,  # Minimum SOL amount to trigger
        "event_handler": handle_wallet_event
    }
    
    trigger = SolanaWalletTrigger()
    await trigger.initialize(config)
    
    # Start monitoring
    await trigger.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
```

### Time-based Triggers

```python
import asyncio
from agenticmaid_triggers import TimeTrigger, TriggerEvent

async def handle_time_event(event: TriggerEvent):
    """Handle scheduled time events."""
    print(f"Scheduled task triggered: {event.data['task_name']}")

async def main():
    # Initialize time-based trigger
    config = {
        "schedules": [
            {
                "name": "daily_report",
                "cron": "0 9 * * *",  # Every day at 9 AM
                "enabled": True
            },
            {
                "name": "hourly_check",
                "cron": "0 * * * *",  # Every hour
                "enabled": True
            }
        ],
        "event_handler": handle_time_event
    }
    
    trigger = TimeTrigger()
    await trigger.initialize(config)
    
    # Start monitoring
    await trigger.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
```

### File System Watcher

```python
import asyncio
from agenticmaid_triggers import FileWatcherTrigger, TriggerEvent

async def handle_file_event(event: TriggerEvent):
    """Handle file system events."""
    print(f"File event: {event.event_type}")
    print(f"Path: {event.data['path']}")

async def main():
    # Initialize file watcher
    config = {
        "watch_paths": [
            "/path/to/watch/directory",
            "/path/to/specific/file.txt"
        ],
        "recursive": True,
        "patterns": ["*.txt", "*.json"],  # Only watch specific file types
        "ignore_patterns": ["*.tmp", "*.log"],
        "event_handler": handle_file_event
    }
    
    trigger = FileWatcherTrigger()
    await trigger.initialize(config)
    
    # Start monitoring
    await trigger.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
```

### Using the Trigger Factory

```python
from agenticmaid_triggers import create_trigger

# Create a Solana wallet trigger
solana_trigger = create_trigger("solana_wallet", {
    "wallet_addresses": ["wallet_address"],
    "rpc_endpoint": "https://api.mainnet-beta.solana.com"
})

# Create a time trigger
time_trigger = create_trigger("time", {
    "schedules": [{"name": "test", "cron": "0 * * * *"}]
})
```

## Available Triggers

### SolanaWalletTrigger

Monitor Solana wallet addresses for:

- Incoming/outgoing transactions
- Balance changes
- Token transfers
- NFT transfers
- Program interactions

**Configuration:**
```json
{
  "wallet_addresses": ["address1", "address2"],
  "rpc_endpoint": "https://api.mainnet-beta.solana.com",
  "check_interval": 30,
  "min_sol_amount": 0.1,
  "track_tokens": true,
  "track_nfts": false
}
```

### TimeTrigger

Schedule events based on time with cron-like syntax:

- Cron expressions for flexible scheduling
- Multiple schedules per trigger
- Timezone support
- One-time and recurring events

**Configuration:**
```json
{
  "schedules": [
    {
      "name": "daily_task",
      "cron": "0 9 * * *",
      "timezone": "UTC",
      "enabled": true
    }
  ]
}
```

### FileWatcherTrigger

Monitor file system changes:

- File creation, modification, deletion
- Directory monitoring
- Pattern-based filtering
- Recursive watching

**Configuration:**
```json
{
  "watch_paths": ["/path/to/watch"],
  "recursive": true,
  "patterns": ["*.txt", "*.json"],
  "ignore_patterns": ["*.tmp"],
  "debounce_seconds": 1.0
}
```

### WebhookTrigger

Receive HTTP webhooks:

- Configurable endpoints
- Request validation
- Custom response handling
- Authentication support

**Configuration:**
```json
{
  "port": 8080,
  "endpoints": [
    {
      "path": "/webhook",
      "methods": ["POST"],
      "auth_token": "secret_token"
    }
  ]
}
```

## Trigger Interface

All triggers implement the `TriggerInterface` which provides a consistent API:

```python
from agenticmaid_triggers.core import TriggerInterface

class CustomTrigger(TriggerInterface):
    async def initialize(self, config: dict) -> bool:
        """Initialize the trigger with configuration."""
        pass
    
    async def start_monitoring(self) -> None:
        """Start monitoring for trigger conditions."""
        pass
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring for trigger conditions."""
        pass
    
    async def check_condition(self) -> bool:
        """Check if trigger condition is met."""
        pass
    
    async def is_active(self) -> bool:
        """Check if the trigger is actively monitoring."""
        pass
```

## Event Types

Triggers generate `TriggerEvent` objects with different event types:

```python
from agenticmaid_triggers import TriggerEvent

# Wallet transaction event
wallet_event = TriggerEvent(
    event_type="wallet_transaction",
    source="solana_wallet_trigger",
    data={
        "wallet_address": "address",
        "transaction_hash": "hash",
        "amount": 1.5,
        "token": "SOL"
    },
    timestamp=datetime.utcnow()
)

# Scheduled time event
time_event = TriggerEvent(
    event_type="scheduled_task",
    source="time_trigger",
    data={
        "task_name": "daily_report",
        "cron": "0 9 * * *"
    },
    timestamp=datetime.utcnow()
)
```

## Integration with AgenticMaid Core

To use triggers with AgenticMaid Core:

```python
from agenticmaid import AgenticMaid
from agenticmaid_triggers import SolanaWalletTrigger

# Initialize AgenticMaid
config = {
    "ai_services": {
        "default_service": {
            "provider": "Google",
            "model": "gemini-2.5-pro",
            "api_key": "your_api_key"
        }
    },
    "messaging_triggers": {
        "solana_monitor": {
            "type": "solana_wallet",
            "enabled": true,
            "config": {
                "wallet_addresses": ["wallet_address"],
                "rpc_endpoint": "https://api.mainnet-beta.solana.com",
                "check_interval": 30
            }
        }
    }
}

agent = AgenticMaid(config)
await agent.async_initialize()

# The Solana trigger will be automatically initialized and integrated
```

## Configuration Reference

### Common Configuration Options

All triggers support these common configuration options:

- `enabled`: Whether the trigger is enabled (default: true)
- `event_handler`: Custom event handler function
- `max_events_per_minute`: Rate limiting (default: 60)
- `retry_attempts`: Number of retry attempts on failure (default: 3)

### Event Filtering

Configure event filtering to process only relevant events:

```json
{
  "filters": {
    "event_types": ["wallet_transaction", "balance_change"],
    "min_amount": 1.0,
    "exclude_addresses": ["spam_address"]
  }
}
```

## Error Handling

The package provides comprehensive error handling:

```python
from agenticmaid_triggers.core.exceptions import (
    TriggerError,
    ConfigurationError,
    MonitoringError,
    ConnectionError
)

try:
    await trigger.start_monitoring()
except ConfigurationError:
    print("Configuration error - check your settings")
except ConnectionError:
    print("Connection failed - check your network")
except MonitoringError:
    print("Monitoring error - check trigger conditions")
except TriggerError as e:
    print(f"Trigger error: {e}")
```

## Development

### Running Tests

```bash
pip install agenticmaid-triggers[dev]
pytest
```

### Code Style

```bash
black agenticmaid_triggers/
flake8 agenticmaid_triggers/
mypy agenticmaid_triggers/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Links

- [Main Project](https://github.com/agenticmaid/agenticmaid)
- [Core Package](https://github.com/agenticmaid/agenticmaid-core)
- [Documentation](https://github.com/agenticmaid/agenticmaid-triggers#readme)
- [Issues](https://github.com/agenticmaid/agenticmaid-triggers/issues)
- [PyPI](https://pypi.org/project/agenticmaid-triggers/)
