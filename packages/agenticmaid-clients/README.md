# AgenticMaid Clients

Messaging client implementations for the AgenticMaid framework. This package provides various client implementations that enable AgenticMaid agents to communicate through different messaging platforms.

## Overview

AgenticMaid Clients is a modular package that provides messaging client implementations for various platforms. Each client implements a common interface, making it easy to switch between different messaging platforms or use multiple platforms simultaneously.

## Features

- **Telegram Bot Client**: Full-featured Telegram bot integration
- **Discord Bot Client**: Discord server and DM support (planned)
- **Slack Bot Client**: Slack workspace integration (planned)
- **Webhook Client**: HTTP webhook receiver for custom integrations
- **WebSocket Client**: Real-time WebSocket communication (planned)
- **Unified Interface**: Common interface for all client implementations
- **Async Support**: Full asynchronous operation support
- **Event-Driven**: Event-based message handling and processing

## Installation

### Basic Installation

```bash
pip install agenticmaid-clients
```

### With Specific Platform Support

```bash
# Telegram support
pip install agenticmaid-clients[telegram]

# Discord support
pip install agenticmaid-clients[discord]

# Slack support
pip install agenticmaid-clients[slack]

# Webhook support
pip install agenticmaid-clients[webhook]

# All platforms
pip install agenticmaid-clients[all]

# Development tools
pip install agenticmaid-clients[dev]
```

## Quick Start

### Telegram Client

```python
import asyncio
from agenticmaid_clients import TelegramClient, Message, MessageType

async def main():
    # Initialize Telegram client
    config = {
        "bot_token": "your_telegram_bot_token",
        "allowed_users": [],  # Empty list allows all users
        "polling_interval": 1.0,
        "timeout": 30,
        "max_retries": 3
    }
    
    client = TelegramClient()
    await client.initialize(config)
    
    # Send a message
    message = Message(
        content="Hello from AgenticMaid!",
        message_type=MessageType.TEXT,
        sender_id="system",
        recipient_id="user_chat_id"
    )
    
    success = await client.send_message(message)
    print(f"Message sent: {success}")
    
    # Start listening for messages
    await client.start_listening()

if __name__ == "__main__":
    asyncio.run(main())
```

### Using the Client Factory

```python
from agenticmaid_clients import create_client

# Create a Telegram client
telegram_client = create_client("telegram", {
    "bot_token": "your_bot_token"
})

# Create a Discord client
discord_client = create_client("discord", {
    "bot_token": "your_discord_token"
})
```

## Available Clients

### TelegramClient

Full-featured Telegram bot client with support for:

- Text messages
- Media messages (photos, documents, etc.)
- Inline keyboards
- Command handling
- User authentication
- Group chat support

**Configuration:**
```json
{
  "bot_token": "your_telegram_bot_token",
  "allowed_users": ["user_id_1", "user_id_2"],
  "polling_interval": 1.0,
  "timeout": 30,
  "max_retries": 3,
  "parse_mode": "HTML"
}
```

### DiscordClient (Planned)

Discord bot client with support for:

- Text and voice channels
- Direct messages
- Slash commands
- Embeds and reactions
- Server management

### SlackClient (Planned)

Slack bot client with support for:

- Channel messages
- Direct messages
- Slash commands
- Interactive components
- File sharing

### WebhookClient

HTTP webhook receiver for custom integrations:

- Configurable endpoints
- Request validation
- Custom response handling
- Integration with external services

## Client Interface

All clients implement the `ClientInterface` which provides a consistent API:

```python
from agenticmaid_clients.core import ClientInterface

class CustomClient(ClientInterface):
    async def initialize(self, config: dict) -> bool:
        """Initialize the client with configuration."""
        pass
    
    async def send_message(self, message: Message) -> bool:
        """Send a message through this client."""
        pass
    
    async def start_listening(self) -> None:
        """Start listening for incoming messages."""
        pass
    
    async def stop_listening(self) -> None:
        """Stop listening for incoming messages."""
        pass
    
    async def is_connected(self) -> bool:
        """Check if the client is connected."""
        pass
```

## Message Types

The package supports various message types through the `Message` class:

```python
from agenticmaid_clients import Message, MessageType

# Text message
text_msg = Message(
    content="Hello, world!",
    message_type=MessageType.TEXT,
    sender_id="user_123"
)

# Image message
image_msg = Message(
    content="path/to/image.jpg",
    message_type=MessageType.IMAGE,
    sender_id="user_123",
    metadata={"caption": "Check out this image!"}
)

# Command message
command_msg = Message(
    content="/start",
    message_type=MessageType.COMMAND,
    sender_id="user_123"
)
```

## Integration with AgenticMaid Core

To use clients with AgenticMaid Core:

```python
from agenticmaid import AgenticMaid
from agenticmaid_clients import TelegramClient

# Initialize AgenticMaid
config = {
    "ai_services": {
        "default_service": {
            "provider": "Google",
            "model": "gemini-2.5-pro",
            "api_key": "your_api_key"
        }
    },
    "messaging_clients": {
        "telegram": {
            "type": "telegram",
            "enabled": true,
            "config": {
                "bot_token": "your_telegram_bot_token"
            }
        }
    }
}

agent = AgenticMaid(config)
await agent.async_initialize()

# The Telegram client will be automatically initialized and integrated
```

## License

MIT License - see LICENSE file for details.

## Links

- [Main Project](https://github.com/agenticmaid/agenticmaid)
- [Core Package](https://github.com/agenticmaid/agenticmaid-core)
- [Documentation](https://github.com/agenticmaid/agenticmaid-clients#readme)
- [Issues](https://github.com/agenticmaid/agenticmaid-clients/issues)
- [PyPI](https://pypi.org/project/agenticmaid-clients/)
