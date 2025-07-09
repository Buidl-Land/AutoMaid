# AgenticMaid Core

The core framework for AgenticMaid - A Python framework for building reactive, multi-agent systems.

## Overview

AgenticMaid Core provides the fundamental building blocks for creating intelligent, reactive agent systems. It includes the core client framework, configuration management, memory protocols, and basic messaging infrastructure.

## Features

- **Core Agent Framework**: Base `AgenticMaid` class for building intelligent agents
- **Configuration Management**: Flexible configuration system with environment variable support
- **Memory Protocol**: Long-term memory capabilities for agents
- **MCP Integration**: Multi-Capability Protocol server integration
- **Async Support**: Full asynchronous operation support
- **Logging & Monitoring**: Comprehensive conversation logging and monitoring
- **CLI & API**: Command-line interface and REST API server

## Installation

```bash
# Install core package only
pip install agenticmaid-core

# Install with development tools
pip install agenticmaid-core[dev]
```

## Quick Start

### Basic Usage

```python
import asyncio
from agenticmaid import AgenticMaid

async def main():
    config = {
        "ai_services": {
            "default_service": {
                "provider": "Google",
                "model": "gemini-2.5-pro",
                "api_key": "your_api_key"
            }
        },
        "default_llm_service_name": "default_service"
    }
    
    # Create and initialize the agent
    agent = AgenticMaid(config)
    await agent.async_initialize()
    
    # Run an interaction
    response = await agent.run_mcp_interaction(
        messages=[{"role": "user", "content": "Hello, how can you help me?"}],
        llm_service_name="default_service"
    )
    
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### Command Line Usage

```bash
# Run with configuration file
agenticmaid --config-file config.json

# Start interactive CLI
agenticmaid --config-file config.json --cli

# Start API server
agenticmaid-api
```

## Configuration

### Basic Configuration

```json
{
  "ai_services": {
    "default_service": {
      "provider": "Google",
      "model": "gemini-2.5-pro",
      "api_key": "your_api_key"
    }
  },
  "default_llm_service_name": "default_service",
  "memory_protocol": {
    "enabled": true,
    "storage_type": "chromadb"
  }
}
```

### Environment Variables

```bash
# AI Service API Keys
export GOOGLE_API_KEY="your_google_api_key"
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Memory Configuration
export APP_MEMORY_PROTOCOL_ENABLED=true
```

## Core Components

### AgenticMaid Client

The main client class that orchestrates all agent operations:

```python
from agenticmaid import AgenticMaid

# Initialize with config file
agent = AgenticMaid("config.json")

# Initialize with config dictionary
agent = AgenticMaid(config_dict)
```

### Configuration Manager

Handles configuration loading and environment variable management:

```python
from agenticmaid import ConfigManager

config_manager = ConfigManager("config.json")
config = config_manager.get_config()
```

### Memory Protocol

Provides long-term memory capabilities:

```python
from agenticmaid import MemoryProtocol

memory = MemoryProtocol(config)
await memory.store_memory("user_123", "conversation", data)
memories = await memory.retrieve_memories("user_123", "conversation")
```

## API Reference

### AgenticMaid Class

#### Methods

- `async_initialize()`: Initialize the agent and all services
- `run_mcp_interaction()`: Execute an MCP interaction with an LLM
- `run_scheduled_task()`: Execute a scheduled task
- `async_run_all_enabled_scheduled_tasks()`: Run all enabled scheduled tasks

#### Properties

- `config`: Current configuration dictionary
- `memory_protocol`: Memory protocol instance (if enabled)
- `conversation_logger`: Conversation logger instance

## Extension Points

AgenticMaid Core is designed to be extended with additional modules:

- **agenticmaid-clients**: Messaging clients (Telegram, Discord, etc.)
- **agenticmaid-triggers**: Event triggers (Solana monitoring, webhooks, etc.)
- **agenticmaid-legacy**: Legacy API compatibility layer

## Development

### Running Tests

```bash
pip install agenticmaid-core[dev]
pytest
```

### Code Style

```bash
black agenticmaid/
flake8 agenticmaid/
mypy agenticmaid/
```

## License

MIT License - see LICENSE file for details.

## Links

- [Main Project](https://github.com/agenticmaid/agenticmaid)
- [Documentation](https://github.com/agenticmaid/agenticmaid-core#readme)
- [Issues](https://github.com/agenticmaid/agenticmaid-core/issues)
- [PyPI](https://pypi.org/project/agenticmaid-core/)
