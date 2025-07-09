# AgenticMaid

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AgenticMaid is a Python framework for building reactive, multi-agent systems. It enables dynamic interaction with one or more **Multi-Capability Protocol (MCP)** servers, allowing agents to discover and use tools at runtime. It features a robust configuration system, scheduled task execution, chat service integration, and a powerful new **Memory Protocol** for giving agents long-term memory.

## Key Features

*   **Multi-Server MCP Interaction**: Connect to and utilize tools from multiple MCP servers simultaneously.
*   **Dynamic Tool Discovery**: Agents fetch and learn to use available tools from MCP servers at runtime.
*   **Configurable Memory Protocol**: Equip agents with long-term memory using pluggable vector databases (ChromaDB) and embedding models (e.g., OpenAI).
*   **Flexible Configuration**: Configure the system via Python dictionaries, JSON files, or environment variables.
*   **Multi-Agent Dispatch**: Build hierarchical agent structures where agents can delegate tasks to one another.
*   **Scheduled & Ad-Hoc Tasks**: Run agents on a cron-like schedule or trigger them on-demand via CLI or API.
*   **Chat Service Integration**: Easily expose agents through chat interfaces with support for streaming and system prompts.
*   **Extensible AI Model Support**: Integrates with major LLM providers (OpenAI, Google, Anthropic, Azure) and local models.
*   **Concurrency & Resource Management**: Execute scheduled tasks and dispatch agents concurrently, with built-in request limiting to prevent overloading AI services.

## Modular Architecture

AgenticMaid is designed as a modular ecosystem with independently installable packages:

```
AgenticMaid Ecosystem
├── agenticmaid-core          # Core framework and agent system
├── agenticmaid-clients       # Messaging clients (Telegram, Discord, etc.)
├── agenticmaid-triggers      # Event triggers (Solana, webhooks, etc.)
└── agenticmaid-legacy        # Backward compatibility and migration tools
```

### Package Overview

| Package | Description | Installation |
|---------|-------------|--------------|
| **agenticmaid-core** | Core framework with agent system, memory protocol, and MCP integration | `pip install agenticmaid-core` |
| **agenticmaid-clients** | Messaging clients for Telegram, Discord, Slack, and webhooks | `pip install agenticmaid-clients[telegram]` |
| **agenticmaid-triggers** | Event triggers for Solana monitoring, file watching, and webhooks | `pip install agenticmaid-triggers[solana]` |
| **agenticmaid-legacy** | Backward compatibility and migration tools for older versions | `pip install agenticmaid-legacy` |

### Installation Options

```bash
# Install core framework only
pip install agenticmaid-core

# Install with specific client support
pip install agenticmaid-core agenticmaid-clients[telegram]

# Install with trigger support
pip install agenticmaid-core agenticmaid-triggers[solana]

# Install everything
pip install agenticmaid-core agenticmaid-clients[all] agenticmaid-triggers[all]

# Legacy compatibility
pip install agenticmaid-legacy
```

## Getting Started

### 1. Prerequisites

*   Python 3.8+

### 2. Installation

#### Option A: Install from PyPI (Recommended)

Install AgenticMaid as a Python package:

```bash
# Basic installation
pip install agenticmaid

# With optional dependencies for Telegram support
pip install agenticmaid[telegram]

# With optional dependencies for Solana support
pip install agenticmaid[solana]

# With all optional dependencies
pip install agenticmaid[all]

# Development installation with testing tools
pip install agenticmaid[dev]
```

#### Option B: Install from Source

1.  Clone the repository:
    ```bash
    git clone https://github.com/agenticmaid/agenticmaid.git
    cd agenticmaid
    ```

2.  Install in development mode:
    ```bash
    pip install -e .
    ```

3.  Or install with optional dependencies:
    ```bash
    pip install -e .[all]
    ```

#### Option C: Manual Installation (Legacy)

1.  Clone the repository or add `AgenticMaid` to your project.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration

AgenticMaid uses a layered configuration system that provides flexibility for development and production environments. The layers are loaded in the following order, with later layers overriding earlier ones:

1.  **Default settings** hard-coded in the application.
2.  **`config.json` file**: The primary configuration file for defining services, agents, and tasks.
3.  **Environment variables**: For overriding specific settings, ideal for sensitive data and CI/CD environments.

---

#### a. Main Configuration (`config.json`)

This is the central file for configuring your multi-agent system. It defines everything from AI model connections to scheduled tasks. Create your own `config.json` file based on the complete, annotated example at [`examples/configs/config.example.json`](./examples/configs/config.example.json).

**Key Sections:**

*   `ai_services`: Define connections to LLM providers (e.g., Google, OpenAI, Anthropic, Azure, local models).
*   `mcp_servers`: Configure connections to one or more MCP servers to provide agents with tools.
*   `memory_protocol`: Configure the long-term memory system for agents.
*   `multi_agent_dispatch`: Enable and configure agent-to-agent delegation.
*   `agents`: Pre-define agent configurations with specific models, system prompts, and descriptions.
*   `scheduled_tasks`: Define tasks to be run on a cron schedule.
*   `chat_services`: Configure endpoints for interactive chat with agents.
*   `default_llm_service_name`: Specifies the default LLM service to use if not otherwise specified.

---

#### b. Environment Variable Overrides

Any setting in `config.json` can be overridden using environment variables. This is the recommended way to handle API keys and other sensitive data, or to change settings in different environments without modifying the configuration file.

**Format:**
The environment variable name is constructed by prefixing the setting's path with `APP_`, converting to uppercase, and joining with underscores.

`APP_{SECTION_NAME}_{KEY_NAME}`

**Examples:**

To override the `retrieval_k` setting in the `memory_protocol` section:
```env
APP_MEMORY_PROTOCOL_RETRIEVAL_K=10
```

To set an API key for a specific AI service named `google_gemini_default`:
```env
APP_AI_SERVICES_GOOGLE_GEMINI_DEFAULT_API_KEY="your-google-api-key"
```

You can place these variables in a `.env` file in the project root, and they will be loaded automatically.

**Example `.env` file:**
```env
# .env

# --- AI Service API Keys ---
# These will be picked up automatically by the respective providers if api_key is not set in config.json
OPENAI_API_KEY="your_openai_api_key"
ANTHROPIC_API_KEY="your_anthropic_api_key"
GOOGLE_API_KEY="your_google_api_key"

# --- Specific Overrides ---
# This overrides the 'enabled' key in the 'memory_protocol' section of config.json
APP_MEMORY_PROTOCOL_ENABLED="true"

# This overrides the embedding model name
APP_MEMORY_PROTOCOL_EMBEDDING_MODEL_NAME="text-embedding-3-large"
```

---

#### c. Command-Line Arguments

The CLI tool (`cli.py`) provides arguments for running tasks and interacting with the system.

```bash
python -m cli --config-file path/to/config.json [OPTIONS]
```

**Available Arguments:**

| Argument | Description | Required |
| :--- | :--- | :--- |
| `--config-file <path>` | Specifies the path to the JSON configuration file. | **Yes** |
| `--run-task-now <name>` | Runs a specific scheduled task by its name immediately, ignoring its cron schedule. | No |
| `--cli` | Launches an interactive command-line chat session with the default agent. | No |

**Usage Examples:**

*   **Run all enabled scheduled tasks:**
    ```bash
    python -m cli --config-file config.json
    ```

*   **Run a single, specific task:**
    ```bash
    python -m cli --config-file config.json --run-task-now "Hourly Summary Bot"
    ```

*   **Start an interactive chat:**
    ```bash
    python -m cli --config-file config.json --cli
    ```

## Core Concepts

### 1. Memory Protocol

The Memory Protocol gives agents long-term memory, allowing them to persist and recall information across interactions. It uses a vector database to store text as embeddings and retrieve relevant memories based on semantic similarity.

#### How It Works

1.  **Ingestion**: Text is converted into a vector embedding using a configured model (e.g., OpenAI's `text-embedding-3-small`).
2.  **Storage**: The embedding and original text are stored in a vector database (e.g., ChromaDB).
3.  **Retrieval**: When an agent needs to recall information, a query is embedded, and the database returns the most similar memories.

#### Configuration

The `memory_protocol` section in `config.json` controls this feature.

```json
"memory_protocol": {
  "enabled": true,
  "retrieval_k": 5,
  "database": {
    "mode": "persistent",
    "path": "./.rooroo/memory_db",
    "host": "localhost",
    "port": 8000
  },
  "embedding": {
    "provider": "openai",
    "model_name": "text-embedding-3-small",
    "api_key": "your_openai_api_key_or_use_env",
    "base_url": "https://api.openai.com/v1"
  }
}
```

**Fields:**

*   `enabled` (boolean): Set to `true` to activate the memory system.
*   `retrieval_k` (integer): The default number of memories to retrieve per query.
*   **`database`**:
    *   `mode` (string): The ChromaDB client mode.
        *   `persistent`: Stores the database on disk at the specified `path`.
        *   `http`: Connects to a running ChromaDB server at `host` and `port`.
*   **`embedding`**:
    *   `provider` (string): The embedding model provider. Currently supports `openai`.
    *   `model_name` (string): The specific model to use for generating embeddings.
    *   `api_key` (string, optional): The API key for the embedding provider. If omitted, it falls back to the corresponding environment variable (e.g., `OPENAI_API_KEY`).
    *   `base_url` (string, optional): A custom base URL for the API, useful for local or proxy servers.

#### Overriding with Environment Variables

You can override any setting in the `memory_protocol` section using environment variables. This is useful for CI/CD pipelines or for keeping sensitive keys out of `config.json`.

The format is `APP_MEMORY_PROTOCOL_{SETTING_NAME}`.

**Examples:**
*   `APP_MEMORY_PROTOCOL_ENABLED=true`
*   `APP_MEMORY_PROTOCOL_DATABASE_MODE=http`
*   `APP_MEMORY_PROTOCOL_EMBEDDING_API_KEY="sk-..."`

### 2. Multi-Agent Dispatch

This feature allows agents to invoke and delegate tasks to other agents, creating complex, hierarchical workflows. An orchestrator agent can break down a problem and assign sub-tasks to specialized agents.

#### Configuration

Enable and configure this feature in the `multi_agent_dispatch` section of `config.json`.

```json
"multi_agent_dispatch": {
  "enabled": true,
  "default_mode": "concurrent",
  "allowed_invocations": {
    "orchestrator_agent": ["*"],
    "summary_agent": ["report_agent"],
    "report_agent": []
  }
}
```

*   `enabled`: Set to `true` to make the `dispatch` tool available to agents.
*   `default_mode`: `sync` (waits for result) or `concurrent` (fire-and-forget).
*   `allowed_invocations`: Defines which agents can call others. Use `"*"` for unrestricted access.

#### Usage

In a prompt, instruct an agent to use the `dispatch` tool.

> **Prompt:** "Please use the dispatch tool to ask the 'report_agent' to generate a detailed analysis of the latest user feedback. Run this in sync mode."

The agent will execute the tool call: `dispatch(agent_id='report_agent', prompt='Generate a detailed analysis...', mode='sync')`.

### 3. Chat Services & Dual-Prompt System

Expose agents via chat interfaces defined in the `chat_services` config section. The dual-prompt system allows for fine-grained control over agent behavior.

*   **`system_prompt`**: High-level instructions defining the agent's persona and purpose. Injected as the first `system` message.
*   **`role_prompt`**: Turn-specific instructions guiding the agent's immediate response. Injected as a `user` message before the user's actual input.

**Example `chat_services` config:**
```json
"chat_services": [
  {
    "service_id": "support_chat",
    "llm_service_name": "google_gemini_default",
    "system_prompt": "You are a helpful support assistant for AgenticMaid.",
    "role_prompt": "Answer the user's question politely and professionally."
  }
]
```

### 4. Concurrency & Resource Management

AgenticMaid includes features to manage concurrent operations and protect resources from being overloaded.

#### a. Concurrent Task Execution

You can run all enabled scheduled tasks concurrently instead of sequentially.

**Programmatic Execution:**
```python
# In your application code
await client.async_run_all_enabled_scheduled_tasks()
```

**CLI Execution:**
The CLI now runs all tasks concurrently by default.
```bash
python -m agentic_maid.cli --config-file path/to/your/config.json
```

#### b. Concurrent Agent Dispatch

When using the `dispatch` tool, you can invoke an agent in `concurrent` mode. This is a "fire-and-forget" operation that immediately returns a task ID, allowing the calling agent to continue its work without waiting for the result.

> **Prompt:** "Please use the dispatch tool to ask the 'report_agent' to generate a report, but run it in concurrent mode."

The agent will execute: `dispatch(agent_id='report_agent', prompt='Generate a report', mode='concurrent')`.

#### c. LLM Request Throttling

To prevent rate-limiting errors from AI providers, you can limit the number of concurrent requests sent to any LLM service. Add the `max_concurrent_requests` property to any service in the `ai_services` section of your `config.json`.

**Example `config.json` with Request Limiting:**
```json
"ai_services": {
  "google_gemini_default": {
    "provider": "Google",
    "model": "gemini-2.5-pro",
    "max_concurrent_requests": 5
  }
}
```
If this property is omitted, no limit is applied to that service.

## Usage

### 1. Command Line Interface

After installation, you can use AgenticMaid directly from the command line:

```bash
# Run with a configuration file
agenticmaid --config-file config.json

# Run a specific task
agenticmaid --config-file config.json --run-task-now "task_name"

# Start interactive CLI session
agenticmaid --config-file config.json --cli

# Start the API server
agenticmaid-api
```

### 2. Programmatic Usage

#### Basic Usage

```python
import asyncio
from agenticmaid import AgenticMaid

async def main():
    # Load config from a dictionary or a JSON file path
    client = AgenticMaid(config_path_or_dict='config.json')
    await client.async_initialize()

    # Run an agent interaction
    response = await client.run_mcp_interaction(
        messages=[{"role": "user", "content": "What is the weather in London?"}],
        llm_service_name="google_gemini_default",
        agent_key="weather_agent_01"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

#### Using Configuration Dictionary

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

    client = AgenticMaid(config)
    await client.async_initialize()

    response = await client.run_mcp_interaction(
        messages=[{"role": "user", "content": "Hello!"}],
        llm_service_name="default_service"
    )
    print(response)

asyncio.run(main())
```

See the detailed, runnable example: [`examples/demos/direct_invocation_example.py`](./examples/demos/direct_invocation_example.py).

### 3. FastAPI API Server

The project includes a FastAPI application to expose `AgenticMaid` functionality over an HTTP API.

**Run the API Server:**
```bash
# Using the entry point (recommended)
agenticmaid-api

# Or manually
uvicorn agenticmaid.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

**Key Endpoints:**
*   `POST /client/init`: Initialize the client with a new configuration.
*   `POST /client/chat`: Process a message using a configured chat service.
*   `POST /client/run_task/{task_name}`: Trigger a specific scheduled task by name.

### 4. Legacy CLI Usage

For backward compatibility, you can still run the CLI directly:

```bash
python -m agenticmaid.cli --config-file path/to/your/config.json
```

This command will load the config, initialize the client, and run all tasks where `"enabled": true`.

## Advanced Usage

### Client Implementation Guide

AgenticMaid's messaging system provides a flexible architecture for integrating with various communication platforms. The system supports multiple client types including Telegram, Discord, webhooks, and custom implementations.

#### Architecture Overview

The messaging system consists of:

* **Client Interface**: Abstract base for all messaging clients
* **Message Types**: Standardized message format for cross-platform compatibility
* **Event System**: Asynchronous event handling for real-time communication
* **Configuration Management**: JSON-based configuration for easy deployment

#### Creating Custom Clients

AgenticMaid supports extensible client implementations through the messaging system. You can create custom clients for different communication platforms:

```python
from messaging_system.core.client_interface import ClientInterface
from messaging_system.core.message import Message, MessageType
import asyncio
import logging

class CustomClient(ClientInterface):
    """Custom client implementation example."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.config = {}

    async def initialize(self, config: dict) -> bool:
        """Initialize the client with configuration."""
        self.config = config
        self.logger.info(f"Initializing {self.__class__.__name__} with config: {config}")

        # Validate required configuration
        required_fields = ['api_key', 'endpoint']  # Example required fields
        for field in required_fields:
            if field not in config:
                self.logger.error(f"Missing required configuration field: {field}")
                return False

        # Initialize your client-specific resources here
        # e.g., API connections, authentication, etc.

        return True

    async def send_message(self, message: Message) -> bool:
        """Send a message through this client."""
        try:
            self.logger.info(f"Sending message: {message.content}")

            # Implement your message sending logic here
            # Example: API call to your messaging platform
            # response = await self.api_client.send_message(
            #     channel=message.channel_id,
            #     content=message.content,
            #     message_type=message.message_type
            # )

            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False

    async def start_listening(self) -> None:
        """Start listening for incoming messages."""
        self.is_running = True
        self.logger.info("Starting message listener")

        while self.is_running:
            try:
                # Implement your message receiving logic here
                # Example: Poll API for new messages
                # messages = await self.api_client.get_new_messages()
                #
                # for msg in messages:
                #     message = Message(
                #         content=msg['content'],
                #         sender_id=msg['sender_id'],
                #         channel_id=msg['channel_id'],
                #         message_type=MessageType.TEXT
                #     )
                #     await self.handle_incoming_message(message)

                await asyncio.sleep(1)  # Polling interval

            except Exception as e:
                self.logger.error(f"Error in message listener: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def stop_listening(self) -> None:
        """Stop the message listener."""
        self.is_running = False
        self.logger.info("Stopped message listener")

    async def handle_incoming_message(self, message: Message) -> None:
        """Handle incoming messages and trigger AgenticMaid processing."""
        # This method should be implemented to integrate with AgenticMaid
        # Example: Forward message to AgenticMaid for processing
        pass
```

#### Client Registration

To register your custom client with the messaging system:

```python
from messaging_system.core.client_registry import ClientRegistry

# Register your custom client
ClientRegistry.register_client('custom', CustomClient)

# Now you can use it in configuration
config = {
    "messaging_clients": {
        "my_custom_client": {
            "type": "custom",
            "enabled": True,
            "config": {
                "api_key": "your_api_key",
                "endpoint": "https://api.example.com"
            }
        }
    }
}
```

#### Telegram Client Configuration

For Telegram integration, configure your client in the messaging system:

```json
{
  "messaging_clients": {
    "telegram_bot": {
      "type": "telegram",
      "enabled": true,
      "config": {
        "bot_token": "your_telegram_bot_token",
        "allowed_users": [],
        "polling_interval": 1.0,
        "timeout": 30,
        "max_retries": 3
      }
    }
  }
}
```

### Trigger System Setup

The trigger system enables event-driven automation by monitoring external conditions and automatically triggering AgenticMaid responses. This system supports various trigger types including time-based, blockchain events, file system changes, and custom conditions.

#### Trigger Architecture

The trigger system includes:

* **Trigger Interface**: Base class for all trigger implementations
* **Event Types**: Standardized event format for different trigger sources
* **Monitoring Engine**: Asynchronous monitoring with configurable intervals
* **Action Dispatcher**: Automatic routing of triggered events to appropriate agents

#### Creating Custom Triggers

The trigger system allows you to create event-driven responses. Here's how to implement custom triggers:

```python
from messaging_system.core.trigger_interface import TriggerInterface
from messaging_system.core.trigger_event import TriggerEvent
import asyncio
import logging
from datetime import datetime

class CustomTrigger(TriggerInterface):
    """Custom trigger implementation example."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_monitoring = False
        self.config = {}
        self.last_check = None

    async def initialize(self, config: dict) -> bool:
        """Initialize the trigger with configuration."""
        self.config = config
        self.logger.info(f"Initializing {self.__class__.__name__} with config: {config}")

        # Validate configuration
        required_fields = ['check_interval', 'condition_type']
        for field in required_fields:
            if field not in config:
                self.logger.error(f"Missing required configuration field: {field}")
                return False

        # Initialize trigger-specific resources
        self.check_interval = config.get('check_interval', 60)  # Default 60 seconds
        self.condition_type = config['condition_type']

        return True

    async def start_monitoring(self) -> None:
        """Start monitoring for trigger conditions."""
        self.is_monitoring = True
        self.logger.info(f"Starting monitoring with {self.check_interval}s interval")

        while self.is_monitoring:
            try:
                if await self.check_condition():
                    # Create and dispatch trigger event
                    event = TriggerEvent(
                        trigger_id=self.config.get('id', 'custom_trigger'),
                        event_type=self.condition_type,
                        timestamp=datetime.now(),
                        data=await self.get_event_data()
                    )
                    await self.dispatch_event(event)

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def check_condition(self) -> bool:
        """Check if trigger condition is met."""
        try:
            # Implement your condition checking logic here
            # Examples:
            # - Check API endpoint for new data
            # - Monitor file system changes
            # - Check database for updates
            # - Monitor external services

            current_time = datetime.now()

            # Example: Time-based trigger
            if self.condition_type == 'time_interval':
                if self.last_check is None:
                    self.last_check = current_time
                    return False

                time_diff = (current_time - self.last_check).total_seconds()
                trigger_interval = self.config.get('trigger_interval', 3600)  # 1 hour default

                if time_diff >= trigger_interval:
                    self.last_check = current_time
                    return True

            # Example: API monitoring trigger
            elif self.condition_type == 'api_change':
                # Check API for changes
                # api_response = await self.check_api()
                # return api_response.has_changes()
                pass

            return False

        except Exception as e:
            self.logger.error(f"Error checking condition: {e}")
            return False

    async def get_event_data(self) -> dict:
        """Get additional data for the triggered event."""
        return {
            'trigger_type': self.condition_type,
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }

    async def dispatch_event(self, event: TriggerEvent) -> None:
        """Dispatch the trigger event to AgenticMaid."""
        self.logger.info(f"Dispatching trigger event: {event.event_type}")
        # This should integrate with AgenticMaid's event handling system
        # Example: Send to message queue or directly invoke agent
        pass

    async def stop_monitoring(self) -> None:
        """Stop the monitoring process."""
        self.is_monitoring = False
        self.logger.info("Stopped monitoring")
```

#### Trigger Registration

Register your custom trigger with the system:

```python
from messaging_system.core.trigger_registry import TriggerRegistry

# Register your custom trigger
TriggerRegistry.register_trigger('custom', CustomTrigger)

# Use in configuration
config = {
    "messaging_triggers": {
        "my_custom_trigger": {
            "type": "custom",
            "enabled": True,
            "config": {
                "check_interval": 30,
                "condition_type": "time_interval",
                "trigger_interval": 1800
            }
        }
    }
}
```

#### Solana Wallet Monitoring

For cryptocurrency applications, you can monitor Solana wallets:

```json
{
  "messaging_triggers": {
    "solana_monitor": {
      "type": "solana_wallet",
      "enabled": true,
      "config": {
        "wallet_addresses": ["wallet_address_1", "wallet_address_2"],
        "rpc_endpoint": "https://api.mainnet-beta.solana.com",
        "check_interval": 30,
        "min_sol_amount": 0.1
      }
    }
  }
}
```

### Basic Usage Examples

#### Example 1: Simple Chat Bot

```python
import asyncio
from client import AgenticMaid

async def simple_chatbot():
    config = {
        "ai_services": {
            "default_service": {
                "provider": "Google",
                "model": "gemini-2.5-pro",
                "api_key": "your_api_key"
            }
        }
    }

    maid = AgenticMaid(config)
    await maid.async_initialize()

    response = await maid.run_mcp_interaction(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        llm_service_name="default_service"
    )
    print(response)

asyncio.run(simple_chatbot())
```

#### Example 2: Scheduled Task Execution

```python
import asyncio
from client import AgenticMaid

async def scheduled_tasks():
    config = {
        "ai_services": {
            "task_service": {
                "provider": "Google",
                "model": "gemini-2.5-flash",
                "api_key": "your_api_key"
            }
        },
        "scheduled_tasks": [
            {
                "name": "daily_summary",
                "enabled": true,
                "cron": "0 9 * * *",
                "llm_service_name": "task_service",
                "system_prompt": "You are a helpful assistant that creates daily summaries.",
                "user_prompt": "Create a summary of today's activities."
            }
        ]
    }

    maid = AgenticMaid(config)
    await maid.async_initialize()
    await maid.async_run_all_enabled_scheduled_tasks()

asyncio.run(scheduled_tasks())
```

#### Example 3: Multi-Agent Dispatch

```python
import asyncio
from client import AgenticMaid

async def multi_agent_example():
    config = {
        "ai_services": {
            "orchestrator_service": {
                "provider": "Google",
                "model": "gemini-2.5-pro",
                "api_key": "your_api_key"
            }
        },
        "multi_agent_dispatch": {
            "enabled": true,
            "default_mode": "sync",
            "allowed_invocations": {
                "orchestrator": ["analyst", "reporter"],
                "analyst": [],
                "reporter": []
            }
        },
        "agents": {
            "orchestrator": {
                "llm_service_name": "orchestrator_service",
                "system_prompt": "You coordinate tasks between analyst and reporter agents."
            },
            "analyst": {
                "llm_service_name": "orchestrator_service",
                "system_prompt": "You analyze data and provide insights."
            },
            "reporter": {
                "llm_service_name": "orchestrator_service",
                "system_prompt": "You create reports based on analysis."
            }
        }
    }

    maid = AgenticMaid(config)
    await maid.async_initialize()

    # The orchestrator can dispatch tasks to other agents
    response = await maid.run_mcp_interaction(
        messages=[{"role": "user", "content": "Analyze the latest market data and create a report."}],
        llm_service_name="orchestrator_service",
        agent_key="orchestrator"
    )
    print(response)

asyncio.run(multi_agent_example())
```

## Configuration Requirements

### Minimum Configuration

The minimum configuration requires at least one AI service:

```json
{
  "ai_services": {
    "default_service": {
      "provider": "Google",
      "model": "gemini-2.5-pro",
      "api_key": "your_api_key"
    }
  },
  "default_llm_service_name": "default_service"
}
```

### Environment Variables

Set these environment variables for secure API key management:

```bash
# AI Service API Keys
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional: Override specific configuration values
APP_MEMORY_PROTOCOL_ENABLED=true
APP_MULTI_AGENT_DISPATCH_ENABLED=true
```

### MCP Server Configuration

To use MCP servers, configure them in your config file:

```json
{
  "mcp_servers": {
    "local_server": {
      "adapter_type": "fastapi",
      "base_url": "http://localhost:8001/mcp/v1",
      "name": "Local MCP Server",
      "description": "Local development MCP server"
    }
  }
}
```

## Examples Directory

The `examples/` directory contains comprehensive examples and documentation:

* **`examples/demos/`**: Complete working examples and demo applications
* **`examples/tests/`**: Test scripts and validation tools
* **`examples/configs/`**: Example configuration files for different use cases
* **`examples/scripts/`**: Utility scripts and batch files
* **`examples/docs/`**: Detailed documentation for specific implementations
* **`examples/logs/`**: Example log files from demo runs

### Running Examples

To run the examples:

1. **Direct Invocation Example**:
   ```bash
   cd examples/demos
   python direct_invocation_example.py
   ```

2. **Telegram Bot Example**:
   ```bash
   cd examples/tests
   python test_telegram_bot.py
   ```

3. **Messaging System Example**:
   ```bash
   cd examples/demos
   python messaging_system_example.py
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.