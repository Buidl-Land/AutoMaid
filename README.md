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

## Getting Started

### 1. Prerequisites

*   Python 3.8+

### 2. Installation

1.  Clone the repository or add `AgenticMaid` to your project.
2.  Install the required dependencies. For a full installation including the API and CLI tools, use the provided `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

    Core dependencies include:
    ```bash
    pip install python-dotenv langchain-mcp-adapters langgraph schedule langchain-core langchain-openai langchain-anthropic fastapi pydantic "uvicorn[standard]" chromadb
    ```

### 3. Configuration

Configuration is handled by the `ConfigManager` and can be defined in `config.json` and supplemented with environment variables.

#### a. Environment Variables (`.env`)

Create a `.env` file in the project root for sensitive data like API keys. These values can be referenced in `config.json` or used as fallbacks.

**Example `.env`:**
```env
# .env

# --- AI Service API Keys ---
OPENAI_API_KEY="your_openai_api_key"
ANTHROPIC_API_KEY="your_anthropic_api_key"
GOOGLE_API_KEY="your_google_api_key"

# --- Azure OpenAI ---
# AZURE_OPENAI_API_KEY="your_azure_key"
# AZURE_OPENAI_ENDPOINT="your_azure_endpoint"

# --- Memory Protocol Overrides (Optional) ---
# Override settings in config.json
# APP_MEMORY_PROTOCOL_ENABLED="true"
# APP_MEMORY_PROTOCOL_PROVIDER="openai"
# APP_MEMORY_PROTOCOL_API_KEY="your_openai_api_key_for_embeddings"
```

#### b. Main Configuration (`config.json`)

Create a `config.json` file to define AI services, MCP servers, agents, and features like the Memory Protocol. See [`config.example.json`](./config.example.json) for a comprehensive example.

**Example `config.json` Snippet:**
```json
{
  "ai_services": {
    "google_gemini_default": {
      "provider": "Google",
      "model": "gemini-2.5-pro"
    }
  },
  "mcp_servers": {
    "local_mcp_server": {
      "adapter_type": "fastapi",
      "base_url": "http://localhost:8001/mcp/v1",
      "name": "Local FastAPI MCP Server"
    }
  },
  "memory_protocol": {
    "enabled": true,
    "retrieval_k": 5,
    "database": {
      "mode": "persistent",
      "path": "./.rooroo/memory_db"
    },
    "embedding": {
      "provider": "openai",
      "model_name": "text-embedding-3-small"
    }
  },
  "default_llm_service_name": "google_gemini_default"
}
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

## Usage

### 1. Programmatic Invocation

Instantiate and run `AgenticMaid` directly within your Python application. This is the most flexible method.

See the detailed, runnable example: [`examples/direct_invocation_example.py`](./examples/direct_invocation_example.py).

**Basic Workflow:**
```python
import asyncio
from agentic_maid.client import ClientAgenticMaid

async def main():
    # Load config from a dictionary or a JSON file path
    client = ClientAgenticMaid(config_path_or_dict='config.json')
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

### 2. FastAPI Service

The project includes a FastAPI application to expose `AgenticMaid` functionality over an HTTP API.

**Run the Service:**
```bash
uvicorn agentic_maid.api:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

**Key Endpoints:**
*   `POST /client/init`: Initialize the client with a new configuration.
*   `POST /client/chat`: Process a message using a configured chat service.
*   `POST /client/run_task/{task_name}`: Trigger a specific scheduled task by name.

### 3. CLI Tool

A command-line interface is provided to execute all enabled scheduled tasks from a configuration file.

**Run the CLI:**
```bash
python -m agentic_maid.cli --config-file path/to/your/config.json
```

This command will load the config, initialize the client, and run all tasks where `"enabled": true`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.