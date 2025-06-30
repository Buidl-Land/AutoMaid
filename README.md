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

AgenticMaid uses a layered configuration system that provides flexibility for development and production environments. The layers are loaded in the following order, with later layers overriding earlier ones:

1.  **Default settings** hard-coded in the application.
2.  **`config.json` file**: The primary configuration file for defining services, agents, and tasks.
3.  **Environment variables**: For overriding specific settings, ideal for sensitive data and CI/CD environments.

---

#### a. Main Configuration (`config.json`)

This is the central file for configuring your multi-agent system. It defines everything from AI model connections to scheduled tasks. For a complete, annotated example, please see the [`config.example.json`](./config.example.json) file.

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