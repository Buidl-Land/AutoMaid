# AgenticMaid Project

## Overview

AgenticMaid is a Python library designed to interact with one or more Multi-Capability Protocol (MCP) servers. It allows for dynamic fetching and utilization of tools (capabilities) provided by these servers. The client can also manage configurations for various AI/LLM services, schedule automated tasks, and handle chat service interactions, making it a versatile component for building AI-powered applications.

It leverages `langchain-mcp-adapters` for communication with MCP servers and `langgraph` for creating reactive agents that can use the fetched MCP tools.

## Features

*   **Multi-Server MCP Interaction:** Connects to and utilizes tools from multiple MCP servers.
*   **Dynamic Tool Fetching:** Retrieves available tools from MCP servers at runtime.
*   **Flexible Configuration:** Supports configuration via Python dictionaries, JSON files, and `.env` files for sensitive data.
*   **AI Service Management:** Configures and utilizes various AI/LLM services (e.g., OpenAI, Anthropic, Azure OpenAI, local models).
*   **Scheduled Tasks:** Allows defining and running tasks based on cron-like schedules.
*   **Chat Service Integration:** Provides a framework for handling interactions with defined chat services.
*   **Agent Creation:** Uses `langgraph` to create ReAct agents that can leverage MCP tools and configured LLMs.
*   **Environment Variable Support:** Loads default configurations and sensitive keys (like API keys) from an `.env` file.

## Installation

1.  **Prerequisites:**
    *   Python 3.8+

2.  **Clone the repository (if applicable) or add `AgenticMaid` to your project.**

3.  **Install Dependencies:**
    The client relies on several libraries. Ensure you have a `requirements.txt` file in your project or install them directly. Key dependencies include:
    ```bash
    pip install python-dotenv langchain-mcp-adapters langgraph schedule langchain-core langchain-openai langchain-anthropic fastapi pydantic "uvicorn[standard]"
    ```
    The command above includes core dependencies and those required for the FastAPI service and CLI tool. The file [`AgenticMaid/requirements.txt`](AgenticMaid/requirements.txt) lists dependencies primarily for the API and CLI features.

## Configuration

The `AgenticMaid` can be configured in multiple ways:

1.  **Python Dictionary:** Pass a Python dictionary directly to the `AgenticMaid` constructor.
2.  **JSON File:** Provide a path to a JSON configuration file to the constructor.
3.  **`.env` File:** For default values and sensitive information like API keys, create a `.env` file in the `AgenticMaid/` directory (i.e., alongside [`client.py`](./client.py:1)). Values from the `.env` file can be overridden by the main JSON/dictionary configuration.

### Configuration Structure

The main configuration (Python dictionary or JSON) generally includes the following sections:

*   `model` (optional): Global default settings for AI models.
*   `ai_services`: Definitions for various AI/LLM providers and models.
*   `mcp_servers`: Configuration for the MCP servers the client will connect to.
*   `scheduled_tasks`: An array of tasks to be run on a schedule.
*   `chat_services`: Definitions for chat services the client can interact with.
*   `agents` (optional): Pre-defined agent configurations.
*   `default_llm_service_name` (optional): A global default LLM service to use if not specified elsewhere.

See the [`AgenticMaid/config.example.json`](./config.example.json) file for a detailed example with comments explaining each field.

### 1. Using `.env` File

Create a file named `.env` in the `AgenticMaid` directory (e.g., `AgenticMaid/.env`). This file is used for API keys and other default settings. Values from here serve as defaults and can be overridden by the main configuration file or dictionary.

**Example `AgenticMaid/.env`:**
```env
# AgenticMaid/.env example

# Default API key if not specified per service in main config
# DEFAULT_API_KEY=your_default_api_key_here

# Default model name if not specified per service in main config
# DEFAULT_MODEL_NAME=gpt-3.5-turbo-default-from-env

# Provider-specific defaults
OPENAI_API_KEY=your_openai_api_key_from_env
OPENAI_DEFAULT_MODEL=gpt-3.5-turbo-openai-from-env

ANTHROPIC_API_KEY=your_anthropic_api_key_from_env
ANTHROPIC_DEFAULT_MODEL=claude-2-from-env

# For Azure OpenAI
# AZURE_OPENAI_API_KEY=your_azure_openai_key
# AZURE_OPENAI_ENDPOINT=your_azure_endpoint
# AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# For local OpenAI-compatible servers (e.g., LM Studio)
# LOCAL_OPENAI_BASE_URL=http://localhost:1234/v1
```

### 2. Main Configuration (JSON or Python Dict)

This configuration defines the specifics of your MCP servers, AI services, tasks, and chat interfaces.

#### a. `ai_services`

Define each AI service you intend to use. The key is a custom name for the service.
Refer to [`AgenticMaid/config.example.json`](./config.example.json) for fields like `provider`, `model`, `api_key`, `base_url`, etc. The `api_key` can be sourced from the `.env` file if not provided here.

**Example Snippet (from `config.example.json`):**
```json
{
  "ai_services": {
    "openai_gemini_pro": {
      "provider": "Google",
      "model": "gemini-2.5-pro",
      "api_key": "your_google_api_key_here_or_leave_blank_to_use_env"
    },
    "anthropic_claude_opus": {
      "provider": "Anthropic",
      "model": "claude-4-opus"
    }
  }
}
```

#### b. `mcp_servers`

Define the MCP servers the client should connect to. The `MultiServerAgenticMaid` (or relevant class) will use these configurations.
Refer to [`AgenticMaid/config.example.json`](./config.example.json) for fields like `adapter_type`, `base_url` (for FastAPI), `command_template` (for CLI), `name`, and `description`.

**Example Snippet (from `config.example.json`):**
```json
{
  "mcp_servers": {
    "server_1_local_fastapi": {
      "adapter_type": "fastapi",
      "base_url": "http://localhost:8001/mcp/v1",
      "name": "Local FastAPI MCP Server"
    }
  }
}
```

#### c. `scheduled_tasks`

Define tasks that should run on a schedule. Each task object includes:
*   `name`: A descriptive name for the task.
*   `cron_expression`: A cron-like expression (currently supports simple forms like "daily at HH:MM" or "0 * * * *" for hourly via the `schedule` library's interpretation, which might require custom parsing in `_schedule_tasks` for full cron).
*   `prompt`: The instruction/prompt for the agent.
*   `agent_id` (optional): Reference to a pre-defined agent in the `agents` section.
*   `model_config_name`: The name of the AI service (from `ai_services`) to use for this task's agent.
*   `enabled`: Boolean, `true` to enable the task, `false` to disable.

**Example Snippet (from `config.example.json`):**
```json
{
  "scheduled_tasks": [
    {
      "name": "Hourly Summary Bot",
      "cron_expression": "0 * * * *", // Placeholder, actual parsing depends on _schedule_tasks
      "prompt": "Generate a brief summary of activities from the last hour.",
      "model_config_name": "openai_gemini_pro",
      "enabled": true
    }
  ]
}
}
```
The `cron_expression` interpretation is handled by the `schedule` library. For more complex cron strings, the parsing logic in `_schedule_tasks` within [`AgenticMaid/client.py`](./client.py:237) might need adjustments.
#### d. `chat_services`

Define configurations for different chat interfaces. Each chat service object includes:
*   `service_id`: A unique identifier for the chat service.
*   `llm_service_name`: The name of the AI service (from `ai_services`) to power this chat.
*   `streaming_api_endpoint` (conceptual): A path representing where streaming responses might be served.
*   `non_streaming_api_endpoint` (conceptual): A path for non-streaming (full) responses.

**Example Snippet (from `config.example.json`):**
```json
{
  "chat_services": [
    {
      "service_id": "general_support_chat",
      "service_id": "general_support_chat",
      "llm_service_name": "openai_gemini_pro",
      "streaming_api_endpoint": "/chat/v1/streams/general_support_chat",
      "non_streaming_api_endpoint": "/chat/v1/completions/general_support_chat"
    }
  ]
}
```

##### **Dual-Prompt System**

The `chat_services` configuration now supports a dual-prompt system to provide more context and control over the agent's behavior. This is achieved through two optional fields: `system_prompt` and `role_prompt`.

*   **`system_prompt`**: This prompt is injected as the first message with the `system` role. It's used to give the AI model high-level instructions, context, or constraints that should apply to the entire conversation. For example, you can define the agent's persona, its core function, and its operational boundaries.

*   **`role_prompt`**: This prompt is injected as a `user` message immediately after the system prompt (if one is provided) and before the actual user's message. It's used to guide the AI on how it should behave or what specific role it should adopt for the upcoming turn in the conversation. This can be useful for setting a specific tone or directing the agent's focus for the immediate task.

When a chat request is processed, the final list of messages sent to the AI model will be in the following order:
1.  System Prompt (if provided)
2.  Role Prompt (if provided)
3.  User's Message(s)

**Example with Prompts in `config.json`:**
```json
{
  "chat_services": [
    {
      "service_id": "general_support_chat_gemini",
      "llm_service_name": "google_gemini_default",
      "system_prompt": "You are a helpful and friendly customer support assistant for the AgenticMaid project. Your goal is to provide clear, accurate, and concise answers.",
      "role_prompt": "Please answer the user's question based on the project's documentation and capabilities. Be polite and professional.",
      "streaming_api_endpoint": "/chat/v1/streams/general_support_chat_gemini",
      "non_streaming_api_endpoint": "/chat/v1/completions/general_support_chat_gemini"
    }
  ]
}
```

## Usage

### 1. Initialization

First, import and initialize the `AgenticMaid`. You need to call `await client.async_initialize()` after creating an instance to complete the asynchronous setup (like fetching MCP tools).

```python
import asyncio
from pkg_AgenticMaid.client import ClientAgenticMaid # Placeholder: Actual class name from client.py

async def main():
    # Option 1: Load config from JSON file
    # client = ClientAgenticMaid(config_path_or_dict='AgenticMaid/config.example.json')

    # Option 2: Load config from a Python dictionary (Direct Python Invocation)
    # This method is ideal for embedding AgenticMaid within other Python applications,
    # allowing for dynamic configuration without relying on external JSON files.
    # The .env file for API keys and defaults is still loaded if present.
    config_dict = {
        "ai_services": {
            "my_gemini_service": { # Custom name for your service
                "provider": "Google",
                "model": "gemini-2.5-pro"
                # API key can be provided here directly: "api_key": "AIza...",
                # or if omitted, it will attempt to load from .env (e.g., GOOGLE_API_KEY)
            }
        },
        "mcp_servers": {
            "example_mcp_server": { # Custom name for your MCP server connection
                "adapter_type": "fastapi", # Or other supported adapter types
                "base_url": "http://localhost:8001/mcp/v1", # URL of the target MCP server
                "name": "My Example MCP Server"
            }
        },
        "default_llm_service_name": "my_gemini_service", # Default LLM for agents if not specified
        # Other sections like "scheduled_tasks", "chat_services", "agents" can be added as needed.
        # For a comprehensive, runnable example of direct dictionary invocation,
        # please refer to the script:
        # [`AgenticMaid/examples/direct_invocation_example.py`](./examples/direct_invocation_example.py)
    }
    client = ClientAgenticMaid(config_path_or_dict=config_dict)

    # Perform asynchronous initialization
    await client.async_initialize()

    if client.config and client.mcp_client:
        print("ClientAgenticMaid initialized successfully.")
        print(f"Fetched {len(client.mcp_tools)} MCP tools: {[tool.name for tool in client.mcp_tools]}")
    else:
        print("ClientAgenticMaid initialization failed or no MCP tools found.")
        print("AgenticMaidClient initialization failed or no MCP tools found.")
        return

    # ... use the client ...

if __name__ == "__main__":
    asyncio.run(main())
```

### 1.1. Detailed Example of Direct Dictionary Invocation

For a runnable script demonstrating how to instantiate and use `ClientAgenticMaid` with a direct dictionary configuration, including basic operations like chat, please refer to the example file:

*   [`AgenticMaid/examples/direct_invocation_example.py`](./examples/direct_invocation_example.py)

This example showcases how to set up the configuration dictionary and perform common client actions.

### 2. Running an MCP Interaction (Agent Invocation)

Use the `run_mcp_interaction` method to interact with an agent. The agent will be created (or retrieved if it already exists) using the specified LLM service and all fetched MCP tools.
```python
```python
# Assuming 'client' is an initialized ClientAgenticMaid instance from the example above

    # Example: Run an interaction
    messages_for_agent = [{"role": "user", "content": "What is the weather in London using available tools?"}]
    llm_service_to_use = "my_gemini_service" # Must be a key from your ai_services config
    agent_identifier = "weather_agent_01" # A custom key for this agent instance

    response = await client.run_mcp_interaction(
        messages=messages_for_agent,
        llm_service_name=llm_service_to_use,
        agent_key=agent_identifier
    )

    if response and "error" not in response:
        print(f"Agent Response: {response}")
    else:
        print(f"Agent Interaction Error: {response.get('error') if response else 'Unknown error'}")
```

### 3. Running Scheduled Tasks

To run scheduled tasks, first ensure they are defined in your configuration. Then, start the scheduler. The scheduler runs in a background thread.

```python
# Assuming 'client' is an initialized ClientAgenticMaid instance

    # To start the scheduler (it runs in a background thread):
    if client.scheduler.jobs: # Check if there are any jobs scheduled
        print("Starting scheduler...")
        client.start_scheduler()
        # The scheduler will now run tasks in the background.
        # Keep the main thread alive if you want tasks to continue running.
        # For example, in a long-running application:
        # try:
        #     while True:
        #         await asyncio.sleep(1)
        # except KeyboardInterrupt:
        #     print("Application shutting down.")
        #     client.stop_scheduler() # Conceptual stop
    else:
        print("No tasks scheduled.")
```
**Note:** The `start_scheduler` method runs an infinite loop in a daemon thread. Ensure your main application manages its lifecycle appropriately. The `stop_scheduler` method is currently a placeholder; a more robust stop mechanism (e.g., using `threading.Event`) might be needed for graceful shutdown in complex applications.

### 4. Interacting with Chat Services

To handle messages for a defined chat service, use the `handle_chat_message` method.

```python
# Assuming 'client' is an initialized ClientAgenticMaid instance

    # Example: Interact with a chat service
    chat_service_id_to_use = "general_support_chat" # Must be a service_id from your chat_services config
    chat_messages = [{"role": "user", "content": "Hello, I need help with my account."}]

    chat_response = await client.handle_chat_message(
        service_id=chat_service_id_to_use,
        messages=chat_messages,
        stream=False # Set to True for streaming (currently placeholder)
    )

    if chat_response and "error" not in chat_response:
        print(f"Chat Service Response: {chat_response}")
    else:
        print(f"Chat Service Error: {chat_response.get('error') if chat_response else 'Unknown error'}")

```

<![CDATA[
### 5. Running the FastAPI Service

The `AgenticMaid` project includes a FastAPI application that exposes its functionalities over an HTTP API. This allows `AgenticMaid` to be integrated into other systems or accessed remotely.

**To run the FastAPI service:**

Ensure you have `uvicorn` and `fastapi` installed (they are included in the `pip install` command in the [Installation](#installation) section if you included `AgenticMaid/requirements.txt` dependencies).

From the root directory of the `Agent` project (where `AgenticMaid` is a subdirectory), run:

```bash
python -m uvicorn AgenticMaid.api:app --reload
```

Or, if your `PYTHONPATH` is set up correctly to find the `AgenticMaid` module:

```bash
uvicorn AgenticMaid.api:app --reload
```

The API will typically be available at `http://127.0.0.1:8000`.

#### API Endpoints

The following are the main endpoints provided by the FastAPI service:

*   **`POST /client/init`**:
    *   **Purpose**: Initializes or reconfigures the global `ClientAgenticMaid` instance within the API service.
    *   **Request Body**: A JSON object containing a `config` key. The value can be either a path (string) to a JSON configuration file accessible by the server, or a full configuration dictionary.
        ```json
        {
          "config": "./AgenticMaid/config.example.json"
        }
        ```
        or
        ```json
        {
          "config": {
            "ai_services": { "...": "..." },
            "mcp_servers": { "...": "..." }
            // ... other config sections
          }
        }
        ```
    *   **Response**: A JSON object indicating success or failure.
        ```json
        {
          "status": "success",
          "message": "ClientAgenticMaid (re)configured successfully."
        }
        ```

*   **`POST /client/chat`**:
    *   **Purpose**: Processes a chat message using a configured chat service within `ClientAgenticMaid`.
    *   **Request Body**:
        ```json
        {
          "service_id": "your_chat_service_id_from_config",
          "messages": [
            { "role": "user", "content": "Hello, how can you help me?" }
          ]
        }
        ```
    *   **Response**: The response from the chat agent, which can vary in structure. Typically includes the agent's reply.
        ```json
        {
          "data": {
            "messages": [
              { "role": "user", "content": "Hello, how can you help me?" },
              { "role": "assistant", "content": "I am an AI assistant..." }
            ]
          },
          "error": null
        }
        ```

*   **`POST /client/run_task/{task_name}`**:
    *   **Purpose**: Triggers a specific scheduled task by its name, as defined in the `ClientAgenticMaid` configuration.
    *   **Path Parameter**: `task_name` (string) - The name of the task.
    *   **Response**: A JSON object indicating the status of the task execution.
        ```json
        {
          "status": "success",
          "message": "Task 'YourTaskName' executed.",
          "task_name": "YourTaskName",
          "details": { /* ... task execution result ... */ }
        }
        ```

*   **`POST /client/run_all_scheduled_tasks`**:
    *   **Purpose**: Triggers all *enabled* scheduled tasks defined in the `ClientAgenticMaid` configuration.
    *   **Response**: A JSON object containing a list of results for each task attempted.
        ```json
        {
          "results": [
            {
              "status": "success",
              "message": "Task 'Task1' executed.",
              "task_name": "Task1",
              "details": { /* ... */ }
            },
            {
              "status": "skipped",
              "message": "Task 'Task2' is disabled.",
              "task_name": "Task2",
              "details": null
            }
          ]
        }
        ```

*   **`GET /`** and **`GET /health`**:
    *   **Purpose**: Health check endpoints to verify if the API service is running.
    *   **Response**:
        ```json
        {
          "status": "ok",
          "status": "ok",
          "message": "AgenticMaid API is running."
        }
        ```

### 6. Using the CLI Tool

`AgenticMaid` also provides a command-line interface (CLI) tool to execute all enabled scheduled tasks based on a provided configuration file. This is useful for batch processing or running tasks from a terminal or script.

**To use the CLI tool:**

Ensure the `AgenticMaid` package and its dependencies are accessible in your `PYTHONPATH`.

From the root directory of the `Agent` project, you can run the CLI module as follows:

```bash
python -m AgenticMaid.cli --config-file path/to/your/config.json
```

**Arguments:**

*   `--config-file` (required): Path to the JSON configuration file for `AgenticMaid`. This file should define `ai_services`, `mcp_servers`, and `scheduled_tasks` as needed. Refer to [`AgenticMaid/config.example.json`](./config.example.json) for the structure.

**Behavior:**

1.  The CLI tool will load the specified configuration file.
2.  It will initialize an `ClientAgenticMaid` instance with this configuration.
3.  It will then attempt to execute all tasks listed in the `scheduled_tasks` section of the configuration that have `"enabled": true`.
4.  Output, including task execution status and any results or errors, will be logged to the console.

**Example Command:**

```bash
python -m AgenticMaid.cli --config-file ./AgenticMaid/config.example.json
```

This command will run all enabled scheduled tasks defined in [`AgenticMaid/config.example.json`](./config.example.json). Check the console output for details on each task's execution.

The original "Examples" section is now renumbered.

]]>
### 7. Multi-Agent Dispatch

The `AgenticMaid` supports a multi-agent dispatch feature, allowing one agent to invoke another. This enables the creation of sophisticated, hierarchical agent structures where a primary agent can delegate specific tasks to specialized agents.

#### a. Configuration

To enable this feature, you must add the `multi_agent_dispatch` section to your `config.json` file.

**Configuration Fields:**

*   `enabled` (boolean): Set to `true` to enable the feature.
*   `default_mode` (string): Determines the default invocation mode.
    *   `synchronous` or `sync`: The calling agent waits for the target agent to complete its task and return a result.
    *   `concurrent`: The calling agent invokes the target agent and immediately continues its own execution without waiting for a result.
*   `allowed_invocations` (object): A dictionary defining which agents are permitted to call others.
    *   The keys are the `agent_id` of the *calling* agent (from the `agents` section of your config).
    *   The values are an array of strings, where each string is the `agent_id` of a *target* agent that can be called.
    *   A wildcard `"*"` can be used in the array to allow an agent to call *any* other configured agent.

**Example `config.json` Snippet:**

```json
{
  "multi_agent_dispatch": {
    "enabled": true,
    "default_mode": "concurrent",
    "allowed_invocations": {
      "orchestrator_agent": [
        "*"
      ],
      "summary_agent_config_ref": [
        "report_agent_v2"
      ],
      "report_agent_v2": []
    }
  },
  "agents": {
    "orchestrator_agent": {
      "model_config_name": "google_gemini_default"
    },
    "summary_agent_config_ref": {
      "model_config_name": "google_gemini_default"
    },
    "report_agent_v2": {
      "model_config_name": "anthropic_claude4_opus"
    }
  }
}
```

In this example:
*   `orchestrator_agent` can call any other agent.
*   `summary_agent_config_ref` can only call `report_agent_v2`.
*   `report_agent_v2` cannot call any other agents.

#### b. Usage in Prompts

When the feature is enabled, a `dispatch` tool is automatically made available to the agents that are permitted to call others. To use it, instruct the agent in your prompt to call the `dispatch` tool with the required parameters.

**Dispatch Tool Parameters:**

*   `agent_id` (string): The ID of the target agent to invoke.
*   `prompt` (string): The prompt or instruction to pass to the target agent.
*   `mode` (string, optional): The invocation mode (`sync` or `concurrent`). If omitted, the `default_mode` from the configuration is used.

**Example Prompt:**

```
"Please use the dispatch tool to ask the 'report_agent_v2' to generate a detailed analysis of the latest user feedback. Run this in sync mode."
```

The agent will then parse this instruction and execute the following tool call: `dispatch(agent_id='report_agent_v2', prompt='Generate a detailed analysis of the latest user feedback.', mode='sync')`.
## Examples

### Full Example Script (`example_usage.py`)

```python
import asyncio
import time
from pkg_AgenticMaid.client import ClientAgenticMaid # Placeholder: Actual class name from client.py

async def run_client_operations():
    config = {
        "ai_services": {
            "default_llm": {
                "provider": "Google", # Ensure GOOGLE_API_KEY is in .env
                "model": "gemini-2.5-pro"
            },
            "claude_opus_llm": {
                 "provider": "Anthropic", # Ensure ANTHROPIC_API_KEY is in .env
                 "model": "claude-4-opus"
            }
        },
        "mcp_servers": {
            # Define at least one MCP server for tools to be fetched.
            # This example assumes an MCP server is running at http://localhost:8001/mcp/v1
            # If not, mcp_tools will be empty.
            "my_mcp_server": {
                "adapter_type": "fastapi",
                "base_url": "http://localhost:8001/mcp/v1", # Replace with your actual MCP server URL
                "name": "Example MCP Server"
            }
        },
        "scheduled_tasks": [
            {
                "name": "Test Scheduled Task",
                "cron_expression": "daily at 00:00", # Will run once if current time is past 00:00 and scheduler is kept running
                "prompt": "This is a test scheduled prompt. What time is it using Gemini?",
                "model_config_name": "default_llm",
                "enabled": True # Set to False if you don't want it to run
            }
        ],
        "chat_services": [
            {
                "service_id": "test_chat_gemini",
                "llm_service_name": "default_llm"
            },
            {
                "service_id": "test_chat_claude",
                "llm_service_name": "claude_opus_llm"
            }
        ],
        "default_llm_service_name": "default_llm"
    }

    client = ClientAgenticMaid(config_path_or_dict=config)
    await client.async_initialize()

    if not client.config:
        print("Client configuration failed. Exiting.")
        return

    print(f"ClientAgenticMaid Initialized. Config Source: {client.config_source}")
    print(f"Available MCP Tools: {[tool.name for tool in client.mcp_tools] if client.mcp_tools else 'No tools fetched (check MCP server config and availability)'}")

    # 1. Agent Interaction with Gemini
    print("\n--- Testing Agent Interaction (Gemini) ---")
    interaction_messages_gemini = [{"role": "user", "content": "Tell me a fun fact using Gemini."}]
    interaction_response_gemini = await client.run_mcp_interaction(
        messages=interaction_messages_gemini,
        llm_service_name="default_llm", # Uses gemini-2.5-pro
        agent_key="fun_fact_agent_gemini"
    )
    print(f"Agent Interaction Response (Gemini): {interaction_response_gemini}")

    # 1b. Agent Interaction with Claude
    print("\n--- Testing Agent Interaction (Claude) ---")
    interaction_messages_claude = [{"role": "user", "content": "Tell me a different fun fact using Claude."}]
    interaction_response_claude = await client.run_mcp_interaction(
        messages=interaction_messages_claude,
        llm_service_name="claude_opus_llm", # Uses claude-4-opus
        agent_key="fun_fact_agent_claude"
    )
    print(f"Agent Interaction Response (Claude): {interaction_response_claude}")


    # 2. Chat Service with Gemini
    print("\n--- Testing Chat Service (Gemini) ---")
    chat_messages_gemini = [{"role": "user", "content": "Hi there, how are you? (Gemini)"}]
    chat_response_gemini = await client.handle_chat_message(
        service_id="test_chat_gemini",
        messages=chat_messages_gemini
    )
    print(f"Chat Service Response (Gemini): {chat_response_gemini}")

    # 2b. Chat Service with Claude
    print("\n--- Testing Chat Service (Claude) ---")
    chat_messages_claude = [{"role": "user", "content": "Hi there, how are you? (Claude)"}]
    chat_response_claude = await client.handle_chat_message(
        service_id="test_chat_claude",
        messages=chat_messages_claude
    )
    print(f"Chat Service Response (Claude): {chat_response_claude}")

    # 3. Scheduled Tasks
    print("\n--- Testing Scheduled Tasks ---")
    if client.scheduler.jobs:
        print(f"Scheduled jobs: {client.scheduler.jobs}")
        print("Starting scheduler for a short period (e.g., 5 seconds for demo)...")
        client.start_scheduler() # Starts a daemon thread

        # Keep the main script running for a bit to allow scheduler to work
        # In a real app, this would be part of the main application loop.
        # For this demo, we'll just sleep.
        # Note: 'daily at HH:MM' tasks might not run in this short window unless HH:MM is very soon.
        # Consider a more frequent cron_expression for immediate testing, e.g., using a custom parser for 'every X seconds'.
        await asyncio.sleep(5)
        print("Scheduler demo period finished.")
        # client.stop_scheduler() # Conceptual
    else:
        print("No tasks scheduled.")

if __name__ == "__main__":
    # Note: If your MCP server or .env setup is not complete, parts of this example might show warnings or errors.
    # Ensure an MCP server is running if you expect tools, and .env has API keys for LLM calls.
    print("Make sure your .env file (in AgenticMaid directory) has GOOGLE_API_KEY and ANTHROPIC_API_KEY set for this example to fully work.")
    print("Also, ensure an MCP server is running at the configured URL if you expect MCP tools.")
    asyncio.run(run_client_operations())

```

This README provides a comprehensive guide to installing, configuring, and using the `ClientAgenticMaid`. Remember to adapt paths and configurations to your specific project setup.