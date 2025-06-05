import asyncio
import os

# Ensure the script can find the AgenticMaid module.
# This assumes the script is run from the root of the 'Agent' project directory,
# or that AgenticMaid is in the PYTHONPATH.
# For direct execution within the AgenticMaid/examples directory, you might need to adjust sys.path:
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from AgenticMaid.client import ClientAgenticMaid # Assuming the class is ClientAgenticMaid

async def main():
    """
    Demonstrates how to instantiate and use ClientAgenticMaid with a direct dictionary configuration.
    """
    print("Starting direct invocation example for ClientAgenticMaid...")

    # 1. Define the configuration as a Python dictionary
    # This configuration is minimal and assumes certain environment variables might be set
    # (e.g., GOOGLE_API_KEY for the "default_gemini_service").
    # For a real scenario, expand this dictionary based on your needs, similar to config.example.json.
    direct_config = {
        "ai_services": {
            "default_gemini_service": {
                "provider": "Google",
                "model": "gemini-2.5-pro",
                # "api_key": "your_google_api_key_here" # Or ensure GOOGLE_API_KEY is in .env or environment
            },
            "default_claude_service": {
                "provider": "Anthropic",
                "model": "claude-4-opus",
                # "api_key": "your_anthropic_api_key_here" # Or ensure ANTHROPIC_API_KEY is in .env or environment
            }
        },
        "mcp_servers": {
            "mock_mcp_server": {
                "adapter_type": "fastapi", # This is just for structure; actual connection won't be made if base_url is dummy
                "base_url": "http://localhost:12345/mcp/v1", # A dummy URL, no actual server needs to run here for this example's purpose
                "name": "Mock MCP Server for Example",
                "description": "A mock server definition to show MCP tools fetching (will likely be empty without a real server)."
            }
        },
        "chat_services": [
            {
                "service_id": "example_chat_service_gemini",
                "llm_service_name": "default_gemini_service"
            },
            {
                "service_id": "example_chat_service_claude",
                "llm_service_name": "default_claude_service"
            }
        ],
        "default_llm_service_name": "default_gemini_service",
        "scheduled_tasks": [
            {
              "name": "Example Echo Task Gemini",
              "cron_expression": "daily at 23:59", # Won't actually run in this example
              "prompt": "Echo this: Hello from scheduled task using Gemini!",
              "model_config_name": "default_gemini_service",
              "enabled": False # Disabled for this example
            }
        ]
    }

    print(f"Using direct dictionary configuration: {direct_config}")

    # 2. Instantiate ClientAgenticMaid with the dictionary
    # Note: For API keys (like Google's), ensure they are either in your environment variables
    # (e.g., GOOGLE_API_KEY) and picked up by the .env loading mechanism within ClientAgenticMaid,
    # or explicitly provided in the 'ai_services' part of the dictionary if not using .env.
    # The AgenticMaid/.env file is typically located alongside client.py.
    print("\nInstantiating ClientAgenticMaid...")
    client = ClientAgenticMaid(config_path_or_dict=direct_config)

    # 3. Perform asynchronous initialization
    print("\nPerforming asynchronous initialization (fetching MCP tools, etc.)...")
    # This step will attempt to connect to MCP servers defined in 'mcp_servers'.
    # If 'mock_mcp_server' base_url is not a live MCP server, tool fetching will likely fail or return empty.
    # This is okay for demonstrating the instantiation process.
    initialization_successful = await client.async_initialize()

    if not initialization_successful or not client.config:
        print("ClientAgenticMaid initialization failed. Check configuration and logs.")
        if client.config is None:
            print("Reason: Client configuration object is None after init attempt.")
        return

    print("ClientAgenticMaid initialized successfully.")
    print(f"Fetched {len(client.mcp_tools)} MCP tools: {[tool.name for tool in client.mcp_tools]}")
    if not client.mcp_tools:
        print("Note: No MCP tools were fetched. This is expected if 'mock_mcp_server' is not a real, running MCP server.")

    # 4. Example: Using a chat service (Gemini)
    # This requires the 'default_gemini_service' to be correctly configured
    # and the necessary API key (e.g., GOOGLE_API_KEY) to be available.
    print("\n--- Example: Interacting with a Chat Service (Gemini) ---")
    chat_service_id_gemini = "example_chat_service_gemini"
    if any(cs['service_id'] == chat_service_id_gemini for cs in client.config.get("chat_services", [])):
        print(f"Attempting to use chat service: {chat_service_id_gemini}")
        messages_gemini = [{"role": "user", "content": "Hello! Can you tell me a very short story using Gemini?"}]
        try:
            if not os.getenv("GOOGLE_API_KEY") and \
               not client.ai_services.get("default_gemini_service", {}).get("api_key"):
                print("WARNING: GOOGLE_API_KEY not found in environment or config for 'default_gemini_service'. Gemini chat example will likely fail.")

            chat_response_gemini = await client.handle_chat_message(
                service_id=chat_service_id_gemini,
                messages=messages_gemini
            )
            print(f"Chat Response from '{chat_service_id_gemini}':")
            if isinstance(chat_response_gemini, dict) and "error" in chat_response_gemini:
                print(f"  Error: {chat_response_gemini['error']}")
            elif chat_response_gemini:
                print(f"  Full Response: {chat_response_gemini}")
                if isinstance(chat_response_gemini, dict) and 'messages' in chat_response_gemini and isinstance(chat_response_gemini['messages'], list):
                    for message in reversed(chat_response_gemini['messages']):
                        if message.get('role') == 'ai' or message.get('role') == 'assistant':
                            print(f"  Assistant's reply: {message.get('content')}")
                            break
                else:
                     print(f"  Raw Response: {chat_response_gemini}")
            else:
                print("  No response or an unexpected response format received.")
        except Exception as e:
            print(f"  Error during Gemini chat interaction: {e}")
            print("  This might be due to missing API keys, network issues, or misconfiguration.")
    else:
        print(f"Chat service '{chat_service_id_gemini}' not found in configuration.")

    # 4b. Example: Using a chat service (Claude)
    print("\n--- Example: Interacting with a Chat Service (Claude) ---")
    chat_service_id_claude = "example_chat_service_claude"
    if any(cs['service_id'] == chat_service_id_claude for cs in client.config.get("chat_services", [])):
        print(f"Attempting to use chat service: {chat_service_id_claude}")
        messages_claude = [{"role": "user", "content": "Hello! Can you tell me a very short story using Claude?"}]
        try:
            if not os.getenv("ANTHROPIC_API_KEY") and \
               not client.ai_services.get("default_claude_service", {}).get("api_key"):
                print("WARNING: ANTHROPIC_API_KEY not found in environment or config for 'default_claude_service'. Claude chat example will likely fail.")

            chat_response_claude = await client.handle_chat_message(
                service_id=chat_service_id_claude,
                messages=messages_claude
            )
            print(f"Chat Response from '{chat_service_id_claude}':")
            if isinstance(chat_response_claude, dict) and "error" in chat_response_claude:
                print(f"  Error: {chat_response_claude['error']}")
            elif chat_response_claude:
                print(f"  Full Response: {chat_response_claude}")
                if isinstance(chat_response_claude, dict) and 'messages' in chat_response_claude and isinstance(chat_response_claude['messages'], list):
                    for message in reversed(chat_response_claude['messages']):
                        if message.get('role') == 'ai' or message.get('role') == 'assistant':
                            print(f"  Assistant's reply: {message.get('content')}")
                            break
                else:
                     print(f"  Raw Response: {chat_response_claude}")
            else:
                print("  No response or an unexpected response format received.")
        except Exception as e:
            print(f"  Error during Claude chat interaction: {e}")
    else:
        print(f"Chat service '{chat_service_id_claude}' not found in configuration.")

    # 5. Example: Triggering a (disabled) scheduled task by name (for demonstration)
    print("\n--- Example: Manually Triggering a (Disabled) Scheduled Task by Name (Gemini) ---")
    task_name_to_run_gemini = "Example Echo Task Gemini"
    task_run_result_gemini = await client.async_run_scheduled_task_by_name(task_name_to_run_gemini)

    if task_run_result_gemini:
        print(f"Result of trying to run task '{task_name_to_run_gemini}':")
        print(f"  Status: {task_run_result_gemini.get('status')}")
        print(f"  Message: {task_run_result_gemini.get('message')}")
        if task_run_result_gemini.get('status') == 'success':
            print(f"  Response: {task_run_result_gemini.get('response')}")
        elif task_run_result_gemini.get('status') == 'error':
            print(f"  Error Details: {task_run_result_gemini.get('error_details') or task_run_result_gemini.get('error')}")
    else:
        print(f"No result from trying to run task '{task_name_to_run_gemini}'.")

    print("\nDirect invocation example finished.")

if __name__ == "__main__":
    print("Reminder: For the chat examples to fully work,")
    print("ensure your GOOGLE_API_KEY (for Gemini) and ANTHROPIC_API_KEY (for Claude) are set")
    print("in your environment or in a .env file in the AgenticMaid directory (alongside client.py).\n")
    asyncio.run(main())