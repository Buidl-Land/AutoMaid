import argparse
import asyncio
import json
import os
import logging
import signal
import sys

from .client import AgenticMaid
from .conversation_logger import init_conversation_logger

# --- Logging Setup ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) # Set root logger level

# Remove any existing handlers to avoid duplicates if this module is reloaded
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# This is the most reliable way to set the encoding for stdout on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except TypeError:
    pass # Fails in some environments, but we have a fallback.

# Console Handler with UTF-8 encoding support
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# File Handler for CLI logs with UTF-8 encoding
log_file_path = os.path.join(LOG_DIR, "cli.log")
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.INFO) # Or logging.DEBUG for more verbose file logs
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__) # Get a logger for this specific module

async def start_cli_session(client: AgenticMaid):
    """Starts an interactive command-line session for conversation."""
    logger.info("Starting interactive CLI session. Type 'exit' or 'quit' to end.")

    # Use the default LLM service name from the config, or the first one if not specified
    llm_service_name = client.config.get("default_llm_service_name")
    if not llm_service_name and client.ai_services:
        llm_service_name = list(client.ai_services.keys())[0]

    if not llm_service_name:
        logger.error("No LLM service configured. Cannot start CLI session.")
        return

    logger.info(f"Using LLM service: {llm_service_name}")

    messages = []
    # You can add a system prompt here if you want
    # messages.append({"role": "system", "content": "You are a helpful assistant."})

    while True:
        try:
            prompt = await asyncio.to_thread(input, "You: ")
            if prompt.lower() in ["exit", "quit"]:
                logger.info("Exiting interactive session.")
                break

            messages.append({"role": "user", "content": prompt})

            response = await client.run_mcp_interaction(
                messages=messages,
                llm_service_name=llm_service_name,
                agent_key="cli_agent"
            )

            if response and "messages" in response:
                # The agent's response is expected to be the last message
                ai_message = response["messages"][-1]
                ai_content = ai_message.content if hasattr(ai_message, 'content') else str(ai_message)

                print(f"Agent: {ai_content}")

                # Add the AI's response to the history
                messages.append({"role": "assistant", "content": ai_content})
            else:
                error_message = response.get("error", "An unknown error occurred.")
                print(f"Agent: Sorry, I encountered an error: {error_message}")

        except (KeyboardInterrupt, EOFError):
            logger.info("\nExiting interactive session.")
            break
        except Exception as e:
            logger.error(f"An error occurred during the chat session: {e}", exc_info=True)
            print("Agent: An unexpected error occurred. Please try again.")

async def async_main():
    """
    Main entry point for the CLI tool.
    Parses command line arguments, loads configuration, initializes AgenticMaid, and executes all enabled scheduled tasks.
    """
    parser = argparse.ArgumentParser(
        description="Run AgenticMaid tasks from a configuration file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the JSON configuration file for AgenticMaid."
    )
    parser.add_argument(
        "--run-task-now",
        type=str,
        help="Name of a specific scheduled task to run immediately, ignoring its schedule."
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Launch an interactive CLI session for conversation."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for detailed network requests."
    )

    args = parser.parse_args()

    # If debug mode is enabled, set the root logger level to DEBUG
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        for handler in root_logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled. Log level set to DEBUG.")

    config_file_path = args.config_file

    logger.info(f"Attempting to load configuration from: {config_file_path}")

    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found: {config_file_path}")
        return

    # Initialize conversation logger. This will now use the centralized config manager.
    init_conversation_logger(log_dir="logs", enable_file_logging=True)

    # Global variable to store client for cleanup
    client = None

    # Define signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if client:
            try:
                # Run cleanup in the event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(client.cleanup_mcp_sessions())
                else:
                    asyncio.run(client.cleanup_mcp_sessions())
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load the configuration early to set environment variables if needed
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse config file {config_file_path}: {e}", exc_info=True)
        return

    # Set OpenAI API key from config to environment variable for global access
    # This helps ensure libraries like LangChain can find it reliably.
    if 'ai_services' in config_data:
        for service_name, service_details in config_data['ai_services'].items():
            if 'api_key' in service_details and service_details['api_key']:
                os.environ["OPENAI_API_KEY"] = service_details['api_key']
                logger.info(f"Set OPENAI_API_KEY environment variable from service '{service_name}'.")
                break # Exit after setting the first key

    try:
        # Pass the config file path and debug flag to AgenticMaid
        client = AgenticMaid(config_path_or_dict=config_file_path, debug=args.debug)
    except Exception as e:
        logger.error(f"Failed to instantiate AgenticMaid with config '{config_file_path}': {e}", exc_info=True)
        return

    logger.info("AgenticMaid instantiated. Performing asynchronous initialization...")
    try:
        initialization_success = await client.async_initialize()
        if not initialization_success:
            logger.error("AgenticMaid asynchronous initialization failed.")
            return
        logger.info("AgenticMaid asynchronous initialization complete.")
    except Exception as e:
        logger.error(f"An error occurred during AgenticMaid async_initialize: {e}", exc_info=True)
        return

    if not client.config:
        logger.error("AgenticMaid configuration is missing after initialization. Cannot proceed.")
        return

    if args.cli:
        await start_cli_session(client)
    elif args.run_task_now:
        task_name_to_run = args.run_task_now
        logger.info(f"Attempting to run specific task immediately: {task_name_to_run}")
        try:
            result = await client.async_run_scheduled_task_by_name(task_name_to_run)
            print(result)

            # Handle case where result might be None
            if result is None:
                logger.error(f"Task '{task_name_to_run}' returned None - execution failed")
                return

            status = result.get('status', 'unknown')
            message = result.get('message', '')
            details = result.get('response', result.get('error', ''))
            log_level = logging.INFO
            if status == "error":
                log_level = logging.ERROR
            elif status == "skipped":
                log_level = logging.WARNING

            logger.log(log_level, f"Task '{task_name_to_run}' | Status: {status} | Message: {message} | Details: {details}")
        except Exception as e:
            logger.error(f"An error occurred while running task '{task_name_to_run}': {e}", exc_info=True)
    else:
        if not client.config.get("scheduled_tasks"):
            logger.warning("No 'scheduled_tasks' found in the configuration. Nothing to run by default.")
            logger.info("CLI execution finished: No scheduled tasks to execute.")
            return

        logger.info("Executing all enabled scheduled tasks...")
        try:
            results = await client.async_run_all_enabled_scheduled_tasks()
            logger.info("Finished running all enabled scheduled tasks.")

            if results:
                logger.info("Task execution results:")
                for i, result in enumerate(results):
                    task_name = result.get('task_name', f"Unnamed Task {i+1}")
                    status = result.get('status', 'unknown')
                    message = result.get('message', '')
                    details = result.get('response', result.get('error', ''))

                    log_level = logging.INFO
                    if status == "error":
                        log_level = logging.ERROR
                    elif status == "skipped":
                        log_level = logging.WARNING

                    logger.log(log_level, f"  Task: {task_name} | Status: {status} | Message: {message} | Details: {details}")
            else:
                logger.info("No results returned from running scheduled tasks (this might be normal if no tasks were enabled or found).")
        except Exception as e:
            logger.error(f"An error occurred while running scheduled tasks: {e}", exc_info=True)

    # Clean up MCP sessions before saving logs
    try:
        if hasattr(client, 'cleanup_mcp_sessions'):
            await client.cleanup_mcp_sessions()
    except Exception as e:
        logger.debug(f"MCP cleanup completed with minor issues: {e}")

    # Save conversation records
    try:
        saved_files = client.save_conversation_logs()
        if saved_files:
            logger.info("Conversation records saved successfully")
    except Exception as e:
        logger.warning(f"Error saving conversation records: {e}")

    # Final message to indicate completion
    logger.info("AgenticMaid CLI finished.")

def custom_exception_handler(loop, context):
    """Custom exception handler to suppress known langchain_mcp_adapters cleanup errors"""
    exception = context.get('exception')
    message = context.get('message', '')

    # Suppress known MCP cleanup errors and warnings
    if exception:
        if isinstance(exception, RuntimeError) and "cancel scope" in str(exception):
            return  # Silently ignore
        if isinstance(exception, Exception) and "async_generator_athrow" in str(exception):
            return  # Silently ignore

    # Suppress specific warning messages
    if "Task was destroyed but it is pending" in message:
        return  # Silently ignore
    if "coroutine method 'aclose'" in message:
        return  # Silently ignore

    # For other exceptions, use default handling
    loop.default_exception_handler(context)

def main():
    """Main entry point for the AgenticMaid CLI."""
    logger.info("AgenticMaid CLI starting...")

    # Suppress MCP-related warnings
    import warnings
    warnings.filterwarnings("ignore", message=".*coroutine method 'aclose'.*")
    warnings.filterwarnings("ignore", message=".*Enable tracemalloc.*")

    # Set custom exception handler to suppress MCP cleanup errors
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(custom_exception_handler)

    try:
        loop.run_until_complete(async_main())
    except KeyboardInterrupt:
        logger.info("AgenticMaid CLI interrupted by user (Ctrl+C)")
        # Try to perform graceful cleanup
        try:
            # Note: We can't access the client variable here since it's in async_main()
            # The signal handler should have already handled cleanup
            logger.info("Performing final cleanup...")
        except Exception as cleanup_error:
            logger.debug(f"Cleanup error during interrupt: {cleanup_error}")
    except Exception as e:
        logger.error(f"AgenticMaid CLI encountered an error: {e}")
    finally:
        try:
            # Cancel all pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as e:
            logger.debug(f"Error cancelling tasks: {e}")
        finally:
            loop.close()


if __name__ == "__main__":
    main()