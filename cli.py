import argparse
import asyncio
import json
import os
import logging

# Adjust import path to correctly find the AgenticMaid module
# Assume cli.py and client.py are in the same AgenticMaid directory
from client import AgenticMaid
from conversation_logger import init_conversation_logger

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

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# File Handler for CLI logs
log_file_path = os.path.join(LOG_DIR, "cli.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO) # Or logging.DEBUG for more verbose file logs
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__) # Get a logger for this specific module

async def main():
    """
    Main entry point for the CLI tool.
    Parses command line arguments, loads configuration, initializes AgenticMaid, and executes all enabled scheduled tasks.
    """
    # Initialize conversation logger
    init_conversation_logger(log_dir="logs", enable_file_logging=True)

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

    args = parser.parse_args()
    config_file_path = args.config_file

    logger.info(f"Attempting to load configuration from: {config_file_path}")

    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found: {config_file_path}")
        return

    try:
        client = AgenticMaid(config_path_or_dict=config_file_path)
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

    if args.run_task_now:
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

if __name__ == "__main__":
    # Ensure that when running this script, PYTHONPATH includes the project's root directory,
    # so that `from .client import AgenticMaid` can be parsed correctly.
    # For example, run from the Agent directory: python -m AgenticMaid.cli --config-file AgenticMaid/config.example.json
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
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("AgenticMaid CLI interrupted by user")
    except Exception as e:
        logger.error(f"AgenticMaid CLI encountered an error: {e}")
    finally:
        loop.close()
        logger.info("AgenticMaid CLI finished.")