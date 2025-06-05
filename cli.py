import argparse
import asyncio
import json
import os
import logging

# Adjust import path to correctly find the AgenticMaid module
# Assume cli.py and client.py are in the same AgenticMaid directory
from .client import AgenticMaid

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

    args = parser.parse_args()
    config_file_path = args.config_file

    logger.info(f"Attempting to load configuration from: {config_file_path}")

    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found: {config_file_path}")
        return

    try:
        # AgenticMaid constructor accepts a file path
        client = AgenticMaid(config_path_or_dict=config_file_path)
    except Exception as e:
        logger.error(f"Failed to instantiate AgenticMaid with config '{config_file_path}': {e}", exc_info=True)
        return

    logger.info("AgenticMaid instantiated. Performing asynchronous initialization...")
    try:
        initialization_success = await client.async_initialize()
        if not initialization_success:
            logger.error("AgenticMaid asynchronous initialization failed. Check previous logs for details (e.g., config errors, MCP server issues).")
            if client.config is None:
                logger.error("Client configuration is None after initialization attempt. This often indicates a problem loading or parsing the config file.")
            return
        logger.info("AgenticMaid asynchronous initialization complete.")
    except Exception as e:
        logger.error(f"An error occurred during AgenticMaid async_initialize: {e}", exc_info=True)
        return

    if not client.config:
        logger.error("AgenticMaid configuration is missing after initialization. Cannot proceed.")
        return

    if not client.config.get("scheduled_tasks"):
        logger.warning("No 'scheduled_tasks' found in the configuration. Nothing to run by default.")
        # Even if there are no scheduled tasks, the client is initialized, and it can be considered that the CLI tool has completed the "load and prepare" part.
        # Depending on the goal, if it's just to run scheduled tasks, it can exit early here.
        # But if the goal is a broader "prepare client", then it can continue.
        # The goal is to "execute all scheduled tasks", so it can be considered that no tasks is also an effective completion.
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

if __name__ == "__main__":
    # Ensure that when running this script, PYTHONPATH includes the project's root directory,
    # so that `from .client import AgenticMaid` can be parsed correctly.
    # For example, run from the Agent directory: python -m AgenticMaid.cli --config-file AgenticMaid/config.example.json
    logger.info("AgenticMaid CLI starting...")
    asyncio.run(main())
    logger.info("AgenticMaid CLI finished.")