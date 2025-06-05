import json
import schedule
import time
import threading
import os
import copy
import asyncio
import logging # Added
from dotenv import dotenv_values

from langchain_core.runnables import RunnableConfig
# from langgraph.channels.base import BaseChannel # Not directly used yet
# from langgraph.prebuilt.tool_executor import ToolExecutor # Not directly used yet, tools come from MCP
from langchain_mcp_adapters.client import MultiServerAgenticMaid
from langgraph.prebuilt import create_react_agent
# Import other necessary langchain and AI service related libraries

logger = logging.getLogger(__name__) # Added logger

class AgenticMaid:
    def __init__(self, config_path_or_dict):
        """
        Initializes the AgenticMaid.
        Note: Call `await client.async_initialize()` after creating an instance
        to complete asynchronous setup like fetching MCP tools.

        Args:
            config_path_or_dict (str or dict): Path to a JSON configuration file
                                               or a Python dictionary containing the configuration.
        """
        self.config_source = config_path_or_dict
        self.config = None
        self.mcp_client: MultiServerAgenticMaid | None = None
        self.mcp_tools: list = []
        self.agents: dict = {} # To store created agents

        env_base_config = self._load_env_config()
        main_config = self._load_main_config()

        if main_config is None:
            logger.error("Main configuration could not be loaded. AgenticMaid initialization failed.")
            # self.config remains None
        else:
            self.config = self._merge_configs(env_base_config, main_config)

        # self.mcp_services = {} # This will be managed by MultiServerAgenticMaid
        self.ai_services = {}  # To store AI service (LLM) instances/configurations
        self.scheduler = schedule.Scheduler()

        if self.config:
            # Asynchronous initialization needs to be called separately
            # self._initialize_services() # This is now async_initialize()
            self._schedule_tasks() # Synchronous part of setup
        else:
            logger.warning("AgenticMaid not fully initialized due to configuration errors. Call await async_initialize() after fixing config.")

    async def async_initialize(self, reconfiguring=False):
        """
        Performs asynchronous initialization tasks, primarily initializing MCP services
        and fetching tools. This should be called after the client is constructed or
        when reconfiguring.

        Args:
            reconfiguring (bool): If True, indicates that this is part of a reconfiguration.
        """
        if not self.config:
            logger.error("Cannot perform async initialization because main configuration is missing.")
            return False # Indicate failure

        # Reset services and tools if reconfiguring
        if reconfiguring:
            self.mcp_client: MultiServerAgenticMaid | None = None
            self.mcp_tools: list = []
            self.agents: dict = {}
            self.ai_services = {}
            self.scheduler = schedule.Scheduler() # Reset scheduler

        await self._initialize_services()
        self._schedule_tasks() # Reschedule tasks with new config if any
        # Any other async setup can go here.
        logger.info(f"Async initialization {'(reconfiguration)' if reconfiguring else ''} complete.")
        return True # Indicate success

    async def async_reconfigure(self, new_config_path_or_dict):
        """
        Reconfigures the AgenticMaid with a new configuration.
        This will reload the configuration, re-initialize services, and reschedule tasks.

        Args:
            new_config_path_or_dict (str or dict): Path to a new JSON configuration file
                                                   or a Python dictionary containing the new configuration.
        Returns:
            bool: True if reconfiguration was successful, False otherwise.
        """
        logger.info(f"Attempting to reconfigure AgenticMaid with: {new_config_path_or_dict}")
        self.config_source = new_config_path_or_dict

        env_base_config = self._load_env_config()
        main_config = self._load_main_config()

        if main_config is None:
            logger.error("New main configuration could not be loaded. Reconfiguration failed.")
            self.config = None # Ensure config is None if loading fails
            return False

        self.config = self._merge_configs(env_base_config, main_config)

        if not self.config:
            logger.error("Configuration merging failed during reconfiguration.")
            return False

        return await self.async_initialize(reconfiguring=True)

    def _load_env_config(self):
        """
        Loads configuration from .env file located in AgenticMaid/ directory.
        Example .env file (AgenticMaid/.env):
        ```
        # AgenticMaid/.env example for default values
        # These can be overridden by the main JSON/dict configuration.

        # Default API key if not specified per service in main config
        # DEFAULT_API_KEY=your_default_api_key_here

        # Default model name if not specified per service in main config
        # DEFAULT_MODEL_NAME=gpt-3.5-turbo-default-from-env

        # Provider-specific defaults
        OPENAI_API_KEY=your_openai_api_key_from_env
        OPENAI_DEFAULT_MODEL=gpt-3.5-turbo-openai-from-env

        ANTHROPIC_API_KEY=your_anthropic_api_key_from_env
        ANTHROPIC_DEFAULT_MODEL=claude-2-from-env

        # You can add other general or provider-specific defaults here
        # AZURE_OPENAI_API_KEY=your_azure_openai_key
        # AZURE_OPENAI_ENDPOINT=your_azure_endpoint
        # AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
        ```
        Returns:
            dict: A dictionary of environment variables from .env file.
                  Returns an empty dict if .env is not found or is empty.
        """
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            logger.info(f"Loading .env file from: {env_path}")
            return dotenv_values(env_path)
        else:
            logger.info(f".env file not found at {env_path}. No .env defaults will be loaded.")
            return {}

    def _load_main_config(self):
        """Loads the main configuration from a JSON file or uses the provided dictionary."""
        if isinstance(self.config_source, str):
            # Load from JSON file path
            try:
                with open(self.config_source, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded successfully from {self.config_source}")
                return config
            except FileNotFoundError:
                logger.error(f"Configuration file {self.config_source} not found.")
                return None
            except json.JSONDecodeError:
                logger.error(f"Configuration file {self.config_source} is not valid JSON.")
                return None
            except Exception as e:
                logger.error(f"Error loading configuration from {self.config_source}: {e}")
                return None
        elif isinstance(self.config_source, dict):
            # Use the provided dictionary directly
            logger.info("Configuration loaded successfully from dictionary.")
            return copy.deepcopy(self.config_source) # Return a copy to avoid external modification
        else:
            logger.error("Invalid configuration source. Must be a file path (str) or a dictionary.")
            return None

    def _merge_configs(self, env_config, main_config):
        """
        Merges environment configuration (defaults) with the main configuration.
        Main configuration values take precedence.

        Args:
            env_config (dict): Configuration loaded from .env file.
            main_config (dict): Main configuration (from JSON file or dict).

        Returns:
            dict: The merged configuration.
        """
        if not main_config: # Should not happen if _load_main_config handles errors properly
            return env_config # Or perhaps an empty dict or None

        merged_config = copy.deepcopy(main_config)

        # Merge ai_services: main_config overrides .env values
        # .env provides defaults if specific keys are missing in main_config services
        if "ai_services" in merged_config and isinstance(merged_config["ai_services"], dict):
            for service_name, service_details in merged_config["ai_services"].items():
                if not isinstance(service_details, dict):
                    logger.warning(f"Service '{service_name}' details are not a dictionary. Skipping .env merge for it.")
                    continue

                provider = service_details.get("provider", "").upper()

                # API Key: Use from main_config if present, else try .env
                if "api_key" not in service_details or not service_details["api_key"]:
                    env_api_key = env_config.get(f"{provider}_API_KEY") if provider else None
                    if not env_api_key: # Try a generic default if provider-specific is not found
                        env_api_key = env_config.get("DEFAULT_API_KEY")
                    if env_api_key:
                        service_details["api_key"] = env_api_key
                        logger.info(f"Using API key from .env for AI service '{service_name}'.")
                    else:
                        logger.warning(f"API key for AI service '{service_name}' not found in main config or .env.")

                # Model: Use from main_config if present, else try .env
                if "model" not in service_details or not service_details["model"]:
                    env_model = env_config.get(f"{provider}_DEFAULT_MODEL") if provider else None
                    if not env_model: # Try a generic default
                        env_model = env_config.get("DEFAULT_MODEL_NAME")
                    if env_model:
                        service_details["model"] = env_model
                        logger.info(f"Using model from .env for AI service '{service_name}'.")

                # Example for other common parameters like base_url for OpenAI-compatible APIs
                if "base_url" not in service_details or not service_details["base_url"]:
                    env_base_url = env_config.get(f"{provider}_BASE_URL") if provider else None
                    if not env_base_url:
                        env_base_url = env_config.get("DEFAULT_BASE_URL")
                    if env_base_url:
                         service_details["base_url"] = env_base_url
                         logger.info(f"Using base_url from .env for AI service '{service_name}'.")


        # Placeholder for merging other top-level configurations if needed in the future
        # For example, if mcp_services also needed .env defaults:
        # if "mcp_services" in merged_config and isinstance(merged_config["mcp_services"], dict):
        #     for service_name, service_details in merged_config["mcp_services"].items():
        #         # ... merging logic for mcp_services ...
        #         #         pass

        logger.info("Configuration merged successfully.")
        return merged_config

    async def _initialize_services(self):
        """
        Initializes MCP services using MultiServerAgenticMaid and fetches tools.
        Also prepares AI service configurations (LLMs).
        """
        if not self.config:
            logger.error("Configuration not loaded, cannot initialize services.")
            return

        # Initialize MultiServerAgenticMaid for MCP Tools
        mcp_server_configs = self.config.get("mcp_servers", {}) # Corrected key from "mcp_services"
        if not mcp_server_configs:
            logger.warning("No 'mcp_servers' found in configuration. MCP tools will not be available.")
        else:
            try:
                logger.info(f"Initializing MultiServerAgenticMaid with servers: {mcp_server_configs}")
                self.mcp_client = MultiServerAgenticMaid(mcp_server_configs)
                logger.info("Fetching MCP tools...")
                self.mcp_tools = await self.mcp_client.get_tools()
                logger.info(f"Successfully fetched {len(self.mcp_tools)} MCP tools: {[tool.name for tool in self.mcp_tools]}")
            except Exception as e:
                logger.error(f"Error initializing MultiServerAgenticMaid or fetching tools: {e}", exc_info=True)
                self.mcp_client = None
                self.mcp_tools = []

        # Prepare AI Services (LLM configurations)
        # Actual LLM client instantiation might happen on-demand or here
        # For now, just store the configurations.
        ai_config = self.config.get("ai_services", {})
        if not ai_config:
            logger.warning("No 'ai_services' found in configuration. LLM interactions might fail.")

        self.ai_services = ai_config # Store the merged AI service configs
        for service_name, service_details in self.ai_services.items():
            logger.info(f"AI Service '{service_name}' configured with model: {service_details.get('model')}")
            # Actual LLM client (e.g., ChatOpenAI, ChatAnthropic) could be initialized here
            # and stored, e.g., self.llms[service_name] = ChatOpenAI(**params)
            # For create_react_agent, we often pass the model name string directly or an LLM instance.

    def _schedule_tasks(self):
        """Schedules tasks based on cron expressions in the configuration."""
        if not self.config:
            return

        tasks = self.config.get("scheduled_tasks", [])

        def _execute_task_wrapper(task_details_sync):
            """Synchronous wrapper to run the async _execute_task."""
            # It's generally better to run async tasks in a managed event loop,
            # but for 'schedule' library compatibility, asyncio.run() is often used here.
            # Ensure this doesn't conflict if an outer event loop is already running in the main app.
            # If the main application is already async, different integration patterns for `schedule` are needed.
            try:
                asyncio.run(self._execute_task(task_details_sync))
            except RuntimeError as e:
                if " asyncio.run() cannot be called from a running event loop" in str(e):
                    # This case needs a more sophisticated solution, e.g., using loop.create_task()
                    # from an existing loop, or a thread pool executor for the blocking asyncio.run().
                    # For now, just print a warning.
                    logger.warning(f"Could not run async task '{task_details_sync.get('name')}' via asyncio.run() from current context: {e}")
                    logger.warning("Consider using a different scheduling approach if AgenticMaid is run within an existing asyncio loop.")
                else:
                    logger.error(f"Runtime error executing task '{task_details_sync.get('name')}': {e}", exc_info=True)
            except Exception as e:
                logger.error(f"General error executing task '{task_details_sync.get('name')}': {e}", exc_info=True)


        for task in tasks:
            cron_expr = task.get("cron_expression")
            # 'action' key is part of task_details, not directly used by scheduler.do() itself for condition
            if cron_expr and task.get("enabled", True): # Check if task is enabled
                try:
                    # Simple cron parsing, can be extended
                    if "daily at" in cron_expr.lower():
                        time_str = cron_expr.lower().split("daily at")[1].strip()
                        self.scheduler.every().day.at(time_str).do(_execute_task_wrapper, task_details_sync=task)
                        logger.info(f"Task '{task.get('name', 'Unnamed Task')}' scheduled: {cron_expr}")
                    # Add more cron parsing logic here if needed (e.g., hourly, weekly)
                    # Example:
                    # elif "every hour at" in cron_expr.lower():
                    #     minute_str = cron_expr.lower().split("every hour at")[1].strip() # e.g., ":00" or ":30"
                    #     self.scheduler.every().hour.at(minute_str).do(_execute_task_wrapper, task_details_sync=task)
                    #     logger.info(f"Task '{task.get('name', 'Unnamed Task')}' scheduled: {cron_expr}")
                    else:
                        logger.warning(f"Cannot parse cron expression '{cron_expr}' for task '{task.get('name', 'Unnamed Task')}'. Only 'daily at HH:MM' currently supported by this simple parser.")
                except Exception as e:
                    logger.error(f"Error scheduling task '{task.get('name', 'Unnamed Task')}': {e}", exc_info=True)
            elif not task.get("enabled", True):
                logger.info(f"Task '{task.get('name', 'Unnamed Task')}' is disabled and will not be scheduled.")


    async def _execute_task(self, task_details):
        """
        Executes a scheduled task by creating/retrieving an agent and invoking it.
        """
        task_name = task_details.get('name', 'Unnamed Task')
        logger.info(f"Executing scheduled task: {task_name}")

        prompt = task_details.get("prompt")
        if not prompt:
            logger.error(f"Task '{task_name}' has no prompt. Skipping.")
            return

        # Determine agent configuration:
        # It could be an explicit agent_id, or direct model/tool config within task_details
        agent_id = task_details.get("agent_id")
        model_config_name = task_details.get("model_config_name") # e.g., "default_openai"

        if not agent_id and not model_config_name:
            logger.error(f"Task '{task_name}' needs 'agent_id' or 'model_config_name' to run. Skipping.")
            return

        # TODO: Enhance logic to select or create agent based on agent_id or specific model/tool config
        # For now, assume agent_id maps to a pre-configured or default model for simplicity

        # Use a generic method to handle the interaction
        # This assumes handle_interaction can resolve agent_id or use a default model if needed
        try:
            # For scheduled tasks, we'll need to create a message structure
            messages = [{"role": "user", "content": prompt}]
            # The agent_id or a more complex config object would be passed to handle_interaction
            # If agent_id is present, it implies a pre-configured agent or one to be created with default tools.
            # If model_config_name is present, an agent might be created on-the-fly with that model and all MCP tools.

            # This part needs to align with how run_mcp_interaction / _get_or_create_agent is structured
            # For now, placeholder for direct call if _get_or_create_agent and agent.ainvoke are ready
            agent_key = agent_id or model_config_name # Simplistic agent key

            llm_config_name = model_config_name
            if agent_id and agent_id in self.config.get("agents", {}): # If pre-defined agent configs exist
                 llm_config_name = self.config["agents"][agent_id].get("model_config_name", model_config_name)

            if not llm_config_name: # Fallback to a globally default LLM if any
                llm_config_name = self.config.get("default_llm_service_name") # Assuming such a config might exist

            if not llm_config_name:
                logger.error(f"No LLM configuration specified or found for task '{task_name}'. Skipping.")
                return

            agent = await self._get_or_create_agent(agent_key, llm_config_name)
            if not agent:
                logger.error(f"Could not get or create agent for task '{task_name}'. Skipping.")
                return

            logger.info(f"Invoking agent for task '{task_name}' with prompt: '{prompt}'")
            response = await agent.ainvoke({"messages": messages})
            logger.info(f"Task '{task_name}' response: {response}")
        except Exception as e:
            logger.error(f"Error executing task '{task_name}': {e}", exc_info=True)
            # Optionally re-raise or return an error status
            return {"status": "error", "task_name": task_name, "error": str(e)}
        return {"status": "success", "task_name": task_name, "response": response}


    async def async_run_scheduled_task_by_name(self, task_name_to_run: str):
        """
        Finds a scheduled task by its name and executes it immediately.

        Args:
            task_name_to_run (str): The name of the scheduled task to run.

        Returns:
            dict: A dictionary containing the status of the execution and task details.
        """
        if not self.config or not self.config.get("scheduled_tasks"):
            logger.error(f"No scheduled tasks configured. Cannot run task '{task_name_to_run}'.")
            return {"status": "error", "message": f"No scheduled tasks configured. Cannot run task '{task_name_to_run}'."}

        task_details_to_run = None
        for task in self.config.get("scheduled_tasks", []):
            if task.get("name") == task_name_to_run:
                task_details_to_run = task
                break

        if not task_details_to_run:
            logger.error(f"Scheduled task '{task_name_to_run}' not found in configuration.")
            return {"status": "error", "message": f"Scheduled task '{task_name_to_run}' not found."}

        if not task_details_to_run.get("enabled", True):
            logger.info(f"Task '{task_name_to_run}' is disabled and will not be run.")
            return {"status": "skipped", "message": f"Task '{task_name_to_run}' is disabled."}

        logger.info(f"Manually triggering task: {task_name_to_run}")
        return await self._execute_task(task_details_to_run)

    async def async_run_all_enabled_scheduled_tasks(self):
        """
        Executes all enabled scheduled tasks immediately.

        Returns:
            list: A list of dictionaries, each containing the status of an executed task.
        """
        if not self.config or not self.config.get("scheduled_tasks"):
            logger.error("No scheduled tasks configured.")
            return [{"status": "error", "message": "No scheduled tasks configured."}]

        results = []
        enabled_tasks = [
            task for task in self.config.get("scheduled_tasks", []) if task.get("enabled", True)
        ]

        if not enabled_tasks:
            logger.info("No enabled scheduled tasks to run.")
            return [{"status": "skipped", "message": "No enabled scheduled tasks found."}]

        logger.info(f"Manually triggering all {len(enabled_tasks)} enabled scheduled tasks.")
        for task_details in enabled_tasks:
            task_name = task_details.get('name', 'Unnamed Task')
            logger.info(f"Triggering task: {task_name}")
            result = await self._execute_task(task_details)
            results.append(result)

        return results

    async def run_mcp_interaction(self, messages: list, llm_service_name: str, agent_key: str = "default_agent"):
        """
        Runs an interaction with an agent identified by agent_key,
        using the specified LLM service and available MCP tools.

        Args:
            messages (list): A list of messages for the agent (e.g., [{"role": "user", "content": "Hello"}]).
            llm_service_name (str): The name of the AI service (LLM configuration) to use.
            agent_key (str): A unique key to identify or create the agent.
                             If the agent doesn't exist, it will be created.

        Returns:
            The agent's response.
        """
        if not self.config:
            logger.error("Client not properly configured.")
            return None
        if not self.mcp_tools:
            logger.warning("No MCP tools available. Agent will run without them.")

        agent = await self._get_or_create_agent(agent_key, llm_service_name)
        if not agent:
            return {"error": f"Failed to get or create agent '{agent_key}' with LLM '{llm_service_name}'."}

        logger.info(f"Invoking agent '{agent_key}' (LLM: {llm_service_name}) with messages: {messages}")
        try:
            response = await agent.ainvoke({"messages": messages})
            return response
        except Exception as e:
            logger.error(f"Error during agent invocation for '{agent_key}': {e}", exc_info=True)
            return {"error": str(e)}

    async def _get_or_create_agent(self, agent_key: str, llm_service_name: str):
        """
        Retrieves an existing agent or creates a new one if it doesn't exist.
        Agents are stored in `self.agents`.

        Args:
            agent_key (str): The unique key for the agent.
            llm_service_name (str): The name of the AI service (LLM configuration) to use.

        Returns:
            An agent instance or None if creation fails.
        """
        if agent_key in self.agents:
            # TODO: Check if the llm_service_name matches the existing agent's LLM.
            # If not, decide whether to recreate or return existing. For now, return existing.
            logger.info(f"Returning existing agent: {agent_key}")
            return self.agents[agent_key]

        llm_config = self.ai_services.get(llm_service_name)
        if not llm_config:
            logger.error(f"LLM service configuration '{llm_service_name}' not found.")
            return None

        # We need the actual model identifier string for create_react_agent,
        # or an LLM instance. The config usually holds the model name.
        model_name_or_instance = llm_config.get("model") # e.g., "anthropic:claude-3-opus-latest" or "gpt-4"
        if not model_name_or_instance:
            logger.error(f"'model' not specified in LLM service config '{llm_service_name}'.")
            return None

        # TODO: If we were to initialize LLM instances (e.g. ChatOpenAI(model=...))
        # based on provider, this is where we'd do it and pass the instance.
        # For now, assuming create_react_agent handles model name strings for known providers.
        # Example: "openai:gpt-4-turbo" or just "gpt-4-turbo" if Langchain can infer.
        # The MCP docs show "anthropic:claude-3-7-sonnet-latest".

        logger.info(f"Creating new ReAct agent '{agent_key}' with LLM '{model_name_or_instance}' (from service '{llm_service_name}') and {len(self.mcp_tools)} tools.")
        try:
            agent_executor = create_react_agent(model_name_or_instance, self.mcp_tools)
            self.agents[agent_key] = agent_executor
            logger.info(f"Agent '{agent_key}' created successfully.")
            return agent_executor
        except Exception as e:
            logger.error(f"Error creating ReAct agent '{agent_key}': {e}", exc_info=True)
            return None

    # --- Chat Service Methods ---
    async def handle_chat_message(self, service_id: str, messages: list, stream: bool = False):
        """
        Handles an incoming chat message for a configured chat service.

        Args:
            service_id (str): The ID of the chat service (from config).
            messages (list): The list of messages for the chat.
            stream (bool): Whether to stream the response (not fully implemented yet).

        Returns:
            The agent's response or an error dictionary.
        """
        if not self.config or "chat_services" not in self.config:
            return {"error": "Chat services are not configured."}

        chat_service_config = None
        for service in self.config["chat_services"]:
            if service.get("service_id") == service_id:
                chat_service_config = service
                break

        if not chat_service_config:
            return {"error": f"Chat service with ID '{service_id}' not found."}

        llm_service_name = chat_service_config.get("llm_service_name") # Expect this in chat_service config
        if not llm_service_name:
            return {"error": f"Chat service '{service_id}' does not specify 'llm_service_name'."}

        # Agent key for chat could be the service_id itself or derived
        agent_key = f"chat_agent_{service_id}"

        # This reuses the main interaction logic
        response = await self.run_mcp_interaction(messages, llm_service_name, agent_key)

        if stream:
            # Placeholder for streaming logic
            # For true streaming, run_mcp_interaction and agent.astream_log() or similar would be needed.
            logger.info(f"Streaming for service '{service_id}' (placeholder). Response: {response}")
            # yield response # If this method were an async generator
            return {"warning": "Streaming not fully implemented", "response": response}
        else:
            return response

    # Original run_mcp_interaction is being replaced by the more generic one above.
    # Keeping it commented out for reference if needed, but the new one is preferred.
    # def run_mcp_interaction(self, mcp_service_name, ai_service_name, input_data, agent_definition):
    #     """
    #     Dynamically calls Langchain MCP service and configured AI service.
    #     :param mcp_service_name: Name of the MCP service to use (defined in config)
    #     :param ai_service_name: Name of the AI service to use (defined in config)
    #     :param input_data: Input data for the Langchain Agent
    #     :param agent_definition: Definition or reference of the Langchain Agent
    #     :return: Result of the Agent execution
    #     """
    #     if not self.config:
    #         print("Error: Client not properly configured.")
    #         return None
    #
    #     # mcp_service_config = self.config.get("mcp_servers", {}).get(mcp_service_name) # MCP client is now singular
    #     ai_service_config = self.config.get("ai_services", {}).get(ai_service_name)
    #
    #     # if not mcp_service_config: # Check self.mcp_client and self.mcp_tools instead
    #     #     print(f"Error: MCP service configuration for '{mcp_service_name}' not found.")
    #     #     return None
    #     if not self.mcp_client or not self.mcp_tools:
    #         print(f"Error: MCP Client not initialized or no tools found. Call await async_initialize().")
    #         return None
    #
    #     if not ai_service_config:
    #         print(f"Error: AI service configuration for '{ai_service_name}' not found.")
    #         return None
    #
    #     print(f"Preparing interaction with MCP tools and AI service '{ai_service_name}'.")
    #     print(f"Input data: {input_data}")
    #     print(f"Agent definition: {agent_definition}")
    #
    #     # This is where create_react_agent would be used.
    #     # The agent_definition would specify the model (from ai_service_config)
    #     # and tools would be self.mcp_tools.
    #
    #     # Placeholder for actual agent creation and invocation
    #     # model_name = ai_service_config.get("model")
    #     # agent = create_react_agent(model_name, self.mcp_tools)
    #     # response = await agent.ainvoke({"messages": [{"role": "user", "content": input_data.get("prompt")}]})
    #
    #     return {"message": f"Simulated call with MCP tools, AI:'{ai_service_name}' successful", "input": input_data} # Placeholder return

    def start_scheduler(self):
        """Starts the scheduler thread to run scheduled tasks in the background."""
        if not self.scheduler.jobs:
            logger.info("No scheduled tasks to run.")
            return

        def run_continuously(interval=1):
            # cease_continuous_run = threading.Event() # This was not fully implemented for stopping

            class ScheduleThread(threading.Thread):
                @classmethod
                def run(cls):
                    # while not cease_continuous_run.is_set(): # Needs proper event handling for stop
                    while True: # Simplified for now; consider a proper stop mechanism
                        try:
                            self.scheduler.run_pending()
                            time.sleep(interval)
                        except Exception as e:
                            logger.error(f"Scheduler thread runtime error: {e}", exc_info=True)
                            # Consider more robust error handling

            continuous_thread = ScheduleThread()
            continuous_thread.daemon = True
            continuous_thread.start()
            logger.info("Scheduler started. Press Ctrl+C to stop if running in foreground.")

        # This was creating an extra thread, simplified above.
        # scheduler_thread = threading.Thread(target=run_continuously, daemon=True)
        # scheduler_thread.start()
        # logger.info("Scheduler thread started.")
        run_continuously() # Start directly if not meant to be a separate controller thread for the scheduler thread itself

    def stop_scheduler(self):
        """Stops the scheduler. (Placeholder - actual stop depends on run_continuously implementation)."""
        logger.info("Stopping scheduler (Placeholder, actual stop depends on run_continuously implementation)")
        # If using threading.Event:
        # if hasattr(self, 'cease_continuous_run'):
        # self.cease_continuous_run.set()
        # For now, this is a conceptual stop. Schedule library itself doesn't have a direct stop method for its loop.
