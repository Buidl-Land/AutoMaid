import json
import schedule
import time
import threading
import os
import copy
import asyncio
import logging
from dotenv import dotenv_values
from pydantic import BaseModel, Field
from langchain_core.tools import Tool

from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

class DispatchInput(BaseModel):
    agent_id: str = Field(description="The unique ID of the agent to invoke (must be defined in the config).")
    prompt: str = Field(description="The prompt or question to send to the invoked agent.")
    mode: str = Field(default="synchronous", description="The invocation mode ('synchronous' or 'concurrent').")

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
        self.mcp_client: MultiServerMCPClient | None = None
        self.mcp_tools: list = []
        self.agents: dict = {}

        env_base_config = self._load_env_config()
        main_config = self._load_main_config()

        if main_config is None:
            logger.error("Main configuration could not be loaded. AgenticMaid initialization failed.")
        else:
            self.config = self._merge_configs(env_base_config, main_config)

        self.ai_services = {}
        self.scheduler = schedule.Scheduler()

        if self.config:
            self._schedule_tasks()
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
            return False

        if reconfiguring:
            self.mcp_client: MultiServerMCPClient | None = None
            self.mcp_tools: list = []
            self.agents: dict = {}
            self.ai_services = {}
            self.scheduler = schedule.Scheduler()

        await self._initialize_services()
        self._schedule_tasks()
        logger.info(f"Async initialization {'(reconfiguration)' if reconfiguring else ''} complete.")
        return True

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
            self.config = None
            return False

        self.config = self._merge_configs(env_base_config, main_config)

        if not self.config:
            logger.error("Configuration merging failed during reconfiguration.")
            return False

        return await self.async_initialize(reconfiguring=True)

    def _load_env_config(self):
        """Loads configuration from .env file."""
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
            logger.info("Configuration loaded successfully from dictionary.")
            return copy.deepcopy(self.config_source)
        else:
            logger.error("Invalid configuration source. Must be a file path (str) or a dictionary.")
            return None

    def _merge_configs(self, env_config, main_config):
        """
        Merges environment configuration (defaults) with the main configuration.
        Main configuration values take precedence.
        """
        if not main_config:
            return env_config

        merged_config = copy.deepcopy(main_config)

        if "ai_services" in merged_config and isinstance(merged_config["ai_services"], dict):
            for service_name, service_details in merged_config["ai_services"].items():
                if not isinstance(service_details, dict):
                    logger.warning(f"Service '{service_name}' details are not a dictionary. Skipping .env merge for it.")
                    continue

                provider = service_details.get("provider", "").upper()

                if "api_key" not in service_details or not service_details["api_key"]:
                    env_api_key = env_config.get(f"{provider}_API_KEY") if provider else None
                    if not env_api_key:
                        env_api_key = env_config.get("DEFAULT_API_KEY")
                    if env_api_key:
                        service_details["api_key"] = env_api_key
                        logger.info(f"Using API key from .env for AI service '{service_name}'.")
                    else:
                        logger.warning(f"API key for AI service '{service_name}' not found in main config or .env.")

                if "model" not in service_details or not service_details["model"]:
                    env_model = env_config.get(f"{provider}_DEFAULT_MODEL") if provider else None
                    if not env_model:
                        env_model = env_config.get("DEFAULT_MODEL_NAME")
                    if env_model:
                        service_details["model"] = env_model
                        logger.info(f"Using model from .env for AI service '{service_name}'.")

                if "base_url" not in service_details or not service_details["base_url"]:
                    env_base_url = env_config.get(f"{provider}_BASE_URL") if provider else None
                    if not env_base_url:
                        env_base_url = env_config.get("DEFAULT_BASE_URL")
                    if env_base_url:
                         service_details["base_url"] = env_base_url
                         logger.info(f"Using base_url from .env for AI service '{service_name}'.")

        logger.info("Configuration merged successfully.")
        return merged_config

    async def _initialize_services(self):
        """Initializes MCP services and prepares AI service configurations."""
        if not self.config:
            logger.error("Configuration not loaded, cannot initialize services.")
            return

        mcp_server_configs = self.config.get("mcp_servers", {})
        if not mcp_server_configs:
            logger.warning("No 'mcp_servers' found in configuration. MCP tools will not be available.")
        else:
            try:
                logger.info(f"Initializing MultiServerMCPClient with servers: {mcp_server_configs}")
                self.mcp_client = MultiServerMCPClient(mcp_server_configs)
                logger.info("Fetching MCP tools...")
                self.mcp_tools = await self.mcp_client.get_tools()
                logger.info(f"Successfully fetched {len(self.mcp_tools)} MCP tools: {[tool.name for tool in self.mcp_tools]}")
            except Exception as e:
                logger.error(f"Error initializing MultiServerMCPClient or fetching tools: {e}", exc_info=True)
                self.mcp_client = None
                self.mcp_tools = []

        ai_config = self.config.get("ai_services", {})
        if not ai_config:
            logger.warning("No 'ai_services' found in configuration. LLM interactions might fail.")

        self.ai_services = ai_config
        for service_name, service_details in self.ai_services.items():
            logger.info(f"AI Service '{service_name}' configured with model: {service_details.get('model')}")

    def _schedule_tasks(self):
        """Schedules tasks based on cron expressions in the configuration."""
        if not self.config:
            return

        tasks = self.config.get("scheduled_tasks", [])

        def _execute_task_wrapper(task_details_sync):
            """Synchronous wrapper to run the async _execute_task."""
            try:
                asyncio.run(self._execute_task(task_details_sync))
            except RuntimeError as e:
                if " asyncio.run() cannot be called from a running event loop" in str(e):
                    logger.warning(f"Could not run async task '{task_details_sync.get('name')}' via asyncio.run() from current context: {e}")
                    logger.warning("Consider using a different scheduling approach if AgenticMaid is run within an existing asyncio loop.")
                else:
                    logger.error(f"Runtime error executing task '{task_details_sync.get('name')}': {e}", exc_info=True)
            except Exception as e:
                logger.error(f"General error executing task '{task_details_sync.get('name')}': {e}", exc_info=True)

        for task in tasks:
            cron_expr = task.get("cron_expression")
            if cron_expr and task.get("enabled", True):
                try:
                    if "daily at" in cron_expr.lower():
                        time_str = cron_expr.lower().split("daily at")[1].strip()
                        self.scheduler.every().day.at(time_str).do(_execute_task_wrapper, task_details_sync=task)
                        logger.info(f"Task '{task.get('name', 'Unnamed Task')}' scheduled: {cron_expr}")
                    else:
                        logger.warning(f"Cannot parse cron expression '{cron_expr}' for task '{task.get('name', 'Unnamed Task')}'.")
                except Exception as e:
                    logger.error(f"Error scheduling task '{task.get('name', 'Unnamed Task')}': {e}", exc_info=True)
            elif not task.get("enabled", True):
                logger.info(f"Task '{task.get('name', 'Unnamed Task')}' is disabled and will not be scheduled.")

    async def _execute_task(self, task_details):
        """Executes a scheduled task."""
        task_name = task_details.get('name', 'Unnamed Task')
        logger.info(f"Executing scheduled task: {task_name}")

        prompt = task_details.get("prompt")
        if not prompt:
            logger.error(f"Task '{task_name}' has no prompt. Skipping.")
            return

        agent_id = task_details.get("agent_id")
        model_config_name = task_details.get("model_config_name")

        if not agent_id and not model_config_name:
            logger.error(f"Task '{task_name}' needs 'agent_id' or 'model_config_name' to run. Skipping.")
            return

        try:
            messages = []
            agent_config = self.config.get("agents", {}).get(agent_id, {})

            if agent_config.get("system_prompt"):
                messages.append({"role": "system", "content": agent_config["system_prompt"]})
            if agent_config.get("role_prompt"):
                messages.append({"role": "user", "content": agent_config["role_prompt"]})

            messages.append({"role": "user", "content": prompt})

            agent_key = agent_id or model_config_name
            llm_config_name = model_config_name
            if agent_id and agent_id in self.config.get("agents", {}):
                 llm_config_name = self.config["agents"][agent_id].get("model_config_name", model_config_name)

            if not llm_config_name:
                llm_config_name = self.config.get("default_llm_service_name")

            if not llm_config_name:
                logger.error(f"No LLM configuration specified or found for task '{task_name}'. Skipping.")
                return

            agent = await self._get_or_create_agent(agent_key, llm_config_name, calling_agent_id=agent_id)
            if not agent:
                logger.error(f"Could not get or create agent for task '{task_name}'. Skipping.")
                return

            logger.info(f"Invoking agent for task '{task_name}' with prompt: '{prompt}'")
            response = await agent.ainvoke({"messages": messages})
            logger.info(f"Task '{task_name}' response: {response}")
        except Exception as e:
            logger.error(f"Error executing task '{task_name}': {e}", exc_info=True)
            return {"status": "error", "task_name": task_name, "error": str(e)}
        return {"status": "success", "task_name": task_name, "response": response}

    async def async_run_scheduled_task_by_name(self, task_name_to_run: str):
        """Finds a scheduled task by its name and executes it immediately."""
        if not self.config or not self.config.get("scheduled_tasks"):
            return {"status": "error", "message": f"No scheduled tasks configured. Cannot run task '{task_name_to_run}'."}

        task_details_to_run = None
        for task in self.config.get("scheduled_tasks", []):
            if task.get("name") == task_name_to_run:
                task_details_to_run = task
                break

        if not task_details_to_run:
            return {"status": "error", "message": f"Scheduled task '{task_name_to_run}' not found."}

        if not task_details_to_run.get("enabled", True):
            return {"status": "skipped", "message": f"Task '{task_name_to_run}' is disabled."}

        logger.info(f"Manually triggering task: {task_name_to_run}")
        return await self._execute_task(task_details_to_run)

    async def async_run_all_enabled_scheduled_tasks(self):
        """Executes all enabled scheduled tasks immediately."""
        if not self.config or not self.config.get("scheduled_tasks"):
            return [{"status": "error", "message": "No scheduled tasks configured."}]

        results = []
        enabled_tasks = [task for task in self.config.get("scheduled_tasks", []) if task.get("enabled", True)]

        if not enabled_tasks:
            return [{"status": "skipped", "message": "No enabled scheduled tasks found."}]

        logger.info(f"Manually triggering all {len(enabled_tasks)} enabled scheduled tasks.")
        for task_details in enabled_tasks:
            result = await self._execute_task(task_details)
            results.append(result)

        return results

    async def _dispatch_agent(self, calling_agent_id: str, target_agent_id: str, prompt: str, mode: str) -> dict:
        """
        Handles the logic of one agent invoking another.

        This internal method checks if the dispatch feature is enabled and if the
        calling agent has permission to invoke the target agent based on the
        'allowed_invocations' map in the configuration.

        Args:
            calling_agent_id (str): The ID of the agent initiating the call.
            target_agent_id (str): The ID of the agent to be invoked.
            prompt (str): The prompt to pass to the target agent.
            mode (str): The invocation mode ('sync' or 'concurrent').

        Returns:
            dict: A dictionary containing the status of the operation and the response.
        """
        dispatch_config = self.config.get("multi_agent_dispatch", {})
        if not dispatch_config.get("enabled"):
            return {"status": "error", "message": "Multi-agent dispatch is disabled."}

        # Check the allowlist to see if the calling agent can invoke the target.
        # An agent is allowed if its ID is in the target's list or if the list contains a wildcard "*".
        allowed_list = dispatch_config.get("allowed_invocations", {}).get(calling_agent_id)
        if allowed_list is None or (target_agent_id not in allowed_list and "*" not in allowed_list):
            return {"status": "error", "message": f"Agent '{calling_agent_id}' is not allowed to invoke agent '{target_agent_id}'."}

        agent_config = self.config.get("agents", {}).get(target_agent_id)
        if not agent_config:
             return {"status": "error", "message": f"Target agent '{target_agent_id}' not found in configuration."}

        messages = [{"role": "user", "content": prompt}]
        llm_service_name = agent_config.get("model_config_name")

        if not llm_service_name:
             return {"status": "error", "message": f"No 'model_config_name' for target agent '{target_agent_id}'."}

        response = await self.run_mcp_interaction(messages, llm_service_name, agent_key=target_agent_id, calling_agent_id=target_agent_id, agent_config=agent_config)
        return {"status": "success", "response": response}

    async def run_mcp_interaction(self, messages: list, llm_service_name: str, agent_key: str = "default_agent", calling_agent_id: str = None, agent_config: dict = None):
        """
        Runs an interaction with an agent.
        """
        if not self.config:
            logger.error("Client not properly configured.")
            return None

        agent = await self._get_or_create_agent(agent_key, llm_service_name, calling_agent_id=calling_agent_id, agent_config=agent_config)
        if not agent:
            return {"error": f"Failed to get or create agent '{agent_key}' with LLM '{llm_service_name}'."}

        logger.info(f"Invoking agent '{agent_key}' (LLM: {llm_service_name}) with messages: {messages}")
        try:
            response = await agent.ainvoke({"messages": messages})
            return response
        except Exception as e:
            logger.error(f"Error during agent invocation for '{agent_key}': {e}", exc_info=True)
            return {"error": str(e)}

    async def _get_or_create_agent(self, agent_key: str, llm_service_name: str, calling_agent_id: str = None, agent_config: dict = None):
        """
        Retrieves an existing agent or creates a new one.
        """
        if agent_key in self.agents:
            logger.info(f"Returning existing agent: {agent_key}")
            return self.agents[agent_key]

        llm_config = self.ai_services.get(llm_service_name)
        if not llm_config:
            logger.error(f"LLM service configuration '{llm_service_name}' not found.")
            return None

        model_name_or_instance = llm_config.get("model")
        if not model_name_or_instance:
            logger.error(f"'model' not specified in LLM service config '{llm_service_name}'.")
            return None

        effective_agent_id = calling_agent_id or agent_key
        agent_tools = self.mcp_tools[:]
        dispatch_config = self.config.get("multi_agent_dispatch", {})
        allowed_invocations = dispatch_config.get("allowed_invocations", {})

        # If the multi-agent dispatch feature is enabled and the current agent
        # is listed in the 'allowed_invocations' configuration, create and add
        # a special 'dispatch' tool to this agent's available tools.
        if dispatch_config.get("enabled") and effective_agent_id in allowed_invocations:
            async def dispatch_wrapper(agent_id: str, prompt: str, mode: str = "synchronous") -> str:
                """A wrapper for the _dispatch_agent method to be used as a tool."""
                result = await self._dispatch_agent(
                    calling_agent_id=effective_agent_id,
                    target_agent_id=agent_id,
                    prompt=prompt,
                    mode=mode
                )
                return json.dumps(result)

            dispatch_tool = Tool(
                name="dispatch",
                func=dispatch_wrapper,
                description="Invokes another agent. Use this to delegate tasks. Input must be a JSON object with 'agent_id', 'prompt', and optional 'mode' ('synchronous' or 'concurrent').",
                args_schema=DispatchInput
            )
            agent_tools.append(dispatch_tool)

        logger.info(f"Creating new ReAct agent '{agent_key}' with LLM '{model_name_or_instance}' and {len(agent_tools)} tools.")
        try:
            agent_executor = create_react_agent(model_name_or_instance, agent_tools)
            self.agents[agent_key] = agent_executor
            logger.info(f"Agent '{agent_key}' created successfully.")
            return agent_executor
        except Exception as e:
            logger.error(f"Error creating ReAct agent '{agent_key}': {e}", exc_info=True)
            return None

    async def handle_chat_message(self, service_id: str, messages: list, stream: bool = False):
        """Handles an incoming chat message for a configured chat service."""
        if not self.config or "chat_services" not in self.config:
            return {"error": "Chat services are not configured."}

        chat_service_config = None
        for service in self.config["chat_services"]:
            if service.get("service_id") == service_id:
                chat_service_config = service
                break

        if not chat_service_config:
            return {"error": f"Chat service with ID '{service_id}' not found."}

        llm_service_name = chat_service_config.get("llm_service_name")
        if not llm_service_name:
            return {"error": f"Chat service '{service_id}' does not specify 'llm_service_name'."}

        agent_key = f"chat_agent_{service_id}"
        response = await self.run_mcp_interaction(messages, llm_service_name, agent_key, calling_agent_id=service_id, agent_config=chat_service_config)

        if stream:
            logger.info(f"Streaming for service '{service_id}' (placeholder). Response: {response}")
            return {"warning": "Streaming not fully implemented", "response": response}
        else:
            return response

    def start_scheduler(self):
        """Starts the scheduler thread to run scheduled tasks in the background."""
        if not self.scheduler.jobs:
            logger.info("No scheduled tasks to run.")
            return

        def run_continuously(interval=1):
            class ScheduleThread(threading.Thread):
                @classmethod
                def run(cls):
                    while True:
                        try:
                            self.scheduler.run_pending()
                            time.sleep(interval)
                        except Exception as e:
                            logger.error(f"Scheduler thread runtime error: {e}", exc_info=True)

            continuous_thread = ScheduleThread()
            continuous_thread.daemon = True
            continuous_thread.start()
            logger.info("Scheduler started. Press Ctrl+C to stop if running in foreground.")

        run_continuously()

    def stop_scheduler(self):
        """Stops the scheduler. (Placeholder)."""
        logger.info("Stopping scheduler (Placeholder)")
