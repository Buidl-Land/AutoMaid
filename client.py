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
from conversation_logger import get_conversation_logger
from mcp_stdio_adapter import get_stdio_manager
from functools import wraps
import inspect

logger = logging.getLogger(__name__)

class DispatchInput(BaseModel):
    agent_id: str = Field(description="The unique ID of the agent to invoke (must be defined in the config).")
    prompt: str = Field(description="The prompt or question to send to the invoked agent.")
    mode: str = Field(default="synchronous", description="The invocation mode ('synchronous' or 'concurrent').")

class AgenticMaid:
    def __init__(self, config_path_or_dict, enable_conversation_logging=True):
        """
        Initializes the AgenticMaid.
        Note: Call `await client.async_initialize()` after creating an instance
        to complete asynchronous setup like fetching MCP tools.

        Args:
            config_path_or_dict (str or dict): Path to a JSON configuration file
                                               or a Python dictionary containing the configuration.
            enable_conversation_logging (bool): Whether to enable conversation logging to files
        """
        self.config = None
        self.config_path_or_dict = config_path_or_dict
        self.ai_services = {}
        self.mcp_client = None
        self.mcp_tools = []
        self.mcp_sessions = {}  # Stores persistent sessions
        self.stdio_manager = None  # stdio tool manager
        self.agents = {}  # Stores agents
        self.scheduler = schedule
        self.scheduler_thread = None
        self.scheduler_stop_event = None

        # Initialize conversation logger
        self.enable_conversation_logging = enable_conversation_logging
        if enable_conversation_logging:
            self.conversation_logger = get_conversation_logger()
            self.conversation_logger.log_system_event("agenticmaid_init", {
                "config_source": str(config_path_or_dict),
                "logging_enabled": True
            })
        else:
            self.conversation_logger = None

        # Initialize stdio tool manager
        self.stdio_manager = get_stdio_manager()
        if self.conversation_logger:
            self.stdio_manager.set_conversation_logger(self.conversation_logger)

        logger.info("AgenticMaid instance created.")

    def _wrap_tool_with_logging(self, tool, task_name="current_task"):
        """Wrap a tool to add conversation logging using monkey patching"""
        if not self.conversation_logger:
            return tool

        # Store original methods
        original_run = tool.run
        original_arun = tool.arun

        def logged_run(*args, **kwargs):
            """Wrapper for synchronous run method"""
            try:
                # Log tool call start
                self.conversation_logger.log_tool_call(
                    task_name=task_name,
                    tool_name=tool.name,
                    tool_args={"args": args, "kwargs": kwargs},
                    tool_result=None,
                    status="started"
                )

                # Execute the original tool
                result = original_run(*args, **kwargs)

                # Log successful completion
                self.conversation_logger.log_tool_call(
                    task_name=task_name,
                    tool_name=tool.name,
                    tool_args={"args": args, "kwargs": kwargs},
                    tool_result=result,
                    status="success"
                )

                return result

            except Exception as e:
                # Log error
                self.conversation_logger.log_tool_call(
                    task_name=task_name,
                    tool_name=tool.name,
                    tool_args={"args": args, "kwargs": kwargs},
                    tool_result=str(e),
                    status="error"
                )
                raise

        async def logged_arun(*args, **kwargs):
            """Wrapper for asynchronous arun method"""
            try:
                # Log tool call start
                self.conversation_logger.log_tool_call(
                    task_name=task_name,
                    tool_name=tool.name,
                    tool_args={"args": args, "kwargs": kwargs},
                    tool_result=None,
                    status="started"
                )

                # Execute the original async tool
                result = await original_arun(*args, **kwargs)

                # Log successful completion
                self.conversation_logger.log_tool_call(
                    task_name=task_name,
                    tool_name=tool.name,
                    tool_args={"args": args, "kwargs": kwargs},
                    tool_result=result,
                    status="success"
                )

                return result

            except Exception as e:
                # Log error
                self.conversation_logger.log_tool_call(
                    task_name=task_name,
                    tool_name=tool.name,
                    tool_args={"args": args, "kwargs": kwargs},
                    tool_result=str(e),
                    status="error"
                )
                raise

        # Monkey patch the tool's methods
        tool.run = logged_run
        tool.arun = logged_arun

        return tool

    def _create_langchain_stdio_tools(self):
        """Convert stdio tools to langchain tools"""
        from langchain_core.tools import Tool
        import asyncio

        stdio_langchain_tools = []
        for tool_name, stdio_tool in self.stdio_manager.tools.items():

            def create_tool_func(name):
                def tool_func(*args, **kwargs):
                    """Wrapper function for stdio tool"""
                    # Handle both positional and keyword arguments
                    if args:
                        # If there are positional arguments, treat the first one as input
                        if isinstance(args[0], str):
                            # If it's a string, try to parse as JSON or use as URL
                            try:
                                import json
                                kwargs.update(json.loads(args[0]))
                            except (json.JSONDecodeError, TypeError):
                                # If not JSON, assume it's a URL parameter
                                kwargs['url'] = args[0]
                        elif isinstance(args[0], dict):
                            kwargs.update(args[0])

                    # Get current event loop or create new one
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is running, we need to run in a thread
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, self._run_stdio_tool(name, kwargs))
                                result = future.result()
                        else:
                            result = loop.run_until_complete(self._run_stdio_tool(name, kwargs))
                    except RuntimeError:
                        # No event loop, create one
                        result = asyncio.run(self._run_stdio_tool(name, kwargs))

                    if result.get("success"):
                        # Return the most relevant part of the result
                        if "parsed_output" in result:
                            return str(result["parsed_output"])
                        elif "stdout" in result and result["stdout"].strip():
                            return result["stdout"]
                        else:
                            return str(result)
                    else:
                        raise Exception(f"Tool {name} failed: {result.get('error', 'Unknown error')}")

                return tool_func

            # Create langchain tool
            langchain_tool = Tool(
                name=tool_name,
                description=stdio_tool.description,
                func=create_tool_func(tool_name)
            )

            stdio_langchain_tools.append(langchain_tool)

        # Add stdio tools to mcp_tools list
        self.mcp_tools.extend(stdio_langchain_tools)
        logger.info(f"Converted {len(stdio_langchain_tools)} stdio tools to langchain tools")

    async def _run_stdio_tool(self, tool_name: str, kwargs: dict):
        """Run stdio tool asynchronously"""
        return await self.stdio_manager.call_tool("current_task", tool_name, kwargs)

    async def async_initialize(self, reconfiguring=False):
        """
        Performs asynchronous initialization tasks, primarily initializing MCP services
        and fetching tools. This should be called after the client is constructed or
        when reconfiguring.

        Args:
            reconfiguring (bool): If True, indicates that this is part of a reconfiguration.
        """
        # Load configuration if not already loaded
        if not self.config:
            env_base_config = self._load_env_config()
            main_config = self._load_main_config()

            if main_config is None:
                logger.error("Main configuration could not be loaded. AgenticMaid initialization failed.")
                return False
            else:
                self.config = self._merge_configs(env_base_config, main_config)

            if self.config:
                self._schedule_tasks()
            else:
                logger.warning("AgenticMaid not fully initialized due to configuration errors.")
                return False

        if reconfiguring:
            # Clean up existing MCP sessions
            if hasattr(self, 'mcp_sessions') and self.mcp_sessions:
                await self.cleanup_mcp_sessions()

            self.mcp_client = None
            self.mcp_tools = []
            self.mcp_sessions = {}
            self.agents = {}
            self.ai_services = {}
            self.scheduler = schedule

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
        self.config_path_or_dict = new_config_path_or_dict

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
        if isinstance(self.config_path_or_dict, str):
            try:
                with open(self.config_path_or_dict, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded successfully from {self.config_path_or_dict}")
                return config
            except FileNotFoundError:
                logger.error(f"Configuration file {self.config_path_or_dict} not found.")
                return None
            except json.JSONDecodeError:
                logger.error(f"Configuration file {self.config_path_or_dict} is not valid JSON.")
                return None
            except Exception as e:
                logger.error(f"Error loading configuration from {self.config_path_or_dict}: {e}")
                return None
        elif isinstance(self.config_path_or_dict, dict):
            logger.info("Configuration loaded successfully from dictionary.")
            return copy.deepcopy(self.config_path_or_dict)
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

                # Important: Use persistent sessions instead of short connection mode
                logger.info("Creating persistent MCP sessions...")
                self.mcp_sessions = {}
                self.mcp_tools = []

                # Create persistent sessions for each MCP server
                for server_name in mcp_server_configs.keys():
                    try:
                        logger.info(f"Creating persistent session for server: {server_name}")
                        # Create and store persistent session
                        session_context = self.mcp_client.session(server_name)
                        session = await session_context.__aenter__()
                        self.mcp_sessions[server_name] = {
                            'session': session,
                            'context': session_context
                        }

                        # Load tools from persistent session
                        from langchain_mcp_adapters.tools import load_mcp_tools
                        server_tools = await load_mcp_tools(session)

                        # Store tools without wrapping to avoid Pydantic issues
                        # We'll wrap them later when creating the agent
                        self.mcp_tools.extend(server_tools)
                        logger.info(f"Loaded {len(server_tools)} tools from {server_name}")

                    except Exception as e:
                        logger.error(f"Failed to create persistent session for {server_name}: {e}")
                        continue

                logger.info(f"Successfully created persistent sessions and loaded {len(self.mcp_tools)} MCP tools total")

            except Exception as e:
                logger.error(f"Error initializing MultiServerMCPClient: {e}", exc_info=True)
                self.mcp_client = None
                self.mcp_tools = []
                self.mcp_sessions = {}

        ai_config = self.config.get("ai_services", {})
        if not ai_config:
            logger.warning("No 'ai_services' found in configuration. LLM interactions might fail.")

        self.ai_services = ai_config
        for service_name, service_details in self.ai_services.items():
            logger.info(f"AI Service '{service_name}' configured with model: {service_details.get('model')}")

        # Initialize stdio tools
        stdio_tools_config = self.config.get("stdio_tools", [])
        if stdio_tools_config:
            try:
                self.stdio_manager.register_tools_from_config(stdio_tools_config)
                stdio_tools = self.stdio_manager.list_tools()
                logger.info(f"Successfully loaded {len(stdio_tools)} stdio tools")

                # Convert stdio tools to langchain tools
                self._create_langchain_stdio_tools()

            except Exception as e:
                logger.error(f"Error loading stdio tools: {e}", exc_info=True)

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
            return {"status": "error", "task_name": task_name, "error": "No prompt provided"}

        agent_id = task_details.get("agent_id")
        model_config_name = task_details.get("model_config_name")

        if not agent_id and not model_config_name:
            logger.error(f"Task '{task_name}' needs 'agent_id' or 'model_config_name' to run. Skipping.")
            return {"status": "error", "task_name": task_name, "error": "No agent_id or model_config_name provided"}

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
                return {"status": "error", "task_name": task_name, "error": "No LLM configuration found"}

            agent = await self._get_or_create_agent(agent_key, llm_config_name, calling_agent_id=agent_id)
            if not agent:
                logger.error(f"Could not get or create agent for task '{task_name}'. Skipping.")
                return {"status": "error", "task_name": task_name, "error": "Could not create agent"}

            logger.info(f"Invoking agent for task '{task_name}' with prompt: '{prompt[:100]}...'")

            # Log conversation start
            if self.conversation_logger:
                self.conversation_logger.log_conversation_start(task_name, prompt)

            # Add retry mechanism for LLM server errors
            max_retries = 3
            retry_delay = 5  # seconds

            for attempt in range(max_retries):
                try:
                    response = await agent.ainvoke({"messages": messages})
                    break  # Success, exit retry loop
                except Exception as e:
                    error_str = str(e)
                    if "502" in error_str or "Bad Gateway" in error_str:
                        if attempt < max_retries - 1:
                            logger.warning(f"LLM server error (attempt {attempt + 1}/{max_retries}): {e}")
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            logger.error(f"LLM server failed after {max_retries} attempts: {e}")
                            raise
                    else:
                        # Non-502 error, re-raise immediately
                        raise

            if response and "messages" in response:
                messages = response["messages"]
                if messages:
                    final_message = messages[-1]
                    if hasattr(final_message, 'content'):
                        result_content = final_message.content
                    else:
                        result_content = str(final_message)

                    logger.info(f"Task '{task_name}' completed successfully.")

                    # Log AI response
                    if self.conversation_logger:
                        self.conversation_logger.log_ai_response(task_name, result_content, "success")

                    return {
                        "status": "success",
                        "task_name": task_name,
                        "response": result_content
                    }
                else:
                    logger.warning(f"Task '{task_name}' returned empty messages.")
                    return {
                        "status": "success",
                        "task_name": task_name,
                        "response": "Task completed but no content returned."
                    }
            else:
                logger.warning(f"Task '{task_name}' returned unexpected response format.")
                return {
                    "status": "success",
                    "task_name": task_name,
                    "response": str(response) if response else "No response received."
                }
        except Exception as e:
            logger.error(f"Error executing task '{task_name}': {e}", exc_info=True)

            # Log error
            if self.conversation_logger:
                self.conversation_logger.log_error(task_name, str(e), "execution_error")

            return {"status": "error", "task_name": task_name, "error": str(e)}

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
        allowed_list = dispatch_config.get("allowed_invocations", {})
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
            logger.info(f"Agent '{agent_key}' successfully invoked. Raw response: {response}")
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

        # Wrap MCP tools with logging (do this here to avoid Pydantic issues during loading)
        if self.conversation_logger and agent_tools:
            wrapped_tools = []
            for tool in agent_tools:
                try:
                    wrapped_tool = self._wrap_tool_with_logging(tool, task_name=agent_key)
                    wrapped_tools.append(wrapped_tool)
                except Exception as e:
                    logger.warning(f"Failed to wrap tool {tool.name} with logging: {e}")
                    wrapped_tools.append(tool)  # Use original tool if wrapping fails
            agent_tools = wrapped_tools

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
            # Create model instance based on provider configuration
            provider = llm_config.get("provider", "openai").lower()
            if provider == "openai":
                from langchain_openai import ChatOpenAI

                # Prepare model parameters
                model_params = {
                    "model": model_name_or_instance,
                    "api_key": llm_config.get("api_key"),
                    "base_url": llm_config.get("base_url"),
                    "temperature": llm_config.get("temperature", 0.7)
                }

                # Add max_tokens if specified
                if "max_tokens" in llm_config:
                    model_params["max_tokens"] = llm_config["max_tokens"]

                # Add max_completion_tokens if specified (for newer OpenAI models)
                if "max_completion_tokens" in llm_config:
                    model_params["max_completion_tokens"] = llm_config["max_completion_tokens"]

                model_instance = ChatOpenAI(**model_params)
            else:
                # Fallback to automatic detection for other providers
                model_instance = model_name_or_instance

            agent_executor = create_react_agent(model_instance, agent_tools)
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

    async def cleanup_mcp_sessions(self):
        """Clean up MCP persistent sessions"""
        if hasattr(self, 'mcp_sessions') and self.mcp_sessions:
            logger.info("Cleaning up MCP persistent sessions...")

            # Simply clear the sessions without attempting complex cleanup
            # This avoids asyncio task conflicts during shutdown
            session_count = len(self.mcp_sessions)
            self.mcp_sessions = {}

            logger.info(f"Cleared {session_count} MCP sessions (graceful shutdown)")

            # Also clear the MCP client if it exists
            if hasattr(self, 'mcp_client'):
                self.mcp_client = None
                logger.info("MCP client cleared")

    def save_conversation_logs(self):
        """Save conversation records to local files"""
        if self.conversation_logger:
            return self.conversation_logger.cleanup_session()
        return None

    def __del__(self):
        """Destructor to ensure cleanup"""
        # Save conversation records
        if hasattr(self, 'conversation_logger') and self.conversation_logger:
            try:
                self.conversation_logger.cleanup_session()
            except Exception as e:
                logger.warning(f"Error saving conversation logs during cleanup: {e}")

        # Silently clear MCP sessions without complex async cleanup
        try:
            if hasattr(self, 'mcp_sessions'):
                self.mcp_sessions = {}
            if hasattr(self, 'mcp_client'):
                self.mcp_client = None
        except Exception:
            pass  # Ignore all errors during destruction
