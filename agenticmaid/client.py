import json
import schedule
import time
import threading
import os
import copy
import asyncio
import logging
import re
from dotenv import dotenv_values
from pydantic import BaseModel, Field
from langchain_core.tools import Tool
from croniter import croniter

from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from .conversation_logger import get_conversation_logger
from functools import wraps
import inspect
from .simple_memory import SimpleMemory
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class DispatchInput(BaseModel):
    agent_id: str = Field(description="The unique ID of the agent to invoke (must be defined in the config).")
    prompt: str = Field(description="The prompt or question to send to the invoked agent.")
    mode: str = Field(default="synchronous", description="The invocation mode ('synchronous' or 'concurrent').")

class AgenticMaid:
    def __init__(self, config_path_or_dict, enable_conversation_logging=True, debug=False):
        """
        Initializes the AgenticMaid.
        Note: Call `await client.async_initialize()` after creating an instance
        to complete asynchronous setup like fetching MCP tools.

        Args:
            config_path_or_dict (str or dict): Path to a JSON configuration file
                                               or a Python dictionary containing the configuration.
            enable_conversation_logging (bool): Whether to enable conversation logging to files.
            debug (bool): If True, enables verbose debug logging for network requests.
        """
        self.debug_mode = debug
        if self.debug_mode:
            logging.getLogger("httpx").setLevel(logging.DEBUG)
            logging.getLogger("openai").setLevel(logging.DEBUG)
            logger.info("Debug mode enabled. HTTPX and OpenAI will log detailed network requests.")

        self.config_path_or_dict = config_path_or_dict
        self.config = self._load_and_merge_config()
        self.ai_services = {}
        self.mcp_client = None
        self.mcp_tools = []
        self.mcp_sessions = {}  # Stores persistent sessions
        self.simple_memory = None # For simple key-value storage
        self.agents = {}  # Stores agents
        self.agent_creation_locks = {} # Locks for creating agents
        self.llm_semaphores = {} # Semaphores to limit concurrent requests to LLMs
        self.background_tasks = {} # Stores background tasks from concurrent dispatches
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

        if self.config:
            self._schedule_tasks()
        else:
            logger.warning("AgenticMaid not fully initialized due to configuration errors.")

        logger.info("AgenticMaid instance created.")

    async def _parse_and_execute_tool_calls(self, response_text: str, available_tools: list, task_name: str = "current_task"):
        """Parse tool calls from model response and execute them"""
        tool_call_pattern = r'<tool_use>(.*?)</tool_use>'
        matches = re.findall(tool_call_pattern, response_text, re.DOTALL)

        results = []
        for tool_block in matches:
            tool_name = "unknown"
            try:
                # Extract tool name and arguments from the XML block
                tool_name_match = re.search(r'<name>(.*?)</name>', tool_block, re.DOTALL)
                arguments_match = re.search(r'<arguments>(.*?)</arguments>', tool_block, re.DOTALL)

                if not tool_name_match:
                    error_msg = "Tool call error: missing <name> tag in tool_use block"
                    results.append(error_msg)
                    if self.conversation_logger:
                        self.conversation_logger.log_tool_call(task_name, "unknown", {}, error_msg, "error")
                    continue

                tool_name = tool_name_match.group(1).strip()

                params = {}
                if arguments_match:
                    params_str = arguments_match.group(1).strip()
                    if params_str:
                        try:
                            # The arguments are a JSON string
                            params = json.loads(params_str)
                        except json.JSONDecodeError:
                            error_msg = f"Tool '{tool_name}' error: Invalid JSON in <arguments>: {params_str}"
                            results.append(error_msg)
                            if self.conversation_logger:
                                self.conversation_logger.log_tool_call(task_name, tool_name, {}, error_msg, "error")
                            continue

                # Find the tool
                tool = None
                for t in available_tools:
                    if t.name == tool_name:
                        tool = t
                        break

                if not tool:
                    # Fallback matching logic
                    for t in available_tools:
                        if t.name.replace('_', '-') == tool_name.replace('_', '-') or \
                           t.name.replace('-', '_') == tool_name.replace('-', '_') or \
                           t.name.lower() == tool_name.lower() or \
                           t.name.replace('-', '').replace('_', '').replace('.', '') == tool_name.replace('-', '').replace('_', '').replace('.', ''):
                            tool = t
                            break

                if not tool:
                    error_msg = f"Tool '{tool_name}' not found"
                    results.append(error_msg)
                    if self.conversation_logger:
                        self.conversation_logger.log_tool_call(task_name, tool_name, params, error_msg, "error")
                    continue

                execution_params = params
                logger.info(f"Executing tool '{tool_name}' with params: {execution_params}")

                result = None
                status = "error"
                details = {}
                try:
                    # Try different invocation methods
                    if hasattr(tool, 'func') and callable(getattr(tool, 'func')):
                        if inspect.iscoroutinefunction(tool.func):
                            result = await tool.func(**execution_params)
                        else:
                            result = tool.func(**execution_params)
                    elif hasattr(tool, 'arun') and callable(getattr(tool, 'arun')):
                        result = await tool.arun(execution_params)
                    elif hasattr(tool, 'run') and callable(getattr(tool, 'run')):
                        result = tool.run(execution_params)
                    elif hasattr(tool, 'ainvoke') and callable(getattr(tool, 'ainvoke')):
                        result = await tool.ainvoke(execution_params)
                    elif hasattr(tool, 'invoke') and callable(getattr(tool, 'invoke')):
                        result = tool.invoke(execution_params)
                    else:
                        raise NotImplementedError("Tool execution method not found")

                    status = "success"
                    if hasattr(result, 'response_metadata'):
                        details = result.response_metadata
                    elif isinstance(result, dict) and 'response_metadata' in result:
                        details = result.pop('response_metadata')

                except Exception as execution_error:
                    result = f"Tool execution failed: {execution_error}"
                    logger.error(f"Error executing tool '{tool_name}': {execution_error}", exc_info=True)

                if self.conversation_logger:
                    self.conversation_logger.log_tool_call(task_name, tool_name, execution_params, result, status, details)

                results.append(f"Tool '{tool_name}' result: {result}")

            except Exception as e:
                logger.error(f"Error processing tool block for tool '{tool_name}': {e}", exc_info=True)
                error_msg = f"Tool '{tool_name}' error: {str(e)}"
                results.append(error_msg)
                if self.conversation_logger:
                    self.conversation_logger.log_tool_call(task_name, tool_name, {}, error_msg, "error")

        return results

    def _generate_system_prompt(self, tools: list, base_prompt: str) -> str:
        """Generates a system prompt including tool definitions."""
        tool_descriptions = ""
        for tool in tools:
            # Extracting schema for arguments
            args_schema = {}
            if hasattr(tool, 'args_schema') and tool.args_schema:
                # Pydantic models have a schema() method
                if hasattr(tool.args_schema, 'schema'):
                    args_schema = tool.args_schema.schema()
                # Fallback for dicts
                elif isinstance(tool.args_schema, dict):
                    args_schema = tool.args_schema

            tool_descriptions += f"<tool>\n  <name>{tool.name}</name>\n  <description>{tool.description}</description>\n  <arguments>\n    {json.dumps(args_schema, indent=2)}\n  </arguments>\n</tool>\n\n"

        # This is the new prompt structure based on the user's request
        # It combines a standard tool-use preamble with the specific prompt from the config.
        preamble = f"""In this environment you have access to a set of tools you can use to answer the user's question. You can use one tool per message, and will receive the result of that tool use in the user's response. You use tools step-by-step to accomplish a given task, with each tool use informed by the result of the previous tool use.

## Tool Use Formatting

Tool use is formatted using XML-style tags. The tool name is enclosed in opening and closing tags, and each parameter is similarly enclosed within its own set of tags. Here's the structure:

<tool_use>
  <name>{{tool_name}}</name>
  <arguments>{{json_arguments}}</arguments>
</tool_use>

The tool name should be the exact name of the tool you are using, and the arguments should be a JSON object containing the parameters required by that tool.

The user will respond with the result of the tool use, which should be formatted as follows:

<tool_use_result>
  <name>{{tool_name}}</name>
  <result>{{result}}</result>
</tool_use_result>

## Tool Use Available Tools
You only have access to these tools:
<tools>
{tool_descriptions}</tools>
"""
        # Combine the preamble with the user-provided base prompt
        system_prompt = f"{preamble}\n\n# Task\n\n{base_prompt}"
        return system_prompt


    def _wrap_tool_with_logging(self, tool, task_name="current_task"):
        """Wrap a tool to add conversation logging using monkey patching"""
        if not self.conversation_logger:
            return tool

        # Check what type of tool we're dealing with and handle accordingly
        original_run = None
        original_arun = None

        # Check for standard run/arun methods first
        if hasattr(tool, 'run') and callable(getattr(tool, 'run')):
            original_run = tool.run
        if hasattr(tool, 'arun') and callable(getattr(tool, 'arun')):
            original_arun = tool.arun

        # If we don't have run/arun methods, check for 'func' attribute (StructuredTool)
        if not original_run and hasattr(tool, 'func') and callable(getattr(tool, 'func')):
            original_run = tool.func
            # If the func is not async, we need to wrap it for async usage
            if inspect.iscoroutinefunction(tool.func):
                original_arun = tool.func
            else:
                async def arun_wrapper(*args, **kwargs):
                    return original_run(*args, **kwargs)
                original_arun = arun_wrapper

        # If we couldn't find any callable methods, skip wrapping but don't warn
        # This is normal for some tool types and the tools will still work
        if not original_run and not original_arun:
            return tool

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

        # Monkey patch the tool's methods based on what we found
        if original_run:
            if hasattr(tool, 'run'):
                tool.run = logged_run
            elif hasattr(tool, 'func'):
                tool.func = logged_run

        if original_arun:
            if hasattr(tool, 'arun'):
                tool.arun = logged_arun
            # For StructuredTool with func, we don't need to patch arun since it uses func

        return tool

    async def async_initialize(self, reconfiguring=False):
        """
        Performs asynchronous initialization tasks, primarily initializing MCP services
        and fetching tools. This should be called after the client is constructed or
        when reconfiguring.

        Args:
            reconfiguring (bool): If True, indicates that this is part of a reconfiguration.
        """
        if not self.config:
            logger.error("AgenticMaid configuration is missing. Cannot proceed with async initialization.")
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

                if not service_details.get("api_key"):
                    # Try to get the specific API key from env
                    env_api_key = env_config.get(f"{provider.upper()}_API_KEY")
                    # Fallback to OPENAI_API_KEY for backward compatibility
                    if not env_api_key:
                        env_api_key = env_config.get("OPENAI_API_KEY")

                    if env_api_key:
                        service_details["api_key"] = env_api_key
                        logger.info(f"Using API key from environment for AI service '{service_name}'.")
                    else:
                        logger.warning(f"API key for AI service '{service_name}' not found in config or environment variables.")

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

    def _load_and_merge_config(self):
        """Loads the main config and merges it with .env defaults."""
        env_base_config = self._load_env_config()
        main_config = self._load_main_config()
        if main_config is None:
            logger.error("Main configuration could not be loaded.")
            return None
        return self._merge_configs(env_base_config, main_config)

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
                        # Wrap tools with logging as they are loaded
                        if self.conversation_logger:
                            wrapped_tools = []
                            for t in server_tools:
                                try:
                                    wrapped_tools.append(self._wrap_tool_with_logging(t, task_name="mcp_tool_init"))
                                except Exception as e:
                                    logger.debug(f"Could not wrap tool {t.name} with logging: {e}")
                                    wrapped_tools.append(t)
                            self.mcp_tools.extend(wrapped_tools)
                        else:
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
            if "max_concurrent_requests" in service_details:
                max_requests = service_details["max_concurrent_requests"]
                self.llm_semaphores[service_name] = asyncio.Semaphore(max_requests)
                logger.info(f"Semaphore configured for '{service_name}' with a limit of {max_requests} concurrent requests.")

        # Initialize Simple Memory if configured
        simple_memory_config = self.config.get("simple_memory")
        if simple_memory_config:
            try:
                self.simple_memory = SimpleMemory(simple_memory_config)
                provider_name = simple_memory_config.get('provider')
                logger.info(f"Simple memory initialized with provider: {provider_name}")
            except Exception as e:
                logger.error(f"Error initializing SimpleMemory: {e}", exc_info=True)
                self.simple_memory = None

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
                    if croniter.is_valid(cron_expr):
                        # This is a valid cron expression, schedule it.
                        # Note: schedule library doesn't directly support cron syntax.
                        # This is a simplified placeholder. For full cron support,
                        # a more advanced scheduler would be needed.
                        # For this fix, we'll assume the cron is for minutes for simplicity
                        if cron_expr.startswith("*/"):
                            minutes = int(cron_expr.split(" ")[0].split("/")[1])
                            self.scheduler.every(minutes).minutes.do(_execute_task_wrapper, task_details_sync=task)
                            logger.info(f"Task '{task.get('name', 'Unnamed Task')}' scheduled to run every {minutes} minutes.")
                        else:
                             logger.warning(f"Cron expression '{cron_expr}' for task '{task.get('name', 'Unnamed Task')}' is valid but not supported by this scheduler's simple implementation.")
                    else:
                        logger.warning(f"Invalid cron expression '{cron_expr}' for task '{task.get('name', 'Unnamed Task')}'.")
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
            agent_key = agent_id or model_config_name
            llm_config_name = model_config_name
            if agent_id and agent_id in self.config.get("agents", {}):
                 llm_config_name = self.config["agents"][agent_id].get("model_config_name", model_config_name)

            if not llm_config_name:
                llm_config_name = self.config.get("default_llm_service_name")

            if not llm_config_name:
                logger.error(f"No LLM configuration specified or found for task '{task_name}'. Skipping.")
                return {"status": "error", "task_name": task_name, "error": "No LLM configuration found"}

            # This method now directly calls run_mcp_interaction, which handles the ReAct loop.
            # The prompt is passed as the initial message.
            messages = [HumanMessage(content=prompt)]
            llm_service_name = model_config_name
            if agent_id and agent_id in self.config.get("agents", {}):
                llm_service_name = self.config["agents"][agent_id].get("model_config_name", model_config_name)

            if not llm_service_name:
                llm_service_name = self.config.get("default_llm_service_name")

            if not llm_service_name:
                logger.error(f"No LLM configuration specified or found for task '{task_name}'. Skipping.")
                return {"status": "error", "task_name": task_name, "error": "No LLM configuration found"}

            # Log conversation start
            if self.conversation_logger:
                self.conversation_logger.log_conversation_start(task_name, prompt)

            final_response = await self.run_mcp_interaction(
                messages=messages,
                llm_service_name=llm_service_name,
                agent_key=agent_key,
                calling_agent_id=agent_id
            )

            if final_response and "output" in final_response:
                result_content = final_response["output"]
                logger.info(f"Task '{task_name}' completed successfully.")
                if self.conversation_logger:
                    self.conversation_logger.log_ai_response(task_name, final_response, "success")
                return {"status": "success", "task_name": task_name, "response": result_content}
            else:
                error_message = final_response.get("error", "An unknown error occurred.")
                logger.error(f"Task '{task_name}' failed with error: {error_message}")
                if self.conversation_logger:
                    self.conversation_logger.log_error(task_name, error_message, "execution_error")
                return {"status": "error", "task_name": task_name, "error": error_message}
        except Exception as e:
            logger.error(f"Error executing task '{task_name}': {e}", exc_info=True)
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

        logger.info(f"Concurrently triggering all {len(enabled_tasks)} enabled scheduled tasks.")
        tasks_to_run = [self._execute_task(task_details) for task_details in enabled_tasks]
        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        # Process results to handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"A concurrent task failed: {result}", exc_info=result)
                processed_results.append({"status": "error", "error": str(result)})
            else:
                processed_results.append(result)

        return processed_results

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

        if mode == 'concurrent':
            logger.info(f"Dispatching agent '{target_agent_id}' in concurrent mode.")
            # For concurrent mode, we start the task but don't wait for it.
            task = asyncio.create_task(self.run_mcp_interaction(messages, llm_service_name, agent_key=target_agent_id, calling_agent_id=target_agent_id, agent_config=agent_config))

            task_id = f"concurrent_task_{len(self.background_tasks) + 1}"
            self.background_tasks[task_id] = task

            # Add a callback to remove the task from the dict when it's done
            task.add_done_callback(lambda t: self.background_tasks.pop(task_id, None))

            return {"status": "submitted", "task_id": task_id, "message": f"Agent '{target_agent_id}' invoked concurrently."}
        else: # synchronous mode
            logger.info(f"Dispatching agent '{target_agent_id}' in synchronous mode.")
            response = await self.run_mcp_interaction(messages, llm_service_name, agent_key=target_agent_id, calling_agent_id=target_agent_id, agent_config=agent_config)
            return {"status": "success", "response": response}

    async def run_mcp_interaction(self, messages: list, llm_service_name: str, agent_key: str = "default_agent", calling_agent_id: str = None, agent_config: dict = None):
        """
        Runs an interaction with an agent.
        """
        if not self.config:
            logger.error("Client not properly configured.")
            return None

        model, agent_tools = await self._get_or_create_agent(agent_key, llm_service_name, calling_agent_id=calling_agent_id, agent_config=agent_config)
        if not model:
            return {"error": f"Failed to get or create agent '{agent_key}' with LLM '{llm_service_name}'."}

        base_prompt = ""
        if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
            base_prompt = messages[-1].get("content", "")
        elif messages and isinstance(messages[-1], HumanMessage):
            base_prompt = messages[-1].content

        system_prompt = self._generate_system_prompt(agent_tools, base_prompt)

        # Start with a system prompt and the initial user message
        conversation_history = [
            SystemMessage(content=system_prompt),
            messages[-1] # The user's actual prompt
        ]

        react_config = self.config.get("react", {})
        max_iterations = react_config.get("max_iterations", 10)

        for i in range(max_iterations):
            logger.info(f"ReAct Iteration {i+1}/{max_iterations} for agent '{agent_key}'")

            semaphore = self.llm_semaphores.get(llm_service_name)
            if semaphore:
                await semaphore.acquire()

            try:
                response = await model.ainvoke(conversation_history)
                response_text = response.content
                logger.debug(f"Agent '{agent_key}' raw response: {response_text}")
            finally:
                if semaphore:
                    semaphore.release()

            # Append the assistant's response (which may contain tool calls)
            conversation_history.append(response)

            tool_results = await self._parse_and_execute_tool_calls(response_text, agent_tools, task_name=agent_key)

            if not tool_results:
                logger.info(f"No tool calls detected. Agent '{agent_key}' finished.")
                return {"output": response_text}

            # Format tool results and add to history
            tool_result_message = "\n".join([f"<tool_use_result>\n<name>{r.split(':')[0].strip()}</name>\n<result>{r.split(':', 1)[1].strip()}</result>\n</tool_use_result>" for r in tool_results])
            conversation_history.append(HumanMessage(content=tool_result_message))
            logger.debug(f"Appended tool results to conversation: {tool_result_message}")

        return {"error": f"Agent '{agent_key}' exceeded max iterations ({max_iterations})."}

    async def _get_or_create_agent(self, agent_key: str, llm_service_name: str, calling_agent_id: str = None, agent_config: dict = None):
        """
        Retrieves an existing agent or creates a new one, with locking to prevent race conditions.
        """
        # Get or create a lock for the specific agent key
        if agent_key not in self.agent_creation_locks:
            self.agent_creation_locks[agent_key] = asyncio.Lock()

        lock = self.agent_creation_locks[agent_key]

        async with lock:
            if agent_key in self.agents:
                logger.info(f"Returning existing agent components: {agent_key}")
                return self.agents[agent_key]['model'], self.agents[agent_key]['tools']

            logger.info(f"Creating new agent components '{agent_key}' under lock.")
            llm_config = self.ai_services.get(llm_service_name)
        if not llm_config:
            logger.error(f"LLM service configuration '{llm_service_name}' not found.")
            return None, []

        model_name_or_instance = llm_config.get("model")
        if not model_name_or_instance:
            logger.error(f"'model' not specified in LLM service config '{llm_service_name}'.")
            return None, []

        effective_agent_id = calling_agent_id or agent_key
        agent_tools = self.mcp_tools[:]

        # Add simple memory tools if available
        if self.simple_memory:
            memory_tool_names = ["get", "set", "append", "delete"]
            for tool_name in memory_tool_names:
                tool_func = getattr(self.simple_memory, tool_name)
                # Create a Pydantic model for the arguments dynamically
                from pydantic import create_model, Field
                from typing import Any

                if tool_name in ["get", "delete"]:
                    InputModel = create_model(f"SimpleMemory{tool_name.capitalize()}Input", key=(str, Field(description="Memory key")))
                elif tool_name in ["set", "append"]:
                    InputModel = create_model(f"SimpleMemory{tool_name.capitalize()}Input",
                                            key=(str, Field(description="Memory key")),
                                            value=(Any, Field(description="Value to store/append")))
                else:
                    continue

                memory_tool = Tool(
                    name=f"simple_memory.{tool_name}",
                    func=tool_func,
                    description=tool_func.__doc__,
                    args_schema=InputModel
                )
                agent_tools.append(memory_tool)
            logger.info(f"Added {len(memory_tool_names)} simple memory tools to agent '{agent_key}'.")

        # Tools are already wrapped at load time, so no need to re-wrap here.

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

        logger.info(f"Creating new agent components '{agent_key}' with LLM '{model_name_or_instance}' and {len(agent_tools)} tools.")

        try:
            provider = llm_config.get("provider", "openai").lower()
            if provider == "openai":
                from langchain_openai import ChatOpenAI
                model_params = {
                    "model": model_name_or_instance,
                    "api_key": llm_config.get("api_key"),
                    "base_url": llm_config.get("base_url"),
                    "temperature": llm_config.get("temperature", 0.7),
                    "timeout": llm_config.get("timeout", 60),
                    "max_retries": llm_config.get("max_retries", 3),
                }
                if "max_tokens" in llm_config:
                    model_params["max_tokens"] = llm_config["max_tokens"]

                # We are not using native tool calling, so remove this
                # if llm_config.get("supports_tools", True):
                #     model_params["model_kwargs"] = {"tool_choice": "auto"}

                model_instance = ChatOpenAI(**model_params)
            else:
                model_instance = model_name_or_instance

            # Store the components instead of a pre-built agent
            self.agents[agent_key] = {
                "model": model_instance,
                "tools": agent_tools
            }
            logger.info(f"Agent components for '{agent_key}' created successfully.")
            return model_instance, agent_tools
        except Exception as e:
            logger.error(f"Error creating agent components for '{agent_key}': {e}", exc_info=True)
            return None, []

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
