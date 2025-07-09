import json
import subprocess
import asyncio
import logging
from typing import Dict, Any, List, Optional
import os
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)

class MCPStdioAdapter:
    """
    MCP adapter that supports stdio communication with non-Python tools
    Enables calling external executables, scripts, and command-line tools as MCP tools
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stdio MCP adapter
        
        Args:
            config: Configuration dictionary containing:
                - name: Tool name
                - command: Command to execute
                - description: Tool description
                - input_schema: JSON schema for input validation
                - timeout: Execution timeout in seconds (default: 30)
                - working_dir: Working directory for command execution
                - env_vars: Environment variables to set
        """
        self.name = config.get("name", "stdio_tool")
        self.command = config.get("command", "")
        self.description = config.get("description", "External stdio tool")
        self.input_schema = config.get("input_schema", {})
        self.timeout = config.get("timeout", 30)
        self.working_dir = config.get("working_dir", os.getcwd())
        self.env_vars = config.get("env_vars", {})
        self.conversation_logger = None
        
        # Validate configuration
        if not self.command:
            raise ValueError("Command is required for stdio MCP adapter")
    
    def set_conversation_logger(self, logger):
        """Set conversation logger for tool call recording"""
        self.conversation_logger = logger
    
    async def call_tool(self, task_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the external tool via stdio
        
        Args:
            task_name: Name of the task calling this tool
            args: Arguments to pass to the tool
            
        Returns:
            Dictionary containing tool execution result
        """
        start_time = datetime.now()
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(self.env_vars)
            
            # Prepare input data
            input_data = json.dumps(args, ensure_ascii=False)
            
            # Log tool call start
            if self.conversation_logger:
                self.conversation_logger.log_tool_call(
                    task_name=task_name,
                    tool_name=self.name,
                    tool_args=args,
                    tool_result=None,
                    status="started"
                )
            
            logger.info(f"Executing stdio tool '{self.name}' for task '{task_name}'")
            logger.debug(f"Command: {self.command}")
            logger.debug(f"Input: {input_data}")
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
                env=env
            )
            
            # Send input and get output
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=input_data.encode('utf-8')),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Tool '{self.name}' execution timed out after {self.timeout} seconds")
            
            # Process results
            return_code = process.returncode
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                "success": return_code == 0,
                "return_code": return_code,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "execution_time": execution_time
            }
            
            # Try to parse stdout as JSON if possible
            if stdout_text.strip():
                try:
                    parsed_output = json.loads(stdout_text)
                    result["parsed_output"] = parsed_output
                except json.JSONDecodeError:
                    # Keep as plain text if not valid JSON
                    pass
            
            # Log tool call completion
            if self.conversation_logger:
                status = "success" if return_code == 0 else "error"
                self.conversation_logger.log_tool_call(
                    task_name=task_name,
                    tool_name=self.name,
                    tool_args=args,
                    tool_result=result,
                    status=status
                )
            
            if return_code == 0:
                logger.info(f"Tool '{self.name}' completed successfully in {execution_time:.2f}s")
            else:
                logger.warning(f"Tool '{self.name}' failed with return code {return_code}")
                logger.warning(f"stderr: {stderr_text}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_result = {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
            
            # Log tool call error
            if self.conversation_logger:
                self.conversation_logger.log_tool_call(
                    task_name=task_name,
                    tool_name=self.name,
                    tool_args=args,
                    tool_result=error_result,
                    status="error"
                )
            
            logger.error(f"Error executing tool '{self.name}': {e}")
            return error_result
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "type": "stdio"
        }

class MCPStdioManager:
    """
    Manager for multiple stdio MCP tools
    """
    
    def __init__(self):
        self.tools: Dict[str, MCPStdioAdapter] = {}
        self.conversation_logger = None
    
    def set_conversation_logger(self, logger):
        """Set conversation logger for all tools"""
        self.conversation_logger = logger
        for tool in self.tools.values():
            tool.set_conversation_logger(logger)
    
    def register_tool(self, config: Dict[str, Any]) -> MCPStdioAdapter:
        """Register a new stdio tool"""
        tool = MCPStdioAdapter(config)
        if self.conversation_logger:
            tool.set_conversation_logger(self.conversation_logger)
        
        self.tools[tool.name] = tool
        logger.info(f"Registered stdio tool: {tool.name}")
        return tool
    
    def register_tools_from_config(self, tools_config: List[Dict[str, Any]]):
        """Register multiple tools from configuration"""
        for tool_config in tools_config:
            try:
                self.register_tool(tool_config)
            except Exception as e:
                logger.error(f"Failed to register stdio tool: {e}")
    
    def get_tool(self, name: str) -> Optional[MCPStdioAdapter]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools"""
        return [tool.get_tool_info() for tool in self.tools.values()]
    
    async def call_tool(self, task_name: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        return await tool.call_tool(task_name, args)

# Global stdio manager instance
_global_stdio_manager = None

def get_stdio_manager() -> MCPStdioManager:
    """Get global stdio manager instance"""
    global _global_stdio_manager
    if _global_stdio_manager is None:
        _global_stdio_manager = MCPStdioManager()
    return _global_stdio_manager

def init_stdio_manager(tools_config: List[Dict[str, Any]] = None) -> MCPStdioManager:
    """Initialize global stdio manager with tools"""
    global _global_stdio_manager
    _global_stdio_manager = MCPStdioManager()
    
    if tools_config:
        _global_stdio_manager.register_tools_from_config(tools_config)
    
    return _global_stdio_manager
