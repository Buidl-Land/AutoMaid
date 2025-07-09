from fastapi import FastAPI, HTTPException, Body, Path as FastApiPath
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Union, Optional
import logging
import os

from .client import AgenticMaid # Import from package

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

# File Handler for API logs
log_file_path = os.path.join(LOG_DIR, "api.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO) # Or logging.DEBUG for more verbose file logs
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__) # Get a logger for this specific module

app = FastAPI(
    title="AgenticMaid API",
    description="API for interacting with the AgenticMaid.",
    version="0.1.0"
)

# --- Global AgenticMaid Instance ---
# Strategy: Single global instance, initialized at startup (potentially with a default config)
# and reconfigurable via an API endpoint.
mcp_client_instance: Optional[AgenticMaid] = None
DEFAULT_CONFIG_PATH = "AgenticMaid/config.json" # Default config to try at startup

async def get_mcp_client() -> AgenticMaid:
    """Dependency to get the initialized AgenticMaid instance."""
    if mcp_client_instance is None:
        logger.error("AgenticMaid instance is not initialized.")
        raise HTTPException(status_code=503, detail="AgenticMaid is not initialized. Please call /client/init first.")
    return mcp_client_instance

@app.on_event("startup")
async def startup_event():
    global mcp_client_instance
    config_to_load: Union[str, Dict] = {} # Start with empty config if default not found

    if os.path.exists(DEFAULT_CONFIG_PATH):
        logger.info(f"Found default config at {DEFAULT_CONFIG_PATH}, attempting to load.")
        config_to_load = DEFAULT_CONFIG_PATH
    else:
        logger.info(f"Default config {DEFAULT_CONFIG_PATH} not found. AgenticMaid will initialize with empty/default internal config.")
        # AgenticMaid constructor handles dict, so an empty dict means minimal default setup.
        # It will print warnings if critical parts are missing.

    try:
        logger.info(f"Initializing AgenticMaid with: {config_to_load if isinstance(config_to_load, str) else 'empty dictionary'}")
        mcp_client_instance = AgenticMaid(config_path_or_dict=config_to_load)
        initialization_success = await mcp_client_instance.async_initialize()
        if initialization_success:
            logger.info("AgenticMaid initialized successfully at startup.")
        else:
            logger.warning("AgenticMaid initialization at startup encountered issues (e.g. config errors). Check logs. API might be limited until re-init.")
            # mcp_client_instance might still be an object, but not fully functional.
            # The /client/init endpoint can be used to fix this.
    except Exception as e:
        logger.error(f"Failed to initialize AgenticMaid at startup: {e}", exc_info=True)
        mcp_client_instance = None # Ensure it's None if startup fails catastrophically

# --- Pydantic Models ---
class ClientInitRequest(BaseModel):
    config: Union[Dict[str, Any], str] = Field(
        ...,
        description="Full JSON configuration as a dictionary or a path to a JSON configuration file accessible by the server."
    )

class ClientInitResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    service_id: str = Field(..., description="ID of the chat service to use (from AgenticMaid config)")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    # stream: bool = Field(default=False, description="Whether to stream the response (not fully supported yet)")

class ChatResponse(BaseModel):
    # Define based on the expected structure from AgenticMaid's handle_chat_message
    # This could be complex, for now, a generic Any.
    # Example: response: Dict[str, Any] or specific Pydantic model for agent output
    data: Any
    error: Optional[str] = None


class TaskRunResponse(BaseModel):
    status: str
    message: Optional[str] = None
    task_name: Optional[str] = None
    details: Optional[Any] = None # Could be the actual task result or error info

class MultiTaskRunResponse(BaseModel):
    results: List[TaskRunResponse]


# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
async def read_root():
    logger.info("Health check endpoint was called.")
    return {"status": "ok", "message": "AgenticMaid API is running."}

@app.get("/health", tags=["Health Check"])
async def health_check():
    logger.info("Health check endpoint (/health) was called.")
    return {"status": "ok", "message": "AgenticMaid API is healthy."}

@app.post("/client/init", response_model=ClientInitResponse, tags=["Client Management"])
async def init_client(payload: ClientInitRequest = Body(...)):
    """
    Initializes or reconfigures the AgenticMaid instance.
    Accepts a full configuration dictionary or a path to a configuration file.
    """
    global mcp_client_instance
    logger.info(f"Received request to initialize/reconfigure client. Config type: {'path' if isinstance(payload.config, str) else 'dictionary'}")

    try:
        if mcp_client_instance is None: # First time init via API or if startup failed
            logger.info("AgenticMaid instance is None, creating new instance for init.")
            mcp_client_instance = AgenticMaid(config_path_or_dict=payload.config)
            success = await mcp_client_instance.async_initialize()
        else: # Reconfiguring existing instance
            logger.info("Reconfiguring existing AgenticMaid instance.")
            success = await mcp_client_instance.async_reconfigure(new_config_path_or_dict=payload.config)

        if success:
            logger.info("AgenticMaid (re)configured successfully via API.")
            return ClientInitResponse(status="success", message="AgenticMaid (re)configured successfully.")
        else:
            logger.error("AgenticMaid (re)configuration via API failed. Check client logs for details.")
            # Attempt to provide some detail if config is available and has issues.
            error_details = {"reason": "Configuration or initialization failed. See server logs."}
            if mcp_client_instance and mcp_client_instance.config is None: # Config loading itself failed
                 error_details = {"reason": "Failed to load or merge the provided configuration."}

            raise HTTPException(status_code=400, detail={"message": "AgenticMaid (re)configuration failed.", "details": error_details})

    except Exception as e:
        logger.error(f"Error during /client/init: {e}", exc_info=True)
        # Ensure mcp_client_instance is None if re/init fails badly, to force re-init attempt
        mcp_client_instance = None
        raise HTTPException(status_code=500, detail=f"Internal server error during client initialization: {str(e)}")


@app.post("/client/chat", response_model=ChatResponse, tags=["Client Actions"])
async def client_chat(request: ChatRequest = Body(...)):
    """
    Processes a chat message using the configured AgenticMaid.
    """
    client = await get_mcp_client() # Ensures client is initialized

    # Convert Pydantic ChatMessage to dicts for AgenticMaid
    messages_dict_list = []
    chat_service_config = None
    for service in client.config.get("chat_services", []):
        if service.get("service_id") == request.service_id:
            chat_service_config = service
            break

    if chat_service_config:
        if chat_service_config.get("system_prompt"):
            messages_dict_list.append({"role": "system", "content": chat_service_config["system_prompt"]})
        if chat_service_config.get("role_prompt"):
            messages_dict_list.append({"role": "user", "content": chat_service_config["role_prompt"]})

    messages_dict_list.extend([msg.model_dump() for msg in request.messages])

    logger.info(f"Received chat request for service_id: {request.service_id}")
    try:
        # Assuming handle_chat_message is async and exists on AgenticMaid
        response_data = await client.handle_chat_message(
            service_id=request.service_id,
            messages=messages_dict_list
            # stream=request.stream # Stream parameter not used yet in this simplified version
        )

        if isinstance(response_data, dict) and "error" in response_data:
            logger.error(f"Chat request failed for service '{request.service_id}': {response_data['error']}")
            raise HTTPException(status_code=400, detail=response_data['error'])

        return ChatResponse(data=response_data)
    except Exception as e:
        logger.error(f"Error during /client/chat for service '{request.service_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during chat processing: {str(e)}")


@app.post("/client/run_task/{task_name}", response_model=TaskRunResponse, tags=["Client Actions"])
async def run_specific_task(task_name: str = FastApiPath(..., description="The name of the scheduled task to run.")):
    """
    Triggers a specific scheduled task by its name.
    """
    client = await get_mcp_client()
    logger.info(f"Received request to run task: {task_name}")
    try:
        result = await client.async_run_scheduled_task_by_name(task_name)

        if result.get("status") == "error":
            logger.error(f"Failed to run task '{task_name}': {result.get('message')}")
            raise HTTPException(status_code=400, detail=result)
        if result.get("status") == "skipped":
            logger.info(f"Task '{task_name}' was skipped: {result.get('message')}")
            # Return 200 OK but indicate skipped
            return TaskRunResponse(status="skipped", message=result.get("message"), task_name=task_name)

        return TaskRunResponse(
            status="success",
            message=f"Task '{task_name}' executed.",
            task_name=task_name,
            details=result # The 'response' from _execute_task or other info
        )
    except Exception as e:
        logger.error(f"Error during /client/run_task/{task_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error running task '{task_name}': {str(e)}")


@app.post("/client/run_all_scheduled_tasks", response_model=MultiTaskRunResponse, tags=["Client Actions"])
async def run_all_tasks():
    """
    Triggers all enabled scheduled tasks.
    """
    client = await get_mcp_client()
    logger.info("Received request to run all enabled scheduled tasks.")
    try:
        results = await client.async_run_all_enabled_scheduled_tasks()
        # Convert internal AgenticMaid task results to TaskRunResponse models
        api_results = []
        for res in results:
            status = res.get("status", "unknown")
            message = res.get("message")
            task_name = res.get("task_name")
            details = res # Pass the whole dict as details for now

            if status == "error":
                 # Log error but include in response list as an error entry
                logger.error(f"Error in one of the tasks during run_all_scheduled_tasks (task: {task_name}): {message or details}")

            api_results.append(TaskRunResponse(
                status=status,
                message=message,
                task_name=task_name,
                details=details
            ))
        return MultiTaskRunResponse(results=api_results)
    except Exception as e:
        logger.error(f"Error during /client/run_all_scheduled_tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error running all tasks: {str(e)}")


def main():
    """Main entry point for the AgenticMaid API server."""
    import uvicorn
    uvicorn.run(
        "agenticmaid.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()


# To run this application manually:
# uvicorn agenticmaid.api:app --reload
# Or use the entry point: agenticmaid-api