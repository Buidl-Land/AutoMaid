import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading

class ConversationLogger:
    """
    Conversation and Log Manager
    Automatically saves program logs and AI conversation records to local files
    """

    def __init__(self, log_dir: str = "logs", enable_file_logging: bool = True):
        """
        Initializes the conversation logger

        Args:
            log_dir: Directory for saving log files
            enable_file_logging: Whether to enable file logging
        """
        self.log_dir = log_dir
        self.enable_file_logging = enable_file_logging
        self.conversation_history = []
        self.session_start_time = datetime.now()
        self.lock = threading.Lock()

        # Create log directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Set up file logger
        if self.enable_file_logging:
            self._setup_file_logging()

    def _setup_file_logging(self):
        """Set up file logging"""
        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"agenticmaid_{timestamp}.log"
        log_filepath = os.path.join(self.log_dir, log_filename)

        # Create file handler
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Set log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)

        self.log_filepath = log_filepath
        logging.info(f"File logging enabled, log file: {log_filepath}")

    def log_conversation_start(self, task_name: str, prompt: str):
        """Log conversation start with full prompt"""
        with self.lock:
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "conversation_start",
                "task_name": task_name,
                "prompt": prompt,  # Save full prompt
                "prompt_length": len(prompt)
            }
            self.conversation_history.append(conversation_entry)
            logging.info(f"Task started: {task_name}")

    def log_ai_response(self, task_name: str, response: str, status: str = "success"):
        """Log AI response with full content"""
        with self.lock:
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "ai_response",
                "task_name": task_name,
                "status": status,
                "response": response,  # Save full response
                "response_length": len(response)
            }
            self.conversation_history.append(conversation_entry)
            logging.info(f"Task '{task_name}' completed with status: {status}")

    def log_error(self, task_name: str, error: str, error_type: str = "general"):
        """Log error with full details"""
        with self.lock:
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "error",
                "task_name": task_name,
                "error_type": error_type,
                "error": error
            }
            self.conversation_history.append(conversation_entry)
            logging.error(f"Task '{task_name}' failed with error ({error_type}): {error}")

    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log system event with full details"""
        with self.lock:
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "system_event",
                "event_type": event_type,
                "details": details
            }
            self.conversation_history.append(conversation_entry)
            logging.info(f"System event: {event_type}")

    def log_tool_call(self, task_name: str, tool_name: str, tool_args: Dict[str, Any], tool_result: Any, status: str = "success"):
        """Log tool call with complete details"""
        with self.lock:
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "tool_call",
                "task_name": task_name,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": str(tool_result) if tool_result is not None else None,
                "status": status,
                "result_length": len(str(tool_result)) if tool_result is not None else 0
            }
            self.conversation_history.append(conversation_entry)
            logging.info(f"Tool call: {tool_name} for task '{task_name}' - Status: {status}")

    def save_conversation_history(self, filename: Optional[str] = None) -> str:
        """Save complete conversation history to JSON file"""
        if filename is None:
            timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_history_{timestamp}.json"

        filepath = os.path.join(self.log_dir, filename)

        with self.lock:
            conversation_data = {
                "session_info": {
                    "start_time": self.session_start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_conversations": len(self.conversation_history)
                },
                "conversations": self.conversation_history
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Conversation history saved to: {filepath}")
        return filepath

    def save_session_summary(self, filename: Optional[str] = None) -> str:
        """Save complete session summary to Markdown file"""
        if filename is None:
            timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"session_summary_{timestamp}.md"

        filepath = os.path.join(self.log_dir, filename)

        with self.lock:
            # Statistics
            total_conversations = len(self.conversation_history)
            successful_tasks = len([c for c in self.conversation_history
                                  if c.get("type") == "ai_response" and c.get("status") == "success"])
            failed_tasks = len([c for c in self.conversation_history
                              if c.get("type") == "error"])
            tool_calls = len([c for c in self.conversation_history
                            if c.get("type") == "tool_call"])
            successful_tool_calls = len([c for c in self.conversation_history
                                       if c.get("type") == "tool_call" and c.get("status") == "success"])

            # Generate Markdown content
            content = f"""# AgenticMaid Session Summary

**Session Time**: {self.session_start_time.strftime("%Y-%m-%d %H:%M:%S")} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Statistics
- Total Conversations: {total_conversations}
- Successful Tasks: {successful_tasks}
- Failed Tasks: {failed_tasks}
- Tool Calls: {tool_calls}
- Successful Tool Calls: {successful_tool_calls}

## Conversation Records

"""

            for i, conv in enumerate(self.conversation_history, 1):
                timestamp = datetime.fromisoformat(conv["timestamp"]).strftime("%H:%M:%S")

                if conv["type"] == "conversation_start":
                    content += f"### {i}. Task Started - {conv['task_name']} ({timestamp})\n\n"
                    content += f"**Prompt**: \n```\n{conv['prompt']}\n```\n\n"
                    content += f"*Prompt length: {conv.get('prompt_length', len(conv['prompt']))} characters*\n\n"

                elif conv["type"] == "ai_response":
                    status_emoji = "✅" if conv["status"] == "success" else "❌"
                    content += f"### {i}. AI Response - {conv['task_name']} ({timestamp}) {status_emoji}\n\n"
                    content += f"**Status**: {conv['status']}\n\n"
                    content += f"**Response Content**:\n```\n{conv['response']}\n```\n\n"
                    content += f"*Response length: {conv.get('response_length', len(conv['response']))} characters*\n\n"

                elif conv["type"] == "error":
                    content += f"### {i}. Error - {conv['task_name']} ({timestamp}) ❌\n\n"
                    content += f"**Error Type**: {conv['error_type']}\n\n"
                    content += f"**Error Details**: {conv['error']}\n\n"

                elif conv["type"] == "tool_call":
                    status_emoji = "✅" if conv["status"] == "success" else "❌"
                    content += f"### {i}. Tool Call - {conv['tool_name']} ({timestamp}) {status_emoji}\n\n"
                    content += f"**Task**: {conv['task_name']}\n\n"
                    content += f"**Status**: {conv['status']}\n\n"
                    content += f"**Arguments**: \n```json\n{json.dumps(conv['tool_args'], ensure_ascii=False, indent=2)}\n```\n\n"
                    if conv.get('tool_result'):
                        content += f"**Result**: \n```\n{conv['tool_result']}\n```\n\n"
                    content += f"*Result length: {conv.get('result_length', 0)} characters*\n\n"

                elif conv["type"] == "system_event":
                    content += f"### {i}. System Event - {conv['event_type']} ({timestamp}) ℹ️\n\n"
                    content += f"**Details**: \n```json\n{json.dumps(conv['details'], ensure_ascii=False, indent=2)}\n```\n\n"

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        logging.info(f"Session summary saved to: {filepath}")
        return filepath

    def get_log_files_info(self) -> Dict[str, str]:
        """Get log file information"""
        info = {}
        if hasattr(self, 'log_filepath'):
            info['log_file'] = self.log_filepath
        return info

    def cleanup_session(self):
        """Clean up session and save all records"""
        logging.info("Saving session records...")

        # Save conversation history
        json_file = self.save_conversation_history()

        # Save session summary
        md_file = self.save_session_summary()

        # Get log files info
        files_info = self.get_log_files_info()

        print(f"\n=== Session Records Saved ===")
        print(f"Conversation History (JSON): {json_file}")
        print(f"Session Summary (Markdown): {md_file}")
        if 'log_file' in files_info:
            print(f"System Log: {files_info['log_file']}")
        print(f"Log Directory: {os.path.abspath(self.log_dir)}")

        return {
            'conversation_history': json_file,
            'session_summary': md_file,
            **files_info
        }

# Global conversation logger instance
_global_conversation_logger = None

def get_conversation_logger() -> ConversationLogger:
    """Get global conversation logger instance"""
    global _global_conversation_logger
    if _global_conversation_logger is None:
        _global_conversation_logger = ConversationLogger()
    return _global_conversation_logger

def init_conversation_logger(log_dir: str = "logs", enable_file_logging: bool = True) -> ConversationLogger:
    """Initialize global conversation logger"""
    global _global_conversation_logger
    _global_conversation_logger = ConversationLogger(log_dir, enable_file_logging)
    return _global_conversation_logger
