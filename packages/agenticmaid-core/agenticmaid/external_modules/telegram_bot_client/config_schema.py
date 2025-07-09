"""
Telegram Bot客户端配置模式
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class TelegramBotConfig:
    """Telegram Bot配置"""
    
    # 基本配置
    token: str
    mode: str = "polling"  # polling 或 webhook
    
    # Polling配置
    polling_interval: float = 1.0
    polling_timeout: int = 30
    drop_pending_updates: bool = True
    allowed_updates: List[str] = None
    
    # Webhook配置
    webhook_url: str = ""
    webhook_port: int = 8443
    webhook_listen: str = "0.0.0.0"
    webhook_cert_path: str = ""
    webhook_key_path: str = ""
    
    # 安全配置
    allowed_users: List[str] = None
    allowed_chats: List[str] = None
    
    # 速率限制
    rate_limit_messages_per_minute: int = 20
    rate_limit_commands_per_hour: int = 100
    
    # 消息格式
    use_markdown: bool = True
    include_emojis: bool = True
    max_message_length: int = 4096
    split_long_messages: bool = True
    
    # 错误处理
    max_retries: int = 3
    retry_delay: float = 5.0
    
    def __post_init__(self):
        if self.allowed_updates is None:
            self.allowed_updates = ["message", "callback_query"]
        if self.allowed_users is None:
            self.allowed_users = []
        if self.allowed_chats is None:
            self.allowed_chats = []
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TelegramBotConfig":
        """从字典创建配置"""
        return cls(
            token=config_dict["token"],
            mode=config_dict.get("mode", "polling"),
            polling_interval=config_dict.get("polling_interval", 1.0),
            polling_timeout=config_dict.get("polling_timeout", 30),
            drop_pending_updates=config_dict.get("drop_pending_updates", True),
            allowed_updates=config_dict.get("allowed_updates", ["message", "callback_query"]),
            webhook_url=config_dict.get("webhook_url", ""),
            webhook_port=config_dict.get("webhook_port", 8443),
            webhook_listen=config_dict.get("webhook_listen", "0.0.0.0"),
            webhook_cert_path=config_dict.get("webhook_cert_path", ""),
            webhook_key_path=config_dict.get("webhook_key_path", ""),
            allowed_users=config_dict.get("allowed_users", []),
            allowed_chats=config_dict.get("allowed_chats", []),
            rate_limit_messages_per_minute=config_dict.get("rate_limit_messages_per_minute", 20),
            rate_limit_commands_per_hour=config_dict.get("rate_limit_commands_per_hour", 100),
            use_markdown=config_dict.get("use_markdown", True),
            include_emojis=config_dict.get("include_emojis", True),
            max_message_length=config_dict.get("max_message_length", 4096),
            split_long_messages=config_dict.get("split_long_messages", True),
            max_retries=config_dict.get("max_retries", 3),
            retry_delay=config_dict.get("retry_delay", 5.0)
        )
