"""
独立的Telegram Bot客户端模块
作为外部组件，可以独立使用或集成到messaging_system中
"""

import asyncio
import aiohttp
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Awaitable
from collections import defaultdict

from .config_schema import TelegramBotConfig


class TelegramMessage:
    """Telegram消息封装"""
    
    def __init__(self, raw_message: Dict[str, Any]):
        self.raw = raw_message
        self.message_id = raw_message["message_id"]
        self.chat_id = str(raw_message["chat"]["id"])
        self.user_id = str(raw_message["from"]["id"])
        self.text = raw_message.get("text", "")
        self.timestamp = datetime.fromtimestamp(raw_message["date"])
        self.is_command = self.text.startswith("/")
        
        # 用户信息
        self.user_info = {
            "id": self.user_id,
            "username": raw_message["from"].get("username"),
            "first_name": raw_message["from"].get("first_name"),
            "last_name": raw_message["from"].get("last_name")
        }
        
        # 聊天信息
        self.chat_info = {
            "id": self.chat_id,
            "type": raw_message["chat"]["type"],
            "title": raw_message["chat"].get("title")
        }


class TelegramBotClient:
    """独立的Telegram Bot客户端"""
    
    def __init__(self, config: TelegramBotConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.token[:10]}")
        
        # API配置
        self.api_url = f"https://api.telegram.org/bot{config.token}"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 状态管理
        self.running = False
        self.last_update_id = 0
        
        # 消息处理器
        self.message_handlers: List[Callable[[TelegramMessage], Awaitable[None]]] = []
        self.command_handlers: Dict[str, Callable[[TelegramMessage], Awaitable[None]]] = {}
        
        # 速率限制
        self.rate_limiter = defaultdict(list)
        
        self.logger.info(f"初始化Telegram Bot客户端: {config.token[:10]}...")
    
    def add_message_handler(self, handler: Callable[[TelegramMessage], Awaitable[None]]):
        """添加消息处理器"""
        self.message_handlers.append(handler)
    
    def add_command_handler(self, command: str, handler: Callable[[TelegramMessage], Awaitable[None]]):
        """添加命令处理器"""
        self.command_handlers[command] = handler
    
    async def start(self):
        """启动客户端"""
        try:
            self.logger.info("启动Telegram Bot客户端")
            
            # 创建HTTP会话
            self.session = aiohttp.ClientSession()
            
            # 验证bot token
            if not await self._verify_bot_token():
                raise Exception("Bot token验证失败")
            
            # 设置polling模式
            await self._setup_polling_mode()
            
            # 开始运行
            self.running = True
            
            if self.config.mode == "polling":
                await self._polling_loop()
            else:
                raise NotImplementedError("Webhook模式暂未实现")
        
        except Exception as e:
            self.logger.error(f"启动失败: {e}")
            raise
    
    async def stop(self):
        """停止客户端"""
        self.logger.info("停止Telegram Bot客户端")
        self.running = False
        
        if self.session:
            await self.session.close()
            self.session = None
    
    async def send_message(self, chat_id: str, text: str, parse_mode: str = None) -> bool:
        """发送消息"""
        try:
            # 检查速率限制
            if not self._check_rate_limit(chat_id):
                self.logger.warning(f"速率限制: {chat_id}")
                return False
            
            # 处理长消息
            if len(text) > self.config.max_message_length:
                if self.config.split_long_messages:
                    return await self._send_long_message(chat_id, text, parse_mode)
                else:
                    text = text[:self.config.max_message_length - 3] + "..."
            
            # 发送消息
            url = f"{self.api_url}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": text
            }
            
            if parse_mode or self.config.use_markdown:
                data["parse_mode"] = parse_mode or "Markdown"
            
            async with self.session.post(url, json=data) as response:
                result = await response.json()
                
                if result.get("ok"):
                    self.logger.debug(f"消息发送成功: {chat_id}")
                    return True
                else:
                    self.logger.error(f"发送消息失败: {result.get('description')}")
                    return False
        
        except Exception as e:
            self.logger.error(f"发送消息异常: {e}")
            return False
    
    async def _verify_bot_token(self) -> bool:
        """验证bot token"""
        try:
            url = f"{self.api_url}/getMe"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("ok"):
                        bot_info = result["result"]
                        self.logger.info(f"Bot验证成功: @{bot_info.get('username')} ({bot_info.get('id')})")
                        return True
                    else:
                        self.logger.error(f"Bot token无效: {result.get('description')}")
                        return False
                else:
                    self.logger.error(f"验证请求失败: {response.status}")
                    return False
        
        except Exception as e:
            self.logger.error(f"验证bot token异常: {e}")
            return False
    
    async def _setup_polling_mode(self):
        """设置polling模式"""
        try:
            # 删除可能存在的webhook
            url = f"{self.api_url}/deleteWebhook"
            data = {"drop_pending_updates": self.config.drop_pending_updates}
            
            async with self.session.post(url, json=data) as response:
                result = await response.json()
                if result.get("ok"):
                    self.logger.info("Webhook已清理，polling模式已设置")
                else:
                    self.logger.warning(f"清理webhook警告: {result.get('description')}")
        
        except Exception as e:
            self.logger.error(f"设置polling模式失败: {e}")
    
    async def _polling_loop(self):
        """Polling循环"""
        self.logger.info("开始polling循环")
        
        while self.running:
            try:
                updates = await self._get_updates()
                
                for update in updates:
                    self.last_update_id = max(self.last_update_id, update["update_id"])
                    
                    if "message" in update:
                        message = TelegramMessage(update["message"])
                        await self._process_message(message)
                
                await asyncio.sleep(self.config.polling_interval)
            
            except Exception as e:
                self.logger.error(f"Polling循环错误: {e}")
                await asyncio.sleep(self.config.retry_delay)
        
        self.logger.info("Polling循环结束")
    
    async def _get_updates(self) -> List[Dict[str, Any]]:
        """获取更新"""
        try:
            url = f"{self.api_url}/getUpdates"
            params = {
                "offset": self.last_update_id + 1,
                "timeout": self.config.polling_timeout,
                "allowed_updates": self.config.allowed_updates
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("ok"):
                        return result.get("result", [])
                    else:
                        error_desc = result.get("description", "")
                        if "terminated by other getUpdates request" in error_desc:
                            self.logger.warning("检测到getUpdates冲突，等待重试...")
                            await asyncio.sleep(5)
                            return []
                        else:
                            self.logger.error(f"获取更新失败: {error_desc}")
                            return []
                else:
                    self.logger.error(f"HTTP错误: {response.status}")
                    return []
        
        except Exception as e:
            self.logger.error(f"获取更新异常: {e}")
            return []
    
    async def _process_message(self, message: TelegramMessage):
        """处理消息"""
        try:
            # 检查安全限制
            if not self._check_security(message):
                return
            
            self.logger.info(f"处理消息: {message.text[:50]}... (用户: {message.user_id})")
            
            # 处理命令
            if message.is_command:
                command = message.text.split()[0]
                if command in self.command_handlers:
                    await self.command_handlers[command](message)
                    return
            
            # 处理普通消息
            for handler in self.message_handlers:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"消息处理器错误: {e}")
        
        except Exception as e:
            self.logger.error(f"处理消息失败: {e}")
    
    def _check_security(self, message: TelegramMessage) -> bool:
        """检查安全限制"""
        # 检查允许的用户
        if self.config.allowed_users and message.user_id not in self.config.allowed_users:
            self.logger.debug(f"用户未授权: {message.user_id}")
            return False
        
        # 检查允许的聊天
        if self.config.allowed_chats and message.chat_id not in self.config.allowed_chats:
            self.logger.debug(f"聊天未授权: {message.chat_id}")
            return False
        
        return True
    
    def _check_rate_limit(self, chat_id: str) -> bool:
        """检查速率限制"""
        now = time.time()
        
        # 清理过期记录
        self.rate_limiter[chat_id] = [
            timestamp for timestamp in self.rate_limiter[chat_id]
            if now - timestamp < 60  # 保留最近1分钟的记录
        ]
        
        # 检查速率限制
        if len(self.rate_limiter[chat_id]) >= self.config.rate_limit_messages_per_minute:
            return False
        
        # 记录当前时间
        self.rate_limiter[chat_id].append(now)
        return True
    
    async def _send_long_message(self, chat_id: str, text: str, parse_mode: str = None) -> bool:
        """发送长消息（分割）"""
        max_length = self.config.max_message_length
        success = True
        
        for i in range(0, len(text), max_length):
            chunk = text[i:i + max_length]
            if not await self.send_message(chat_id, chunk, parse_mode):
                success = False
            await asyncio.sleep(0.1)  # 避免发送过快
        
        return success
