"""
Solana Swap Monitor
Monitor swap transactions of specific wallets for meme coin copy trading
"""

import asyncio
import aiohttp
import json
import logging
import time
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Awaitable, Set
from collections import defaultdict

from .config_schema import SolanaSwapConfig, SwapAlert, SwapType, AlertLevel, WalletConfig, RPCNode


class SolanaSwapMonitor:
    """Solana Swap transaction monitor"""

    def __init__(self, config: SolanaSwapConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")

        # Setup logging
        self.setup_logging()

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

        # RPC node management
        self.current_node_index = 0
        self.node_failures: Dict[str, int] = defaultdict(int)
        self.available_nodes = [node for node in config.rpc_nodes if node.enabled]
        self.available_nodes.sort(key=lambda x: x.priority)  # Sort by priority

        # State management
        self.running = False
        self.last_signatures: Dict[str, Set[str]] = defaultdict(set)  # wallet address -> transaction signature set
        self.alert_cooldowns: Dict[str, float] = {}  # wallet address -> last alert time

        # Callback functions
        self.alert_handlers: List[Callable[[SwapAlert], Awaitable[None]]] = []

        self.logger.info(f"Initialized Solana Swap Monitor with {len(self.available_nodes)} RPC nodes, monitoring {len(config.wallets)} wallets")

    def setup_logging(self):
        """设置日志"""
        level = getattr(logging, self.config.log_level.upper())

        # 创建文件处理器
        file_handler = logging.FileHandler(self.config.log_file, encoding='utf-8')
        file_handler.setLevel(level)

        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.setLevel(level)

    def add_alert_handler(self, handler: Callable[[SwapAlert], Awaitable[None]]):
        """添加警报处理器"""
        self.alert_handlers.append(handler)

    async def start(self):
        """启动监控"""
        try:
            self.logger.info("启动Solana Swap监控器")

            # 创建HTTP会话
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.rpc_timeout)
            )

            # 验证RPC连接
            if not await self._test_rpc_connection():
                raise Exception("RPC连接测试失败")

            # 开始监控
            self.running = True
            await self._monitoring_loop()

        except Exception as e:
            self.logger.error(f"启动监控失败: {e}")
            raise

    async def stop(self):
        """停止监控"""
        self.logger.info("停止Solana Swap监控器")
        self.running = False

        if self.session:
            await self.session.close()
            self.session = None

    async def _test_rpc_connection(self) -> bool:
        """测试RPC连接"""
        try:
            response = await self._rpc_call("getVersion", [])
            if response and "result" in response:
                version = response["result"]["solana-core"]
                self.logger.info(f"RPC连接成功，Solana版本: {version}")
                return True
            else:
                self.logger.error("RPC连接测试失败")
                return False
        except Exception as e:
            self.logger.error(f"RPC连接测试异常: {e}")
            return False

    async def _monitoring_loop(self):
        """监控循环"""
        self.logger.info("开始监控循环")

        while self.running:
            try:
                # 监控所有启用的钱包
                tasks = []
                for wallet in self.config.wallets:
                    if wallet.enabled:
                        task = self._monitor_wallet(wallet)
                        tasks.append(task)

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # 等待下次检查
                await asyncio.sleep(self.config.polling_interval)

            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(self.config.retry_delay)

        self.logger.info("监控循环结束")

    async def _monitor_wallet(self, wallet: WalletConfig):
        """监控单个钱包"""
        try:
            # 获取钱包最新交易
            signatures = await self._get_wallet_signatures(wallet.address, limit=10)

            if not signatures:
                return

            # 检查新交易
            known_signatures = self.last_signatures[wallet.address]
            new_signatures = [sig for sig in signatures if sig not in known_signatures]

            if not new_signatures:
                return

            # 分析新交易
            for signature in new_signatures:
                try:
                    await self._analyze_transaction(wallet, signature)
                except Exception as e:
                    self.logger.error(f"分析交易失败 {signature}: {e}")

            # 更新已知交易
            self.last_signatures[wallet.address].update(new_signatures)

            # 限制缓存大小
            if len(self.last_signatures[wallet.address]) > 100:
                # 保留最新的50个
                recent_sigs = list(self.last_signatures[wallet.address])[-50:]
                self.last_signatures[wallet.address] = set(recent_sigs)

        except Exception as e:
            self.logger.error(f"监控钱包失败 {wallet.address}: {e}")

    async def _get_wallet_signatures(self, wallet_address: str, limit: int = 10) -> List[str]:
        """获取钱包交易签名"""
        try:
            response = await self._rpc_call("getSignaturesForAddress", [
                wallet_address,
                {"limit": limit}
            ])

            if response and "result" in response:
                return [item["signature"] for item in response["result"]]
            else:
                return []

        except Exception as e:
            self.logger.error(f"获取钱包交易失败 {wallet_address}: {e}")
            return []

    def _get_next_node(self) -> Optional[RPCNode]:
        """Get next available RPC node"""
        if not self.available_nodes:
            return None

        if self.config.rpc_load_balancing:
            # Round-robin load balancing
            node = self.available_nodes[self.current_node_index]
            self.current_node_index = (self.current_node_index + 1) % len(self.available_nodes)
            return node
        else:
            # Use highest priority node
            return self.available_nodes[0]

    def _mark_node_failure(self, node: RPCNode):
        """Mark node as failed"""
        self.node_failures[node.url] += 1
        self.logger.warning(f"RPC node failure: {node.name} ({node.url}) - {self.node_failures[node.url]} failures")

        # Remove node if too many failures
        if self.node_failures[node.url] >= node.max_retries:
            if node in self.available_nodes:
                self.available_nodes.remove(node)
                self.logger.error(f"Removed failed RPC node: {node.name} ({node.url})")

                # Reset current index if needed
                if self.current_node_index >= len(self.available_nodes):
                    self.current_node_index = 0

    def _reset_node_failures(self, node: RPCNode):
        """Reset node failure count on successful call"""
        if node.url in self.node_failures:
            del self.node_failures[node.url]

    async def _rpc_call(self, method: str, params: List[Any]) -> Optional[Dict[str, Any]]:
        """RPC call with multi-node support"""
        if not self.available_nodes:
            self.logger.error("No available RPC nodes")
            return None

        # Try nodes until success or all fail
        attempted_nodes = set()

        while len(attempted_nodes) < len(self.available_nodes):
            node = self._get_next_node()
            if not node or node.url in attempted_nodes:
                break

            attempted_nodes.add(node.url)

            try:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": params
                }

                timeout = aiohttp.ClientTimeout(total=node.timeout)
                async with self.session.post(node.url, json=payload, timeout=timeout) as response:
                    if response.status == 200:
                        result = await response.json()
                        self._reset_node_failures(node)
                        return result
                    else:
                        self.logger.warning(f"RPC call failed on {node.name}: HTTP {response.status}")
                        self._mark_node_failure(node)

            except asyncio.TimeoutError:
                self.logger.warning(f"RPC call timeout on {node.name}")
                self._mark_node_failure(node)
            except Exception as e:
                self.logger.warning(f"RPC call error on {node.name}: {e}")
                self._mark_node_failure(node)

        self.logger.error(f"All RPC nodes failed for method: {method}")
        return None

    async def _analyze_transaction(self, wallet: WalletConfig, signature: str):
        """分析交易"""
        try:
            # 获取交易详情
            tx_data = await self._get_transaction_details(signature)
            if not tx_data:
                return

            # 解析swap交易
            swap_info = await self._parse_swap_transaction(tx_data)
            if not swap_info:
                return

            # 检查是否符合监控条件
            if not self._should_alert(wallet, swap_info):
                return

            # 创建警报
            alert = await self._create_swap_alert(wallet, signature, swap_info, tx_data)
            if alert:
                await self._handle_alert(alert)

        except Exception as e:
            self.logger.error(f"分析交易失败 {signature}: {e}")

    async def _get_transaction_details(self, signature: str) -> Optional[Dict[str, Any]]:
        """获取交易详情"""
        try:
            response = await self._rpc_call("getTransaction", [
                signature,
                {
                    "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": 0
                }
            ])

            if response and "result" in response and response["result"]:
                return response["result"]
            else:
                return None

        except Exception as e:
            self.logger.error(f"获取交易详情失败 {signature}: {e}")
            return None

    async def _parse_swap_transaction(self, tx_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析swap交易"""
        try:
            # 检查交易是否成功
            if tx_data.get("meta", {}).get("err"):
                return None

            # 解析指令
            instructions = tx_data.get("transaction", {}).get("message", {}).get("instructions", [])

            for instruction in instructions:
                # 检查是否为swap指令
                if self._is_swap_instruction(instruction):
                    return await self._extract_swap_info(instruction, tx_data)

            return None

        except Exception as e:
            self.logger.error(f"解析swap交易失败: {e}")
            return None

    def _is_swap_instruction(self, instruction: Dict[str, Any]) -> bool:
        """检查是否为swap指令"""
        # 检查常见的DEX程序ID
        dex_programs = [
            "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",  # Raydium
            "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium V4
            "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",   # Jupiter
            "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",   # Whirlpool
        ]

        program_id = instruction.get("programId")
        return program_id in dex_programs

    async def _extract_swap_info(self, instruction: Dict[str, Any], tx_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取swap信息"""
        try:
            # 从交易的token余额变化中提取swap信息
            pre_balances = tx_data.get("meta", {}).get("preTokenBalances", [])
            post_balances = tx_data.get("meta", {}).get("postTokenBalances", [])

            # 计算余额变化
            balance_changes = self._calculate_balance_changes(pre_balances, post_balances)

            if len(balance_changes) < 2:
                return None

            # 识别输入和输出token
            token_in = None
            token_out = None
            amount_in = 0
            amount_out = 0

            for change in balance_changes:
                if change["change"] < 0:  # 减少的是输入token
                    token_in = change["mint"]
                    amount_in = abs(change["change"])
                elif change["change"] > 0:  # 增加的是输出token
                    token_out = change["mint"]
                    amount_out = change["change"]

            if not token_in or not token_out:
                return None

            # 计算SOL金额
            sol_amount = await self._calculate_sol_amount(token_in, token_out, amount_in, amount_out)

            return {
                "token_in": token_in,
                "token_out": token_out,
                "amount_in": amount_in,
                "amount_out": amount_out,
                "sol_amount": sol_amount,
                "timestamp": tx_data.get("blockTime", int(time.time()))
            }

        except Exception as e:
            self.logger.error(f"提取swap信息失败: {e}")
            return None

    def _calculate_balance_changes(self, pre_balances: List[Dict], post_balances: List[Dict]) -> List[Dict[str, Any]]:
        """计算token余额变化"""
        changes = []

        # 创建余额映射
        pre_map = {(b["accountIndex"], b["mint"]): float(b["uiTokenAmount"]["uiAmount"] or 0) for b in pre_balances}
        post_map = {(b["accountIndex"], b["mint"]): float(b["uiTokenAmount"]["uiAmount"] or 0) for b in post_balances}

        # 计算变化
        all_keys = set(pre_map.keys()) | set(post_map.keys())

        for key in all_keys:
            account_index, mint = key
            pre_amount = pre_map.get(key, 0)
            post_amount = post_map.get(key, 0)
            change = post_amount - pre_amount

            if abs(change) > 0.000001:  # 忽略极小变化
                changes.append({
                    "account_index": account_index,
                    "mint": mint,
                    "pre_amount": pre_amount,
                    "post_amount": post_amount,
                    "change": change
                })

        return changes

    async def _calculate_sol_amount(self, token_in: str, token_out: str, amount_in: float, amount_out: float) -> float:
        """计算等价SOL金额"""
        # SOL的mint地址
        sol_mint = "So11111111111111111111111111111111111111112"

        if token_in == sol_mint:
            return amount_in
        elif token_out == sol_mint:
            return amount_out
        else:
            # 对于非SOL交易，尝试估算SOL价值
            # 这里可以集成价格API来获取更准确的价格
            # 暂时返回0，表示无法确定SOL价值
            return 0.0

    def _should_alert(self, wallet: WalletConfig, swap_info: Dict[str, Any]) -> bool:
        """检查是否应该发出警报"""
        try:
            sol_amount = swap_info["sol_amount"]
            token_in = swap_info["token_in"]
            token_out = swap_info["token_out"]

            # 检查SOL金额阈值
            if sol_amount < wallet.min_sol_amount or sol_amount > wallet.max_sol_amount:
                return False

            # 检查全局SOL阈值
            sol_mint = "So11111111111111111111111111111111111111112"
            if token_in == sol_mint:  # 买入
                if sol_amount < self.config.min_sol_buy_amount:
                    return False
            elif token_out == sol_mint:  # 卖出
                if sol_amount < self.config.min_sol_sell_amount:
                    return False

            # 检查忽略token列表
            if token_in in self.config.ignore_tokens or token_out in self.config.ignore_tokens:
                return False

            # 检查冷却时间
            now = time.time()
            last_alert = self.alert_cooldowns.get(wallet.address, 0)
            if now - last_alert < self.config.alert_cooldown:
                return False

            return True

        except Exception as e:
            self.logger.error(f"检查警报条件失败: {e}")
            return False

    async def _create_swap_alert(self, wallet: WalletConfig, signature: str, swap_info: Dict[str, Any], tx_data: Dict[str, Any]) -> Optional[SwapAlert]:
        """创建swap警报"""
        try:
            sol_mint = "So11111111111111111111111111111111111111112"
            token_in = swap_info["token_in"]
            token_out = swap_info["token_out"]

            # 确定交易类型
            if token_in == sol_mint:
                swap_type = SwapType.BUY
                token_address = token_out
            elif token_out == sol_mint:
                swap_type = SwapType.SELL
                token_address = token_in
            else:
                swap_type = SwapType.BOTH
                token_address = token_out  # 默认使用输出token

            # 获取token信息
            token_info = await self._get_token_info(token_address)
            token_symbol = token_info.get("symbol", token_address[:8])
            token_name = token_info.get("name", "Unknown Token")

            # 确定警报级别
            sol_amount = swap_info["sol_amount"]
            if sol_amount >= 100:
                alert_level = AlertLevel.CRITICAL
            elif sol_amount >= 50:
                alert_level = AlertLevel.HIGH
            elif sol_amount >= 10:
                alert_level = AlertLevel.MEDIUM
            else:
                alert_level = AlertLevel.LOW

            return SwapAlert(
                wallet_address=wallet.address,
                transaction_signature=signature,
                swap_type=swap_type,
                token_in=token_in,
                token_out=token_out,
                amount_in=swap_info["amount_in"],
                amount_out=swap_info["amount_out"],
                sol_amount=sol_amount,
                token_symbol=token_symbol,
                token_name=token_name,
                token_address=token_address,
                timestamp=swap_info["timestamp"],
                alert_level=alert_level
            )

        except Exception as e:
            self.logger.error(f"创建警报失败: {e}")
            return None

    async def _get_token_info(self, token_address: str) -> Dict[str, Any]:
        """获取token信息"""
        try:
            # 这里可以集成token信息API
            # 暂时返回基本信息
            return {
                "symbol": f"TOKEN_{token_address[:8]}",
                "name": f"Token {token_address[:8]}",
                "address": token_address
            }
        except Exception as e:
            self.logger.error(f"获取token信息失败: {e}")
            return {
                "symbol": token_address[:8],
                "name": "Unknown Token",
                "address": token_address
            }

    async def _handle_alert(self, alert: SwapAlert):
        """处理警报"""
        try:
            # 更新冷却时间
            self.alert_cooldowns[alert.wallet_address] = time.time()

            # 记录警报
            self.logger.info(f"生成警报: {alert.swap_type.value} {alert.sol_amount} SOL - {alert.token_symbol}")

            # 调用所有警报处理器
            for handler in self.alert_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    self.logger.error(f"警报处理器错误: {e}")

        except Exception as e:
            self.logger.error(f"处理警报失败: {e}")

    def format_alert_message(self, alert: SwapAlert) -> str:
        """格式化警报消息"""
        try:
            # 确定使用的模板
            template_key = "buy_alert"
            if alert.swap_type == SwapType.SELL:
                template_key = "sell_alert"

            # 大额交易使用特殊模板
            if alert.sol_amount >= 50:
                if alert.swap_type == SwapType.BUY:
                    template_key = "large_buy"
                else:
                    template_key = "large_sell"

            # 获取模板
            template = self.config.prompt_templates.get(template_key, "")

            # 格式化时间
            timestamp_str = datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S")

            # 替换变量
            message = template.format(
                wallet_name=f"钱包{alert.wallet_address[:8]}",
                wallet_address=alert.wallet_address,
                token_symbol=alert.token_symbol,
                token_name=alert.token_name,
                token_address=alert.token_address,
                sol_amount=f"{alert.sol_amount:.4f}",
                amount_in=f"{alert.amount_in:.6f}",
                amount_out=f"{alert.amount_out:.6f}",
                transaction_signature=alert.transaction_signature,
                timestamp=timestamp_str,
                alert_level=alert.alert_level.value
            )

            return message

        except Exception as e:
            self.logger.error(f"格式化警报消息失败: {e}")
            return f"Swap Alert: {alert.swap_type.value} {alert.sol_amount} SOL - {alert.token_symbol}"

    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        return {
            "running": self.running,
            "monitored_wallets": len([w for w in self.config.wallets if w.enabled]),
            "total_wallets": len(self.config.wallets),
            "alert_handlers": len(self.alert_handlers),
            "cached_signatures": sum(len(sigs) for sigs in self.last_signatures.values()),
            "active_cooldowns": len([cd for cd in self.alert_cooldowns.values() if time.time() - cd < self.config.alert_cooldown]),
            "rpc_nodes": {
                "total_configured": len(self.config.rpc_nodes),
                "available": len(self.available_nodes),
                "current_node": self.available_nodes[self.current_node_index].name if self.available_nodes else None,
                "load_balancing": self.config.rpc_load_balancing,
                "failover": self.config.rpc_failover,
                "node_failures": dict(self.node_failures)
            }
        }
