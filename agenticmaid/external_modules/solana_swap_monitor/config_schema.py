"""
Solana Swap Monitor Configuration Schema
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class SwapType(Enum):
    """Transaction type"""
    BUY = "buy"
    SELL = "sell"
    BOTH = "both"


class AlertLevel(Enum):
    """Alert level"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SwapAlert:
    """Swap alert information"""
    wallet_address: str
    transaction_signature: str
    swap_type: SwapType
    token_in: str
    token_out: str
    amount_in: float
    amount_out: float
    sol_amount: float
    token_symbol: str
    token_name: str
    token_address: str
    timestamp: int
    alert_level: AlertLevel

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "wallet_address": self.wallet_address,
            "transaction_signature": self.transaction_signature,
            "swap_type": self.swap_type.value,
            "token_in": self.token_in,
            "token_out": self.token_out,
            "amount_in": self.amount_in,
            "amount_out": self.amount_out,
            "sol_amount": self.sol_amount,
            "token_symbol": self.token_symbol,
            "token_name": self.token_name,
            "token_address": self.token_address,
            "timestamp": self.timestamp,
            "alert_level": self.alert_level.value
        }


@dataclass
class WalletConfig:
    """Wallet monitoring configuration"""
    address: str
    name: str = ""
    description: str = ""
    enabled: bool = True
    min_sol_amount: float = 0.1
    max_sol_amount: float = 1000.0
    monitor_types: List[SwapType] = None
    alert_level: AlertLevel = AlertLevel.MEDIUM

    def __post_init__(self):
        if self.monitor_types is None:
            self.monitor_types = [SwapType.BOTH]


@dataclass
class RPCNode:
    """RPC node configuration"""
    url: str
    name: str = ""
    priority: int = 1  # Lower number = higher priority
    timeout: int = 30
    max_retries: int = 3
    enabled: bool = True

    def __post_init__(self):
        if not self.name:
            # Extract name from URL
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.url)
                self.name = parsed.netloc or self.url
            except:
                self.name = self.url


@dataclass
class SolanaSwapConfig:
    """Solana Swap monitoring configuration"""

    # RPC configuration - support multiple nodes
    rpc_nodes: List[RPCNode] = None
    rpc_url: str = "https://api.mainnet-beta.solana.com"  # Fallback for backward compatibility
    rpc_timeout: int = 30
    rpc_load_balancing: bool = True  # Enable load balancing across nodes
    rpc_failover: bool = True        # Enable automatic failover

    # Monitoring configuration
    polling_interval: float = 5.0
    max_retries: int = 3
    retry_delay: float = 2.0

    # Wallet list
    wallets: List[WalletConfig] = None

    # Filter conditions
    min_sol_buy_amount: float = 1.0
    min_sol_sell_amount: float = 0.5
    max_sol_amount: float = 1000.0

    # Token filtering
    ignore_tokens: List[str] = None  # Ignored token addresses
    focus_tokens: List[str] = None   # Focused token addresses
    meme_coin_only: bool = True      # Monitor meme coins only

    # Alert configuration
    alert_cooldown: int = 300        # Alert cooldown time for same wallet (seconds)
    duplicate_filter: bool = True    # Filter duplicate transactions

    # Output configuration
    prompt_templates: Dict[str, str] = None
    include_token_info: bool = True
    include_wallet_info: bool = True
    include_market_data: bool = False

    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "solana_swap_monitor.log"

    def __post_init__(self):
        if self.wallets is None:
            self.wallets = []

        # Initialize RPC nodes if not provided
        if self.rpc_nodes is None:
            # Create default node from rpc_url
            self.rpc_nodes = [
                RPCNode(
                    url=self.rpc_url,
                    name="Default Node",
                    priority=1,
                    timeout=self.rpc_timeout,
                    enabled=True
                )
            ]

        if self.ignore_tokens is None:
            self.ignore_tokens = [
                "So11111111111111111111111111111111111111112",  # WSOL
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
            ]

        if self.focus_tokens is None:
            self.focus_tokens = []

        if self.prompt_templates is None:
            self.prompt_templates = {
                "buy_alert": "🚀 **MEME COIN BUY SIGNAL** 🚀\n\n💰 Wallet: {wallet_name} ({wallet_address})\n🪙 Token: {token_symbol} ({token_name})\n📈 Buy: {sol_amount} SOL → {amount_out} {token_symbol}\n🔗 Transaction: {transaction_signature}\n⏰ Time: {timestamp}\n\n💡 **Copy Trade Suggestion**: Consider following this meme coin buy",

                "sell_alert": "📉 **MEME COIN SELL SIGNAL** 📉\n\n💰 Wallet: {wallet_name} ({wallet_address})\n🪙 Token: {token_symbol} ({token_name})\n📉 Sell: {amount_in} {token_symbol} → {sol_amount} SOL\n🔗 Transaction: {transaction_signature}\n⏰ Time: {timestamp}\n\n⚠️ **Risk Warning**: This wallet is selling, be cautious",

                "large_buy": "🔥 **LARGE MEME COIN BUY** 🔥\n\n💰 Wallet: {wallet_name} ({wallet_address})\n🪙 Token: {token_symbol} ({token_name})\n💎 Large Buy: {sol_amount} SOL\n🔗 Transaction: {transaction_signature}\n⏰ Time: {timestamp}\n\n🚨 **Important Signal**: Whale buying, worth attention!",

                "large_sell": "⚠️ **LARGE MEME COIN SELL** ⚠️\n\n💰 Wallet: {wallet_name} ({wallet_address})\n🪙 Token: {token_symbol} ({token_name})\n💸 Large Sell: {sol_amount} SOL\n🔗 Transaction: {transaction_signature}\n⏰ Time: {timestamp}\n\n🚨 **Risk Alert**: Whale selling, trade carefully!"
            }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SolanaSwapConfig":
        """Create configuration from dictionary"""
        # Process wallet configurations
        wallets = []
        for wallet_data in config_dict.get("wallets", []):
            if isinstance(wallet_data, dict):
                wallet_config = WalletConfig(
                    address=wallet_data["address"],
                    name=wallet_data.get("name", ""),
                    description=wallet_data.get("description", ""),
                    enabled=wallet_data.get("enabled", True),
                    min_sol_amount=wallet_data.get("min_sol_amount", 0.1),
                    max_sol_amount=wallet_data.get("max_sol_amount", 1000.0),
                    monitor_types=[SwapType(t) for t in wallet_data.get("monitor_types", ["both"])],
                    alert_level=AlertLevel(wallet_data.get("alert_level", "medium"))
                )
                wallets.append(wallet_config)

        # Process RPC nodes
        rpc_nodes = []
        for node_data in config_dict.get("rpc_nodes", []):
            if isinstance(node_data, dict):
                rpc_node = RPCNode(
                    url=node_data["url"],
                    name=node_data.get("name", ""),
                    priority=node_data.get("priority", 1),
                    timeout=node_data.get("timeout", 30),
                    max_retries=node_data.get("max_retries", 3),
                    enabled=node_data.get("enabled", True)
                )
                rpc_nodes.append(rpc_node)

        return cls(
            rpc_nodes=rpc_nodes if rpc_nodes else None,
            rpc_url=config_dict.get("rpc_url", "https://api.mainnet-beta.solana.com"),
            rpc_timeout=config_dict.get("rpc_timeout", 30),
            rpc_load_balancing=config_dict.get("rpc_load_balancing", True),
            rpc_failover=config_dict.get("rpc_failover", True),
            polling_interval=config_dict.get("polling_interval", 5.0),
            max_retries=config_dict.get("max_retries", 3),
            retry_delay=config_dict.get("retry_delay", 2.0),
            wallets=wallets,
            min_sol_buy_amount=config_dict.get("min_sol_buy_amount", 1.0),
            min_sol_sell_amount=config_dict.get("min_sol_sell_amount", 0.5),
            max_sol_amount=config_dict.get("max_sol_amount", 1000.0),
            ignore_tokens=config_dict.get("ignore_tokens", []),
            focus_tokens=config_dict.get("focus_tokens", []),
            meme_coin_only=config_dict.get("meme_coin_only", True),
            alert_cooldown=config_dict.get("alert_cooldown", 300),
            duplicate_filter=config_dict.get("duplicate_filter", True),
            prompt_templates=config_dict.get("prompt_templates", {}),
            include_token_info=config_dict.get("include_token_info", True),
            include_wallet_info=config_dict.get("include_wallet_info", True),
            include_market_data=config_dict.get("include_market_data", False),
            log_level=config_dict.get("log_level", "INFO"),
            log_file=config_dict.get("log_file", "solana_swap_monitor.log")
        )
