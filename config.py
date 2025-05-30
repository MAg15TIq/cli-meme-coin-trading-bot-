"""
Configuration module for the Solana Memecoin Trading Bot.
Handles loading and managing configuration settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Default configuration
DEFAULT_CONFIG = {
    # Solana network settings
    "rpc_url": os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
    "secondary_rpc_url": os.getenv("SECONDARY_RPC_URL", ""),  # Backup RPC endpoint
    "rpc_ws_url": os.getenv("RPC_WS_URL", "wss://api.mainnet-beta.solana.com"),  # WebSocket RPC endpoint
    "network": os.getenv("SOLANA_NETWORK", "mainnet-beta"),  # mainnet-beta, testnet, devnet
    "rpc_endpoints": {  # Multiple RPC endpoints for failover and load balancing
        "default": os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
        "backup": os.getenv("BACKUP_RPC_URL", ""),
        "jito": os.getenv("JITO_RPC_URL", ""),  # MEV-protected RPC
        "genesysgo": os.getenv("GENESYSGO_RPC_URL", ""),  # Low-latency RPC
    },

    # Wallet settings
    "wallet_path": os.getenv("WALLET_PATH", str(Path.home() / ".solana-trading-bot" / "wallets")),
    "encrypted_key_file": os.getenv("ENCRYPTED_KEY_FILE", "encrypted_key.json"),

    # Hardware wallet settings
    "hardware_wallet_enabled": bool(os.getenv("HARDWARE_WALLET_ENABLED", "False")),
    "hardware_wallet_type": os.getenv("HARDWARE_WALLET_TYPE", "ledger"),
    "hardware_wallet_derivation_path": os.getenv("HARDWARE_WALLET_DERIVATION_PATH", "44'/501'/0'/0'"),
    "hardware_wallet_connection": os.getenv("HARDWARE_WALLET_CONNECTION", "usb"),
    "hardware_wallet_auto_confirm": bool(os.getenv("HARDWARE_WALLET_AUTO_CONFIRM", "False")),
    "hardware_wallet_data_path": os.getenv("HARDWARE_WALLET_DATA_PATH",
                                          str(Path.home() / ".solana-trading-bot" / "hardware_wallets")),

    # Logging settings
    "log_level": os.getenv("LOG_LEVEL", "INFO"),

    # Jupiter API settings
    "jupiter_api_url": "https://quote-api.jup.ag/v6",
    "slippage_bps": int(os.getenv("SLIPPAGE_BPS", "50")),  # Default 0.5%

    # Trading settings
    "stop_loss_percentage": float(os.getenv("STOP_LOSS_PERCENTAGE", "5.0")),  # Default 5%
    "take_profit_percentage": float(os.getenv("TAKE_PROFIT_PERCENTAGE", "20.0")),  # Default 20%
    "monitoring_interval_seconds": int(os.getenv("MONITORING_INTERVAL_SECONDS", "60")),  # Default 60 seconds

    # Multi-tiered take-profit settings
    "take_profit_tiers": [
        {"percentage": 20.0, "sell_percentage": 30.0},  # Sell 30% at 20% profit
        {"percentage": 50.0, "sell_percentage": 40.0},  # Sell 40% at 50% profit
        {"percentage": 100.0, "sell_percentage": 30.0},  # Sell 30% at 100% profit
    ],

    # Enhanced exit strategy settings
    "exit_strategy": {
        "strategy_type": os.getenv("EXIT_STRATEGY_TYPE", "tiered"),  # simple, tiered, trailing, smart
        "trailing_stop": {
            "enabled": bool(os.getenv("TRAILING_STOP_ENABLED", "True").lower() == "true"),
            "trail_percentage": float(os.getenv("TRAILING_STOP_PERCENTAGE", "15.0")),
            "activation_percentage": float(os.getenv("TRAILING_STOP_ACTIVATION", "20.0"))
        },
        "time_based_exit": {
            "enabled": bool(os.getenv("TIME_BASED_EXIT_ENABLED", "True").lower() == "true"),
            "hold_hours": int(os.getenv("TIME_BASED_EXIT_HOURS", "48")),
            "min_profit_percentage": float(os.getenv("TIME_BASED_EXIT_MIN_PROFIT", "5.0"))
        },
        "indicator_based_exit": {
            "enabled": bool(os.getenv("INDICATOR_EXIT_ENABLED", "True").lower() == "true"),
            "rsi_overbought": float(os.getenv("RSI_OVERBOUGHT", "75.0")),
            "macd_signal": bool(os.getenv("MACD_SIGNAL_EXIT", "True").lower() == "true"),
            "bollinger_band": bool(os.getenv("BOLLINGER_BAND_EXIT", "True").lower() == "true")
        },
        "dynamic_stop_loss": {
            "enabled": bool(os.getenv("DYNAMIC_STOP_LOSS_ENABLED", "True").lower() == "true"),
            "initial_percentage": float(os.getenv("INITIAL_STOP_LOSS", "5.0")),
            "profit_step_percentage": float(os.getenv("PROFIT_STEP_PERCENTAGE", "10.0")),
            "stop_loss_step_percentage": float(os.getenv("STOP_LOSS_STEP_PERCENTAGE", "2.5"))
        }
    },

    # Helius API settings
    "helius_api_key": os.getenv("HELIUS_API_KEY", ""),
    "helius_webhook_url": os.getenv("HELIUS_WEBHOOK_URL", ""),
    "helius_webhook_id": os.getenv("HELIUS_WEBHOOK_ID", ""),
    "tracked_wallets": [],

    # Pool monitoring and sniping settings
    "rpc_ws_url": os.getenv("RPC_WS_URL", "wss://api.mainnet-beta.solana.com"),
    "sniping_enabled": bool(os.getenv("SNIPING_ENABLED", "False").lower() == "true"),
    "min_liquidity_sol": float(os.getenv("MIN_LIQUIDITY_SOL", "1.0")),
    "max_liquidity_sol": float(os.getenv("MAX_LIQUIDITY_SOL", "100.0")),
    "snipe_amount_sol": float(os.getenv("SNIPE_AMOUNT_SOL", "0.1")),
    "auto_sell_percentage": float(os.getenv("AUTO_SELL_PERCENTAGE", "200.0")),
    "blacklisted_tokens": [],

    # Enhanced token filtering settings
    "min_initial_liquidity_sol": float(os.getenv("MIN_INITIAL_LIQUIDITY_SOL", "5.0")),
    "min_token_holders": int(os.getenv("MIN_TOKEN_HOLDERS", "0")),  # 0 means don't check
    "max_creator_allocation_percent": float(os.getenv("MAX_CREATOR_ALLOCATION_PERCENT", "50.0")),
    "min_token_age_seconds": int(os.getenv("MIN_TOKEN_AGE_SECONDS", "0")),  # 0 means don't check
    "require_verified_contract": bool(os.getenv("REQUIRE_VERIFIED_CONTRACT", "False").lower() == "true"),
    "require_locked_liquidity": bool(os.getenv("REQUIRE_LOCKED_LIQUIDITY", "False").lower() == "true"),
    "honeypot_detection_enabled": bool(os.getenv("HONEYPOT_DETECTION_ENABLED", "True").lower() == "true"),
    "suspicious_contract_patterns": [
        "selfdestruct", "blacklist", "whitelist", "pause", "freeze", "owner"
    ],

    # Token-specific settings (can be overridden per token)
    "token_settings": {},

    # Priority fee settings
    "priority_fee_percentile": int(os.getenv("PRIORITY_FEE_PERCENTILE", "75")),  # Default 75th percentile
    "min_priority_fee": int(os.getenv("MIN_PRIORITY_FEE", "1000")),  # Default 1000 micro-lamports
    "priority_fee_multipliers": {
        "buy": float(os.getenv("BUY_FEE_MULTIPLIER", "1.0")),
        "sell": float(os.getenv("SELL_FEE_MULTIPLIER", "1.0")),
        "snipe": float(os.getenv("SNIPE_FEE_MULTIPLIER", "2.0")),  # Higher priority for sniping
        "swap": float(os.getenv("SWAP_FEE_MULTIPLIER", "1.2")),    # Slightly higher for swaps
        "default": float(os.getenv("DEFAULT_FEE_MULTIPLIER", "1.0")),
    },
    "fee_optimization_enabled": bool(os.getenv("FEE_OPTIMIZATION_ENABLED", "True").lower() == "true"),
    "time_based_fee_adjustment": bool(os.getenv("TIME_BASED_FEE_ADJUSTMENT", "False").lower() == "true"),
    "urgent_sell_fee_boost": bool(os.getenv("URGENT_SELL_FEE_BOOST", "True").lower() == "true"),
    "compute_unit_limit": int(os.getenv("COMPUTE_UNIT_LIMIT", "200000")),
    "wait_for_finalization": bool(os.getenv("WAIT_FOR_FINALIZATION", "False").lower() == "true"),
    "rpc_health_check_interval": int(os.getenv("RPC_HEALTH_CHECK_INTERVAL", "300")),  # 5 minutes

    # Enhanced gas optimization settings
    "fee_history_max_entries": int(os.getenv("FEE_HISTORY_MAX_ENTRIES", "1000")),
    "congestion_check_interval": int(os.getenv("CONGESTION_CHECK_INTERVAL", "60")),  # seconds
    "low_priority_percentile": int(os.getenv("LOW_PRIORITY_PERCENTILE", "25")),
    "medium_priority_percentile": int(os.getenv("MEDIUM_PRIORITY_PERCENTILE", "50")),
    "high_priority_percentile": int(os.getenv("HIGH_PRIORITY_PERCENTILE", "75")),
    "urgent_priority_percentile": int(os.getenv("URGENT_PRIORITY_PERCENTILE", "90")),
    "limit_order_fee_multiplier": float(os.getenv("LIMIT_ORDER_FEE_MULTIPLIER", "0.8")),
    "withdraw_fee_multiplier": float(os.getenv("WITHDRAW_FEE_MULTIPLIER", "1.5")),
    "buy_compute_limit": int(os.getenv("BUY_COMPUTE_LIMIT", "200000")),
    "sell_compute_limit": int(os.getenv("SELL_COMPUTE_LIMIT", "200000")),
    "snipe_compute_limit": int(os.getenv("SNIPE_COMPUTE_LIMIT", "240000")),
    "swap_compute_limit": int(os.getenv("SWAP_COMPUTE_LIMIT", "220000")),
    "limit_order_compute_limit": int(os.getenv("LIMIT_ORDER_COMPUTE_LIMIT", "200000")),
    "withdraw_compute_limit": int(os.getenv("WITHDRAW_COMPUTE_LIMIT", "200000")),

    # Technical analysis settings
    "technical_analysis_enabled": bool(os.getenv("TECHNICAL_ANALYSIS_ENABLED", "False").lower() == "true"),
    "ta_update_interval": int(os.getenv("TA_UPDATE_INTERVAL", "300")),  # Default: 5 minutes
    "price_data_file": os.getenv("PRICE_DATA_FILE", "price_data.json"),
    "indicators_file": os.getenv("INDICATORS_FILE", "indicators.json"),

    # Chart settings
    "charts_enabled": bool(os.getenv("CHARTS_ENABLED", "False").lower() == "true"),
    "chart_dir": os.getenv("CHART_DIR", "charts"),
    "chart_format": os.getenv("CHART_FORMAT", "ascii"),  # ascii, png, svg

    # Mobile app settings
    "mobile_app_enabled": bool(os.getenv("MOBILE_APP_ENABLED", "False").lower() == "true"),
    "mobile_app_api_key": os.getenv("MOBILE_APP_API_KEY", ""),
    "mobile_app_device_tokens": [],
    "mobile_app_pairing_code": "",
    "mobile_app_pairing_expiry": 0,
    "mobile_app_data_path": os.getenv("MOBILE_APP_DATA_PATH",
                                    str(Path.home() / ".solana-trading-bot" / "mobile_app")),
    "mobile_app_notification_settings": {
        "trade_executed": True,
        "price_alert": True,
        "wallet_connected": True,
        "position_closed": True,
        "error": True,
        "security_alert": True
    },

    # ML token evaluation settings
    "ml_evaluation_enabled": bool(os.getenv("ML_EVALUATION_ENABLED", "False").lower() == "true"),
    "ml_monitoring_interval": int(os.getenv("ML_MONITORING_INTERVAL", "3600")),  # Default: 1 hour
    "ml_data_path": os.getenv("ML_DATA_PATH",
                            str(Path.home() / ".solana-trading-bot" / "ml_models")),

    # Community strategy sharing settings
    "strategy_sharing_enabled": bool(os.getenv("STRATEGY_SHARING_ENABLED", "False").lower() == "true"),
    "community_strategies_path": os.getenv("COMMUNITY_STRATEGIES_PATH",
                                         str(Path.home() / ".solana-trading-bot" / "community_strategies")),

    # Price alerts settings
    "price_alerts_enabled": bool(os.getenv("PRICE_ALERTS_ENABLED", "False").lower() == "true"),
    "price_alert_interval_seconds": int(os.getenv("PRICE_ALERT_INTERVAL_SECONDS", "60")),
    "price_alerts": {},
    "triggered_price_alerts": [],

    # External wallet monitoring settings
    "wallet_monitoring_enabled": bool(os.getenv("WALLET_MONITORING_ENABLED", "False").lower() == "true"),
    "wallet_monitoring_interval_seconds": int(os.getenv("WALLET_MONITORING_INTERVAL_SECONDS", "300")),
    "monitored_wallets": {},
    "wallet_transactions": {},
    "max_transactions_per_wallet": int(os.getenv("MAX_TRANSACTIONS_PER_WALLET", "100")),

    # Withdrawal settings
    "max_withdrawal_amount_sol": float(os.getenv("MAX_WITHDRAWAL_AMOUNT_SOL", "10.0")),
    "require_withdrawal_confirmation": bool(os.getenv("REQUIRE_WITHDRAWAL_CONFIRMATION", "True").lower() == "true"),
    "withdrawal_history": [],

    # Limit orders settings
    "limit_orders_enabled": bool(os.getenv("LIMIT_ORDERS_ENABLED", "False").lower() == "true"),
    "limit_order_interval_seconds": int(os.getenv("LIMIT_ORDER_INTERVAL_SECONDS", "30")),
    "limit_orders": {},

    # DCA orders settings
    "dca_enabled": bool(os.getenv("DCA_ENABLED", "False").lower() == "true"),
    "dca_interval_seconds": int(os.getenv("DCA_INTERVAL_SECONDS", "60")),
    "dca_orders": {},
    "dca_executions": {},

    # Auto-buy settings
    "auto_buy_enabled": bool(os.getenv("AUTO_BUY_ENABLED", "False").lower() == "true"),
    "auto_buy_default_amount": float(os.getenv("AUTO_BUY_DEFAULT_AMOUNT", "0.1")),
    "auto_buy_require_confirmation": bool(os.getenv("AUTO_BUY_REQUIRE_CONFIRMATION", "True").lower() == "true"),

    # Token analytics settings
    "token_analytics_enabled": bool(os.getenv("TOKEN_ANALYTICS_ENABLED", "False").lower() == "true"),
    "analytics_monitoring_interval": int(os.getenv("ANALYTICS_MONITORING_INTERVAL", "3600")),
    "max_analytics_history_entries": int(os.getenv("MAX_ANALYTICS_HISTORY_ENTRIES", "100")),
    "github_token": os.getenv("GITHUB_TOKEN", ""),

    # MEV protection settings
    "mev_protection_enabled": bool(os.getenv("MEV_PROTECTION_ENABLED", "False").lower() == "true"),
    "mev_bundle_tip_lamports": int(os.getenv("MEV_BUNDLE_TIP_LAMPORTS", "100000")),  # 0.0001 SOL
    "mev_protection_level": os.getenv("MEV_PROTECTION_LEVEL", "standard"),  # standard, aggressive, maximum
    "mev_protection_providers": {
        "jito": bool(os.getenv("USE_JITO_MEV", "True").lower() == "true"),
        "helius": bool(os.getenv("USE_HELIUS_MEV", "False").lower() == "true"),
        "flashbots": bool(os.getenv("USE_FLASHBOTS_MEV", "False").lower() == "true")
    },
    "mev_protection_transaction_types": {
        "buy": bool(os.getenv("MEV_PROTECT_BUY", "False").lower() == "true"),
        "sell": bool(os.getenv("MEV_PROTECT_SELL", "True").lower() == "true"),
        "snipe": bool(os.getenv("MEV_PROTECT_SNIPE", "True").lower() == "true"),
        "swap": bool(os.getenv("MEV_PROTECT_SWAP", "False").lower() == "true")
    },

    # Risk management settings
    "risk_management_enabled": bool(os.getenv("RISK_MANAGEMENT_ENABLED", "True").lower() == "true"),
    "max_portfolio_allocation_percent": float(os.getenv("MAX_PORTFOLIO_ALLOCATION_PERCENT", "20.0")),
    "max_token_allocation_percent": float(os.getenv("MAX_TOKEN_ALLOCATION_PERCENT", "5.0")),
    "max_high_risk_allocation_percent": float(os.getenv("MAX_HIGH_RISK_ALLOCATION_PERCENT", "10.0")),
    "max_very_high_risk_allocation_percent": float(os.getenv("MAX_VERY_HIGH_RISK_ALLOCATION_PERCENT", "5.0")),
    "auto_adjust_position_size": bool(os.getenv("AUTO_ADJUST_POSITION_SIZE", "True").lower() == "true"),
    "token_analytics_cache_ttl": int(os.getenv("TOKEN_ANALYTICS_CACHE_TTL", "3600")),

    # Enhanced risk management settings
    "risk_profile": os.getenv("RISK_PROFILE", "moderate"),  # conservative, moderate, aggressive
    "conservative_max_allocation_percent": float(os.getenv("CONSERVATIVE_MAX_ALLOCATION_PERCENT", "20.0")),
    "conservative_max_position_percent": float(os.getenv("CONSERVATIVE_MAX_POSITION_PERCENT", "2.0")),
    "conservative_stop_loss_percent": float(os.getenv("CONSERVATIVE_STOP_LOSS_PERCENT", "5.0")),
    "conservative_max_drawdown_percent": float(os.getenv("CONSERVATIVE_MAX_DRAWDOWN_PERCENT", "10.0")),
    "moderate_max_allocation_percent": float(os.getenv("MODERATE_MAX_ALLOCATION_PERCENT", "40.0")),
    "moderate_max_position_percent": float(os.getenv("MODERATE_MAX_POSITION_PERCENT", "5.0")),
    "moderate_stop_loss_percent": float(os.getenv("MODERATE_STOP_LOSS_PERCENT", "10.0")),
    "moderate_max_drawdown_percent": float(os.getenv("MODERATE_MAX_DRAWDOWN_PERCENT", "20.0")),
    "aggressive_max_allocation_percent": float(os.getenv("AGGRESSIVE_MAX_ALLOCATION_PERCENT", "60.0")),
    "aggressive_max_position_percent": float(os.getenv("AGGRESSIVE_MAX_POSITION_PERCENT", "10.0")),
    "aggressive_stop_loss_percent": float(os.getenv("AGGRESSIVE_STOP_LOSS_PERCENT", "15.0")),
    "aggressive_max_drawdown_percent": float(os.getenv("AGGRESSIVE_MAX_DRAWDOWN_PERCENT", "30.0")),
    "max_low_risk_allocation_percent": float(os.getenv("MAX_LOW_RISK_ALLOCATION_PERCENT", "60.0")),
    "max_medium_risk_allocation_percent": float(os.getenv("MAX_MEDIUM_RISK_ALLOCATION_PERCENT", "30.0")),

    # Parameter refinement settings
    "auto_refinement_enabled": bool(os.getenv("AUTO_REFINEMENT_ENABLED", "False").lower() == "true"),
    "risk_refinement_interval_days": int(os.getenv("RISK_REFINEMENT_INTERVAL_DAYS", "7")),
    "gas_refinement_interval_days": int(os.getenv("GAS_REFINEMENT_INTERVAL_DAYS", "3")),
    "last_risk_refinement": float(os.getenv("LAST_RISK_REFINEMENT", "0")),
    "last_gas_refinement": float(os.getenv("LAST_GAS_REFINEMENT", "0")),
    "min_trades_for_refinement": int(os.getenv("MIN_TRADES_FOR_REFINEMENT", "5")),
    "min_transactions_for_refinement": int(os.getenv("MIN_TRANSACTIONS_FOR_REFINEMENT", "10")),

    # Backtesting settings
    "backtesting_enabled": bool(os.getenv("BACKTESTING_ENABLED", "False").lower() == "true"),
    "backtesting_data_path": os.getenv("BACKTESTING_DATA_PATH",
                                      str(Path.home() / ".solana-trading-bot" / "backtesting_data")),
    "backtesting_trading_fee": float(os.getenv("BACKTESTING_TRADING_FEE", "0.003")),  # 0.3%
    "backtesting_slippage": float(os.getenv("BACKTESTING_SLIPPAGE", "0.001")),  # 0.1%

    # Advanced orders settings
    "advanced_orders_enabled": bool(os.getenv("ADVANCED_ORDERS_ENABLED", "False").lower() == "true"),
    "advanced_orders_execution_interval": int(os.getenv("ADVANCED_ORDERS_EXECUTION_INTERVAL", "30")),  # seconds
    "twap_default_interval_minutes": int(os.getenv("TWAP_DEFAULT_INTERVAL_MINUTES", "5")),
    "vwap_default_participation_rate": float(os.getenv("VWAP_DEFAULT_PARTICIPATION_RATE", "0.1")),  # 10%
    "iceberg_default_slice_percentage": float(os.getenv("ICEBERG_DEFAULT_SLICE_PERCENTAGE", "0.1")),  # 10%

    # Portfolio analytics settings
    "portfolio_analytics_enabled": bool(os.getenv("PORTFOLIO_ANALYTICS_ENABLED", "False").lower() == "true"),
    "portfolio_analytics_data_path": os.getenv("PORTFOLIO_ANALYTICS_DATA_PATH",
                                              str(Path.home() / ".solana-trading-bot" / "portfolio_data")),
    "portfolio_snapshot_interval": int(os.getenv("PORTFOLIO_SNAPSHOT_INTERVAL", "300")),  # 5 minutes
    "portfolio_max_snapshots": int(os.getenv("PORTFOLIO_MAX_SNAPSHOTS", "1000")),

    # Enhanced portfolio management settings
    "enhanced_portfolio_enabled": bool(os.getenv("ENHANCED_PORTFOLIO_ENABLED", "False").lower() == "true"),
    "portfolio_allocation_strategy": os.getenv("PORTFOLIO_ALLOCATION_STRATEGY", "equal_weight"),
    "portfolio_max_positions": int(os.getenv("PORTFOLIO_MAX_POSITIONS", "20")),
    "portfolio_max_position_size": float(os.getenv("PORTFOLIO_MAX_POSITION_SIZE", "10.0")),
    "portfolio_max_sector_allocation": float(os.getenv("PORTFOLIO_MAX_SECTOR_ALLOCATION", "30.0")),
    "portfolio_min_liquidity": float(os.getenv("PORTFOLIO_MIN_LIQUIDITY", "1.0")),
    "portfolio_max_correlation": float(os.getenv("PORTFOLIO_MAX_CORRELATION", "0.8")),
    "portfolio_rebalance_threshold": float(os.getenv("PORTFOLIO_REBALANCE_THRESHOLD", "5.0")),
    "portfolio_auto_rebalance": bool(os.getenv("PORTFOLIO_AUTO_REBALANCE", "False").lower() == "true"),
    "portfolio_rebalance_frequency": int(os.getenv("PORTFOLIO_REBALANCE_FREQUENCY", "24")),  # hours
    "portfolio_target_allocations": {},
    "portfolio_sector_allocations": {},
    "portfolio_last_rebalance": float(os.getenv("PORTFOLIO_LAST_REBALANCE", "0")),
    "portfolio_performance_history": [],
    "portfolio_benchmark_returns": [],
    "portfolio_risk_budget": {
        "low_risk": 40.0,
        "medium_risk": 35.0,
        "high_risk": 20.0,
        "very_high_risk": 5.0
    },

    # Enhanced copy trading settings
    "copy_tracked_wallets": {},
    "copy_wallet_performance": {},
    "copy_delay_seconds": int(os.getenv("COPY_DELAY_SECONDS", "5")),
    "copy_max_position_size_sol": float(os.getenv("COPY_MAX_POSITION_SIZE_SOL", "5.0")),
    "copy_stop_loss_percentage": float(os.getenv("COPY_STOP_LOSS_PERCENTAGE", "10.0")),
    "copy_take_profit_percentage": float(os.getenv("COPY_TAKE_PROFIT_PERCENTAGE", "50.0")),
    "copy_min_success_rate": float(os.getenv("COPY_MIN_SUCCESS_RATE", "60.0")),
    "copy_min_profit_ratio": float(os.getenv("COPY_MIN_PROFIT_RATIO", "1.5")),
    "copy_max_drawdown": float(os.getenv("COPY_MAX_DRAWDOWN", "30.0")),

    # Enhanced pool monitoring settings
    "max_supply_concentration": float(os.getenv("MAX_SUPPLY_CONCENTRATION", "20.0")),
    "min_trading_volume_24h": float(os.getenv("MIN_TRADING_VOLUME_24H", "1.0")),
    "max_price_impact_threshold": float(os.getenv("MAX_PRICE_IMPACT_THRESHOLD", "5.0")),
    "require_social_presence": bool(os.getenv("REQUIRE_SOCIAL_PRESENCE", "False").lower() == "true"),
    "min_liquidity_lock_duration": int(os.getenv("MIN_LIQUIDITY_LOCK_DURATION", "0")),
    "early_entry_enabled": bool(os.getenv("EARLY_ENTRY_ENABLED", "False").lower() == "true"),
    "early_entry_max_age_seconds": int(os.getenv("EARLY_ENTRY_MAX_AGE_SECONDS", "300")),
    "early_entry_multiplier": float(os.getenv("EARLY_ENTRY_MULTIPLIER", "2.0")),
    "early_entry_max_amount": float(os.getenv("EARLY_ENTRY_MAX_AMOUNT", "2.0")),
    "safety_score_threshold": float(os.getenv("SAFETY_SCORE_THRESHOLD", "70.0")),
    "enable_rugpull_detection": bool(os.getenv("ENABLE_RUGPULL_DETECTION", "True").lower() == "true"),
    "enable_whale_tracking": bool(os.getenv("ENABLE_WHALE_TRACKING", "True").lower() == "true"),

    # Advanced alert system settings
    "advanced_alerts_enabled": bool(os.getenv("ADVANCED_ALERTS_ENABLED", "False").lower() == "true"),
    "alert_monitoring_interval": int(os.getenv("ALERT_MONITORING_INTERVAL", "30")),  # seconds
    "alert_price_check_interval": int(os.getenv("ALERT_PRICE_CHECK_INTERVAL", "10")),  # seconds
    "alert_notifications_enabled": bool(os.getenv("ALERT_NOTIFICATIONS_ENABLED", "True").lower() == "true"),
    "alert_notification_channels": ["console", "log"],
    "alert_rate_limit_window": int(os.getenv("ALERT_RATE_LIMIT_WINDOW", "300")),  # 5 minutes
    "max_alerts_per_window": int(os.getenv("MAX_ALERTS_PER_WINDOW", "10")),
    "max_alert_history": int(os.getenv("MAX_ALERT_HISTORY", "1000")),
    "alert_conditions": {},

    # Smart copy trading settings
    "smart_copy_discovery_enabled": bool(os.getenv("SMART_COPY_DISCOVERY_ENABLED", "False").lower() == "true"),
    "wallet_discovery_interval_hours": int(os.getenv("WALLET_DISCOVERY_INTERVAL_HOURS", "24")),
    "min_wallet_score": float(os.getenv("MIN_WALLET_SCORE", "70.0")),
    "min_trades_for_discovery": int(os.getenv("MIN_TRADES_FOR_DISCOVERY", "10")),
    "min_volume_for_discovery": float(os.getenv("MIN_VOLUME_FOR_DISCOVERY", "50.0")),
    "copy_max_correlation": float(os.getenv("COPY_MAX_CORRELATION", "0.7")),
    "copy_portfolio_limit": float(os.getenv("COPY_PORTFOLIO_LIMIT", "0.3")),

    # Multi-DEX monitoring settings
    "multi_dex_enabled": bool(os.getenv("MULTI_DEX_ENABLED", "False").lower() == "true"),
    "arbitrage_detection_enabled": bool(os.getenv("ARBITRAGE_DETECTION_ENABLED", "False").lower() == "true"),
    "min_arbitrage_profit_bps": int(os.getenv("MIN_ARBITRAGE_PROFIT_BPS", "50")),

    # Phase 3: Dynamic Portfolio Optimization settings
    "dynamic_optimization_enabled": bool(os.getenv("DYNAMIC_OPTIMIZATION_ENABLED", "False").lower() == "true"),
    "optimization_method": os.getenv("OPTIMIZATION_METHOD", "mean_variance"),  # mean_variance, risk_parity, black_litterman
    "rebalancing_frequency_hours": int(os.getenv("REBALANCING_FREQUENCY_HOURS", "24")),
    "optimization_lookback_days": int(os.getenv("OPTIMIZATION_LOOKBACK_DAYS", "30")),
    "max_portfolio_turnover": float(os.getenv("MAX_PORTFOLIO_TURNOVER", "0.2")),
    "regime_detection_enabled": bool(os.getenv("REGIME_DETECTION_ENABLED", "True").lower() == "true"),
    "efficient_frontier_points": int(os.getenv("EFFICIENT_FRONTIER_POINTS", "100")),

    # Phase 3: Enhanced AI Pool Analysis settings
    "enhanced_ai_analysis_enabled": bool(os.getenv("ENHANCED_AI_ANALYSIS_ENABLED", "False").lower() == "true"),
    "model_retraining_enabled": bool(os.getenv("MODEL_RETRAINING_ENABLED", "True").lower() == "true"),
    "model_update_frequency_hours": int(os.getenv("MODEL_UPDATE_FREQUENCY_HOURS", "6")),
    "ensemble_models_enabled": bool(os.getenv("ENSEMBLE_MODELS_ENABLED", "True").lower() == "true"),
    "cross_chain_analysis_enabled": bool(os.getenv("CROSS_CHAIN_ANALYSIS_ENABLED", "False").lower() == "true"),
    "model_storage_path": os.getenv("MODEL_STORAGE_PATH", str(Path.home() / ".solana-trading-bot" / "models")),
    "feature_importance_threshold": float(os.getenv("FEATURE_IMPORTANCE_THRESHOLD", "0.01")),

    # Phase 3: Advanced Benchmarking settings
    "advanced_benchmarking_enabled": bool(os.getenv("ADVANCED_BENCHMARKING_ENABLED", "False").lower() == "true"),
    "benchmark_update_frequency_hours": int(os.getenv("BENCHMARK_UPDATE_FREQUENCY_HOURS", "1")),
    "performance_forecasting_enabled": bool(os.getenv("PERFORMANCE_FORECASTING_ENABLED", "True").lower() == "true"),
    "custom_benchmarks": [],
    "risk_free_rate": float(os.getenv("RISK_FREE_RATE", "0.02")),  # 2% annual
    "factor_analysis_enabled": bool(os.getenv("FACTOR_ANALYSIS_ENABLED", "True").lower() == "true"),
    "monte_carlo_simulations": int(os.getenv("MONTE_CARLO_SIMULATIONS", "1000")),
    "forecast_confidence_levels": [0.05, 0.25, 0.5, 0.75, 0.95],

    # Phase 4: Live Trading & Real Data
    "live_trading_enabled": bool(os.getenv("LIVE_TRADING_ENABLED", "False").lower() == "true"),
    "paper_trading_mode": bool(os.getenv("PAPER_TRADING_MODE", "False").lower() == "true"),
    "max_daily_loss_limit": float(os.getenv("MAX_DAILY_LOSS_LIMIT", "1000.0")),
    "emergency_stop_enabled": bool(os.getenv("EMERGENCY_STOP_ENABLED", "True").lower() == "true"),
    "execution_latency_target_ms": int(os.getenv("EXECUTION_LATENCY_TARGET_MS", "100")),
    "real_data_feeds_enabled": bool(os.getenv("REAL_DATA_FEEDS_ENABLED", "False").lower() == "true"),
    "websocket_feeds_enabled": bool(os.getenv("WEBSOCKET_FEEDS_ENABLED", "False").lower() == "true"),
    "historical_data_collection_enabled": bool(os.getenv("HISTORICAL_DATA_COLLECTION_ENABLED", "False").lower() == "true"),
    "data_validation_enabled": bool(os.getenv("DATA_VALIDATION_ENABLED", "True").lower() == "true"),

    # Phase 4: Advanced AI & Machine Learning
    "deep_learning_enabled": bool(os.getenv("DEEP_LEARNING_ENABLED", "False").lower() == "true"),
    "model_training_enabled": bool(os.getenv("MODEL_TRAINING_ENABLED", "True").lower() == "true"),
    "reinforcement_learning_enabled": bool(os.getenv("REINFORCEMENT_LEARNING_ENABLED", "False").lower() == "true"),
    "sentiment_analysis_enabled": bool(os.getenv("SENTIMENT_ANALYSIS_ENABLED", "True").lower() == "true"),
    "pattern_recognition_enabled": bool(os.getenv("PATTERN_RECOGNITION_ENABLED", "True").lower() == "true"),
    "lstm_models_enabled": bool(os.getenv("LSTM_MODELS_ENABLED", "False").lower() == "true"),
    "transformer_models_enabled": bool(os.getenv("TRANSFORMER_MODELS_ENABLED", "False").lower() == "true"),
    "ensemble_ai_enabled": bool(os.getenv("ENSEMBLE_AI_ENABLED", "True").lower() == "true"),
    "ai_model_retraining_hours": int(os.getenv("AI_MODEL_RETRAINING_HOURS", "24")),
    "ai_confidence_threshold": float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.7")),
    "ai_model_storage_path": os.getenv("AI_MODEL_STORAGE_PATH", str(Path.home() / ".solana-trading-bot" / "ai_models")),

    # Phase 4: Cross-Chain Integration
    "cross_chain_enabled": bool(os.getenv("CROSS_CHAIN_ENABLED", "False").lower() == "true"),
    "ethereum_enabled": bool(os.getenv("ETHEREUM_ENABLED", "False").lower() == "true"),
    "bsc_enabled": bool(os.getenv("BSC_ENABLED", "False").lower() == "true"),
    "polygon_enabled": bool(os.getenv("POLYGON_ENABLED", "False").lower() == "true"),
    "cross_chain_arbitrage_enabled": bool(os.getenv("CROSS_CHAIN_ARBITRAGE_ENABLED", "False").lower() == "true"),
    "min_arbitrage_profit_percentage": float(os.getenv("MIN_ARBITRAGE_PROFIT_PERCENTAGE", "1.0")),
    "bridge_cost_estimation_enabled": bool(os.getenv("BRIDGE_COST_ESTIMATION_ENABLED", "True").lower() == "true"),
    "cross_chain_portfolio_enabled": bool(os.getenv("CROSS_CHAIN_PORTFOLIO_ENABLED", "False").lower() == "true"),
    "ethereum_rpc_url": os.getenv("ETHEREUM_RPC_URL", "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"),
    "bsc_rpc_url": os.getenv("BSC_RPC_URL", "https://bsc-dataseed1.binance.org/"),
    "polygon_rpc_url": os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com/"),

    # Phase 4: Enterprise Features
    "api_enabled": bool(os.getenv("API_ENABLED", "False").lower() == "true"),
    "api_host": os.getenv("API_HOST", "0.0.0.0"),
    "api_port": int(os.getenv("API_PORT", "8000")),
    "api_workers": int(os.getenv("API_WORKERS", "1")),
    "multi_user_enabled": bool(os.getenv("MULTI_USER_ENABLED", "False").lower() == "true"),
    "database_enabled": bool(os.getenv("DATABASE_ENABLED", "False").lower() == "true"),
    "database_url": os.getenv("DATABASE_URL", "postgresql://user:password@localhost/trading_bot"),
    "redis_enabled": bool(os.getenv("REDIS_ENABLED", "False").lower() == "true"),
    "redis_host": os.getenv("REDIS_HOST", "localhost"),
    "redis_port": int(os.getenv("REDIS_PORT", "6379")),
    "authentication_required": bool(os.getenv("AUTHENTICATION_REQUIRED", "True").lower() == "true"),
    "jwt_secret_key": os.getenv("JWT_SECRET_KEY", "your-secret-key-here"),
    "rate_limiting_enabled": bool(os.getenv("RATE_LIMITING_ENABLED", "True").lower() == "true"),
    "api_rate_limit_requests": int(os.getenv("API_RATE_LIMIT_REQUESTS", "100")),
    "api_rate_limit_window": int(os.getenv("API_RATE_LIMIT_WINDOW", "60")),

    # Phase 4: Production Monitoring
    "metrics_enabled": bool(os.getenv("METRICS_ENABLED", "True").lower() == "true"),
    "metrics_port": int(os.getenv("METRICS_PORT", "9090")),
    "health_checks_enabled": bool(os.getenv("HEALTH_CHECKS_ENABLED", "True").lower() == "true"),
    "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
    "alerting_enabled": bool(os.getenv("ALERTING_ENABLED", "False").lower() == "true"),
    "email_notifications_enabled": bool(os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "False").lower() == "true"),
    "slack_notifications_enabled": bool(os.getenv("SLACK_NOTIFICATIONS_ENABLED", "False").lower() == "true"),
    "smtp_server": os.getenv("SMTP_SERVER", "localhost"),
    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
    "smtp_username": os.getenv("SMTP_USERNAME", ""),
    "smtp_password": os.getenv("SMTP_PASSWORD", ""),
    "alert_recipients": [],
    "slack_webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
    "logging_level": os.getenv("LOGGING_LEVEL", "INFO"),
    "structured_logging_enabled": bool(os.getenv("STRUCTURED_LOGGING_ENABLED", "True").lower() == "true"),
    "performance_monitoring_enabled": bool(os.getenv("PERFORMANCE_MONITORING_ENABLED", "True").lower() == "true"),
    "log_aggregation_enabled": bool(os.getenv("LOG_AGGREGATION_ENABLED", "False").lower() == "true"),
    "elasticsearch_url": os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),

    # Trading Safety Settings
    "trading_safety_checks_enabled": bool(os.getenv("TRADING_SAFETY_CHECKS_ENABLED", "True").lower() == "true"),
    "require_risk_acknowledgment": bool(os.getenv("REQUIRE_RISK_ACKNOWLEDGMENT", "True").lower() == "true"),
    "min_wallet_balance_sol": float(os.getenv("MIN_WALLET_BALANCE_SOL", "0.01")),
    "startup_safety_warning_enabled": bool(os.getenv("STARTUP_SAFETY_WARNING_ENABLED", "True").lower() == "true"),
    "live_trading_confirmation_required": bool(os.getenv("LIVE_TRADING_CONFIRMATION_REQUIRED", "True").lower() == "true"),
}

# Config file path
CONFIG_DIR = Path.home() / ".solana-trading-bot"
CONFIG_FILE = CONFIG_DIR / "config.json"


def ensure_config_dir() -> None:
    """Ensure the configuration directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    wallet_dir = Path(DEFAULT_CONFIG["wallet_path"])
    wallet_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """
    Load configuration from the config file.
    If the file doesn't exist, create it with default values.
    """
    ensure_config_dir()

    if not CONFIG_FILE.exists():
        # Create default config file
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG

    # Load existing config
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # Update with any missing default values
    updated = False
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
            updated = True

    # Save updated config if needed
    if updated:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)

    return config


def update_config(key: str, value: Any) -> Dict[str, Any]:
    """
    Update a specific configuration value.

    Args:
        key: The configuration key to update
        value: The new value

    Returns:
        The updated configuration dictionary
    """
    config = load_config()
    config[key] = value

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

    return config


def get_config_value(key: str, default: Optional[Any] = None) -> Any:
    """
    Get a specific configuration value.

    Args:
        key: The configuration key to retrieve
        default: Default value if the key doesn't exist

    Returns:
        The configuration value or default if not found
    """
    config = load_config()
    return config.get(key, default)


# Initialize configuration on module import
CONFIG = load_config()
