#!/usr/bin/env python3
"""
Solana Memecoin Trading Bot - Main Entry Point

This is the main entry point for the Solana Memecoin Trading Bot.
It initializes the application and starts the CLI interface.
"""

import os
import sys
import logging
import argparse
import traceback
from pathlib import Path
import qrcode

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from config import load_config, ensure_config_dir, get_config_value
from src.utils.logging_utils import init_logging, get_logger
from src.cli.cli_interface import run_cli
from src.cli.cli_functions import register_all_functions
from src.solana.solana_interact import solana_client
from src.wallet.wallet import wallet_manager
from src.wallet.hardware_wallet import hardware_wallet_manager
from src.trading.position_manager import position_manager
from src.trading.jupiter_api import jupiter_api
from src.trading.helius_api import helius_api
from src.trading.pool_monitor import pool_monitor
from src.trading.copy_trading import copy_trading
from src.trading.sentiment_analysis import sentiment_analyzer
from src.trading.strategy_generator import strategy_generator
from src.trading.technical_analysis import technical_analyzer
from src.trading.price_alerts import price_alert_manager
from src.trading.wallet_monitor import wallet_monitor
from src.trading.limit_orders import limit_order_manager
from src.trading.dca_orders import dca_manager
from src.trading.auto_buy import auto_buy_manager
from src.trading.token_analytics import token_analytics
from src.charts.chart_generator import chart_generator
from src.mobile.mobile_app import mobile_app_manager
from src.ml.token_evaluator import token_evaluator
from src.community.strategy_sharing import strategy_sharing
from src.wallet.withdraw import withdraw_manager
from src.trading.risk_manager import risk_manager
from src.solana.gas_optimizer import gas_optimizer
from src.utils.performance_tracker import performance_tracker
from src.utils.performance_optimizer import performance_optimizer
from src.trading.strategy_engine import strategy_engine
from src.security.security_manager import security_manager
from src.cli.refinement_functions import check_auto_refinement

# Initialize enhanced logging
init_logging()

# Get logger for this module
logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Solana Memecoin Trading Bot")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--max-workers", type=int, help="Maximum number of worker threads/processes")
    parser.add_argument("--cache-ttl", type=int, help="Cache TTL in seconds")
    parser.add_argument("--train-ml", action="store_true", help="Train ML models at startup")
    parser.add_argument("--setup-mfa", action="store_true", help="Set up multi-factor authentication")
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_args()

    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Ensure config directory exists
    ensure_config_dir()

    # Load configuration
    config = load_config()
    logger.info(f"Configuration loaded: {config}")

    # Initialize performance optimizer with custom settings if provided
    if args.max_workers or args.cache_ttl:
        performance_optimizer = PerformanceOptimizer(
            max_workers=args.max_workers or 4,
            cache_ttl=args.cache_ttl or 300
        )
        logger.info(f"Performance optimizer initialized with {args.max_workers or 4} workers")

    # Set up MFA if requested
    if args.setup_mfa:
        logger.info("Setting up multi-factor authentication...")
        success, qr_data = security_manager.setup_mfa("admin")
        if success:
            logger.info("MFA setup successful. Please scan the QR code with your authenticator app.")
            # Generate and display QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(qr_data)
            qr.make(fit=True)
            qr.print_ascii()
        else:
            logger.error("Failed to set up MFA")

    # Initialize position monitoring with parallel processing
    position_manager.ensure_monitoring_running()
    logger.info(f"Loaded {len(position_manager.positions)} positions for monitoring")

    # Initialize pool monitoring if enabled with parallel processing
    if get_config_value("sniping_enabled", False):
        pool_monitor.start_monitoring()
        logger.info("Pool monitoring started for token sniping")

    # Initialize copy trading if enabled
    if get_config_value("copy_trading_enabled", False):
        copy_trading.set_enabled(True)
        logger.info("Copy trading enabled")

    # Initialize sentiment analysis if enabled with caching
    if get_config_value("sentiment_analysis_enabled", False):
        sentiment_analyzer.set_enabled(True)
        logger.info("Sentiment analysis enabled")

    # Initialize strategy generator with parallel processing
    strategies = strategy_generator.list_strategies()
    logger.info(f"Loaded {len(strategies)} trading strategies")

    # Initialize technical analysis if enabled with caching
    if get_config_value("technical_analysis_enabled", False):
        technical_analyzer.set_enabled(True)
        logger.info("Technical analysis enabled")

    # Initialize chart generator if enabled
    if get_config_value("charts_enabled", False):
        chart_generator.set_enabled(True)
        logger.info("Chart generation enabled")

    # Initialize hardware wallet support if enabled
    if get_config_value("hardware_wallet_enabled", False):
        hardware_wallet_manager.set_enabled(True)
        logger.info("Hardware wallet support enabled")

    # Initialize mobile app integration if enabled
    if get_config_value("mobile_app_enabled", False):
        mobile_app_manager.set_enabled(True)
        logger.info("Mobile app integration enabled")

    # Initialize ML token evaluation if enabled with caching
    if get_config_value("ml_evaluation_enabled", False):
        token_evaluator.set_enabled(True)
        logger.info("ML token evaluation enabled")

    # Initialize community strategy sharing if enabled
    if get_config_value("strategy_sharing_enabled", False):
        strategy_sharing.set_enabled(True)
        logger.info("Community strategy sharing enabled")

    # Initialize price alerts if enabled with caching
    if get_config_value("price_alerts_enabled", False):
        price_alert_manager.set_enabled(True)
        logger.info("Price alerts enabled")

    # Initialize wallet monitoring if enabled with parallel processing
    if get_config_value("wallet_monitoring_enabled", False):
        wallet_monitor.set_enabled(True)
        logger.info("External wallet monitoring enabled")

    # Initialize limit orders if enabled with caching
    if get_config_value("limit_orders_enabled", False):
        limit_order_manager.set_enabled(True)
        logger.info("Limit orders enabled")

    # Initialize DCA orders if enabled with caching
    if get_config_value("dca_enabled", False):
        dca_manager.set_enabled(True)
        logger.info("DCA orders enabled")

    # Initialize auto-buy if enabled with caching
    if get_config_value("auto_buy_enabled", False):
        auto_buy_manager.set_enabled(True)
        logger.info("Auto-buy enabled")

    # Initialize token analytics if enabled with caching
    if get_config_value("token_analytics_enabled", False):
        token_analytics.set_enabled(True)
        logger.info("Token analytics enabled")

    # Initialize strategy engine if enabled
    if get_config_value("strategy_engine_enabled", False):
        logger.info("Strategy engine enabled")
        # Train ML models if requested
        if args.train_ml:
            logger.info("Training ML models...")
            # Get training data from performance tracker
            training_data = performance_tracker.get_training_data()
            if training_data:
                for strategy_name in strategy_generator.list_strategies():
                    if strategy_engine.train_ml_model(strategy_name, training_data):
                        logger.info(f"Successfully trained ML model for {strategy_name}")
                    else:
                        logger.warning(f"Failed to train ML model for {strategy_name}")
            else:
                logger.warning("No training data available for ML models")

    # Initialize auto-refinement if enabled
    if get_config_value("auto_refinement_enabled", False):
        logger.info("Auto-refinement enabled")
        # Check if refinement is due
        refinement_checks = performance_tracker.check_auto_refinement()
        if refinement_checks["risk"] or refinement_checks["gas"]:
            logger.info("Auto-refinement checks scheduled to run at startup")

    # Log initialization complete with system info
    import platform
    logger.info(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    logger.info(f"Python: {platform.python_version()}")
    logger.info("Solana Memecoin Trading Bot initialized successfully")

    try:
        # Run auto-refinement if scheduled
        if get_config_value("auto_refinement_enabled", False):
            refinement_checks = performance_tracker.check_auto_refinement()
            if refinement_checks["risk"]:
                logger.info("Running scheduled risk parameter refinement...")
                risk_data = performance_tracker.get_risk_refinement_data()
                if risk_data.get("trade_count", 0) >= 5:
                    result = risk_manager.refine_risk_parameters(risk_data)
                    if result.get("success", False):
                        logger.info("Risk parameter refinement completed successfully")
                        performance_tracker.record_refinement("risk")
                    else:
                        logger.warning(f"Risk parameter refinement failed: {result.get('reason', 'Unknown error')}")
                else:
                    logger.info(f"Not enough trade data for risk refinement (found {risk_data.get('trade_count', 0)}, need at least 5)")

            if refinement_checks["gas"]:
                logger.info("Running scheduled gas parameter refinement...")
                gas_data = performance_tracker.get_gas_refinement_data()
                if gas_data.get("transaction_count", 0) >= 10:
                    result = gas_optimizer.refine_fee_parameters(gas_data.get("transactions", []))
                    if result.get("success", False):
                        logger.info("Gas parameter refinement completed successfully")
                        performance_tracker.record_refinement("gas")
                    else:
                        logger.warning(f"Gas parameter refinement failed: {result.get('reason', 'Unknown error')}")
                else:
                    logger.info(f"Not enough transaction data for gas refinement (found {gas_data.get('transaction_count', 0)}, need at least 10)")

        # Start the CLI
        logger.info("Starting Solana Memecoin Trading Bot CLI")
        register_all_functions()
        run_cli()
    finally:
        logger.info("Shutting down Solana Memecoin Trading Bot...")

        # Stop all monitoring when the application exits
        position_manager.stop_monitoring_thread()
        logger.info("Position monitoring stopped")

        # Stop pool monitoring if it was started
        if get_config_value("sniping_enabled", False):
            pool_monitor.stop_monitoring()
            logger.info("Pool monitoring stopped")

        # Stop copy trading if it was enabled
        if get_config_value("copy_trading_enabled", False):
            copy_trading.set_enabled(False)
            logger.info("Copy trading disabled")

        # Stop sentiment analysis if it was enabled
        if get_config_value("sentiment_analysis_enabled", False):
            sentiment_analyzer.stop_monitoring_thread()
            logger.info("Sentiment analysis stopped")

        # Stop technical analysis if it was enabled
        if get_config_value("technical_analysis_enabled", False):
            technical_analyzer.stop_monitoring_thread()
            logger.info("Technical analysis stopped")

        # Stop ML token evaluation if it was enabled
        if get_config_value("ml_evaluation_enabled", False):
            token_evaluator.stop_monitoring_thread()
            logger.info("ML token evaluation stopped")

        # Stop price alerts if it was enabled
        if get_config_value("price_alerts_enabled", False):
            price_alert_manager.stop_monitoring_thread()
            logger.info("Price alerts stopped")

        # Shutdown performance optimizer
        performance_optimizer.shutdown()
        logger.info("Performance optimizer shut down")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        # Enhanced error handling with full traceback
        error_msg = f"Unhandled exception: {e}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Print error to console for user visibility
        print(f"\nERROR: {error_msg}")
        print("\nPlease check the log file for more details.")

        sys.exit(1)
