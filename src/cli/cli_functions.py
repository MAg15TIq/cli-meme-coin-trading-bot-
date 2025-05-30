"""
Functions for the CLI interface of the Solana Memecoin Trading Bot.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.box import Box, ROUNDED

from src.utils.logging_utils import get_logger
from config import get_config_value, update_config
from src.solana.solana_interact import solana_client
from src.wallet.wallet import wallet_manager
from src.trading.position_manager import position_manager
from src.trading.jupiter_api import jupiter_api
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
from src.trading.enhanced_portfolio_manager import enhanced_portfolio_manager
from src.trading.advanced_alert_system import advanced_alert_system
from src.solana.gas_optimizer import gas_optimizer, TransactionType, TransactionPriority
from src.utils.performance_tracker import performance_tracker

# Import the CLI interface to register functions
from src.cli.cli_interface import register_function, console, bot_running, wallet_connected, wallet_pubkey, current_keypair
from src.cli.enhanced_features_cli import (
    enhanced_copy_trading_menu, enhanced_portfolio_menu, advanced_alerts_menu,
    smart_copy_trading_menu, multi_dex_hunting_menu
)

# Get logger for this module
logger = get_logger(__name__)

# Path to log file
LOG_FILE = os.path.join(get_config_value("log_dir", os.path.expanduser("~/.solana-trading-bot/logs")), "trading_bot.log")

# Number of log lines to display
LOG_LINES = 20

def start_bot():
    """Start the bot's monitoring and trading functionality."""
    global bot_running

    if bot_running:
        console.print("[bold yellow]Bot is already running![/bold yellow]")
        input("Press Enter to continue...")
        return True

    console.print("[bold green]Starting bot...[/bold green]")

    # Initialize position monitoring
    position_manager.ensure_monitoring_running()
    console.print(f"+ Position monitoring started with {len(position_manager.positions)} positions")

    # Initialize pool monitoring if enabled
    if get_config_value("sniping_enabled", False):
        pool_monitor.start_monitoring()
        console.print("+ Pool monitoring started for token sniping")

    # Initialize copy trading if enabled
    if get_config_value("copy_trading_enabled", False):
        copy_trading.set_enabled(True)
        console.print("+ Copy trading enabled")

    # Initialize sentiment analysis if enabled
    if get_config_value("sentiment_analysis_enabled", False):
        sentiment_analyzer.set_enabled(True)
        console.print("+ Sentiment analysis enabled")

    # Initialize technical analysis if enabled
    if get_config_value("technical_analysis_enabled", False):
        technical_analyzer.set_enabled(True)
        console.print("+ Technical analysis enabled")

    # Initialize price alerts if enabled
    if get_config_value("price_alerts_enabled", False):
        price_alert_manager.set_enabled(True)
        console.print("+ Price alerts enabled")

    # Initialize wallet monitoring if enabled
    if get_config_value("wallet_monitoring_enabled", False):
        wallet_monitor.set_enabled(True)
        console.print("✓ External wallet monitoring enabled")

    # Initialize limit orders if enabled
    if get_config_value("limit_orders_enabled", False):
        limit_order_manager.set_enabled(True)
        console.print("✓ Limit orders enabled")

    # Initialize DCA orders if enabled
    if get_config_value("dca_enabled", False):
        dca_manager.set_enabled(True)
        console.print("✓ DCA orders enabled")

    # Initialize auto-buy if enabled
    if get_config_value("auto_buy_enabled", False):
        auto_buy_manager.set_enabled(True)
        console.print("✓ Auto-buy enabled")

    # Initialize token analytics if enabled
    if get_config_value("token_analytics_enabled", False):
        token_analytics.set_enabled(True)
        console.print("✓ Token analytics enabled")

    # Initialize enhanced portfolio management if enabled
    if get_config_value("enhanced_portfolio_enabled", False):
        enhanced_portfolio_manager.set_enabled(True)
        console.print("✓ Enhanced portfolio management enabled")

    # Initialize advanced alert system if enabled
    if get_config_value("advanced_alerts_enabled", False):
        advanced_alert_system.set_enabled(True)
        console.print("✓ Advanced alert system enabled")

    # Initialize smart copy trading discovery if enabled
    if get_config_value("smart_copy_discovery_enabled", False):
        from src.trading.smart_wallet_discovery import smart_wallet_discovery
        smart_wallet_discovery.set_enabled(True)
        console.print("✓ Smart wallet discovery enabled")

    # Initialize multi-DEX monitoring if enabled
    if get_config_value("multi_dex_enabled", False):
        from src.trading.multi_dex_monitor import multi_dex_monitor
        multi_dex_monitor.set_enabled(True)
        console.print("✓ Multi-DEX monitoring enabled")

    bot_running = True
    logger.info("Bot started successfully")
    console.print("[bold green]Bot started successfully![/bold green]")
    input("Press Enter to continue...")
    return True


def stop_bot():
    """Stop the bot's monitoring and trading functionality."""
    global bot_running

    if not bot_running:
        console.print("[bold yellow]Bot is not running![/bold yellow]")
        input("Press Enter to continue...")
        return True

    console.print("[bold yellow]Stopping bot...[/bold yellow]")

    # Stop position monitoring
    position_manager.stop_monitoring_thread()
    console.print("+ Position monitoring stopped")

    # Stop pool monitoring if it was started
    if get_config_value("sniping_enabled", False):
        pool_monitor.stop_monitoring()
        console.print("+ Pool monitoring stopped")

    # Stop copy trading if it was enabled
    if get_config_value("copy_trading_enabled", False):
        copy_trading.set_enabled(False)
        console.print("+ Copy trading disabled")

    # Stop sentiment analysis if it was enabled
    if get_config_value("sentiment_analysis_enabled", False):
        sentiment_analyzer.stop_monitoring_thread()
        console.print("+ Sentiment analysis stopped")

    # Stop technical analysis if it was enabled
    if get_config_value("technical_analysis_enabled", False):
        technical_analyzer.stop_monitoring_thread()
        console.print("+ Technical analysis stopped")

    # Stop price alerts if it was enabled
    if get_config_value("price_alerts_enabled", False):
        price_alert_manager.stop_monitoring_thread()
        console.print("+ Price alerts stopped")

    # Stop wallet monitoring if it was enabled
    if get_config_value("wallet_monitoring_enabled", False):
        wallet_monitor.stop_monitoring_thread()
        console.print("+ External wallet monitoring stopped")

    # Stop limit orders if it was enabled
    if get_config_value("limit_orders_enabled", False):
        limit_order_manager.stop_monitoring_thread()
        console.print("+ Limit orders stopped")

    # Stop DCA orders if it was enabled
    if get_config_value("dca_enabled", False):
        dca_manager.stop_monitoring_thread()
        console.print("+ DCA orders stopped")

    # Stop token analytics if it was enabled
    if get_config_value("token_analytics_enabled", False):
        token_analytics.stop_monitoring()
        console.print("+ Token analytics stopped")

    # Stop enhanced portfolio management if it was enabled
    if get_config_value("enhanced_portfolio_enabled", False):
        enhanced_portfolio_manager.stop_monitoring()
        console.print("+ Enhanced portfolio management stopped")

    # Stop advanced alert system if it was enabled
    if get_config_value("advanced_alerts_enabled", False):
        advanced_alert_system.stop_monitoring()
        console.print("+ Advanced alert system stopped")

    # Stop smart copy trading discovery if it was enabled
    if get_config_value("smart_copy_discovery_enabled", False):
        from src.trading.smart_wallet_discovery import smart_wallet_discovery
        smart_wallet_discovery.set_enabled(False)
        console.print("+ Smart wallet discovery stopped")

    # Stop multi-DEX monitoring if it was enabled
    if get_config_value("multi_dex_enabled", False):
        from src.trading.multi_dex_monitor import multi_dex_monitor
        multi_dex_monitor.set_enabled(False)
        console.print("+ Multi-DEX monitoring stopped")

    bot_running = False
    logger.info("Bot stopped successfully")
    console.print("[bold green]Bot stopped successfully![/bold green]")
    input("Press Enter to continue...")
    return True


def show_dashboard():
    """Display the dashboard with bot status and positions."""
    console.print(Panel(
        "[bold white]Real-time Trading Dashboard[/bold white]",
        title="[bold cyan]DASHBOARD[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))

    # Get current time
    current_time = datetime.now().strftime("%H:%M:%S UTC")

    # Status section with improved layout
    status_content = [
        "[bold white]STATUS:[/bold white] " +
        ("[bold green]* RUNNING[/bold green]" if bot_running else "[bold red]* STOPPED[/bold red]"),
        "",
        "[bold white]NETWORK:[/bold white] [cyan]" + get_config_value("network", "mainnet-beta").upper() + "[/cyan]",
        "[bold white]RPC:[/bold white]     [cyan]" + get_config_value("rpc_url", "api.mainnet-beta.solana.com") + "[/cyan] [green]OK[/green]",
        "[bold white]WALLET:[/bold white]   " + (f"[green]{wallet_pubkey[:6]}...{wallet_pubkey[-6:]}[/green]" if wallet_connected else "[yellow]NOT CONNECTED[/yellow]")
    ]

    status_panel = Panel(
        "\n".join(status_content),
        title=f"[bold cyan]SYSTEM STATUS[/bold cyan] [dim]({current_time})[/dim]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    )
    console.print(status_panel)

    # Positions section with improved styling
    positions = position_manager.get_all_positions()

    if positions:
        positions_table = Table(
            show_header=True,
            header_style="bold white",
            box=ROUNDED,
            border_style="green",
            padding=(0, 1)
        )
        positions_table.add_column("TOKEN", style="magenta")
        positions_table.add_column("AMOUNT", style="cyan", justify="right")
        positions_table.add_column("ENTRY", style="yellow", justify="right")
        positions_table.add_column("CURRENT", style="green", justify="right")
        positions_table.add_column("P/L (SOL)", style="cyan", justify="right")
        positions_table.add_column("P/L (%)", style="bold white", justify="right")

        total_pl_pct = 0
        total_pl_sol = 0
        total_value = 0

        for position in positions:
            pnl = position.get_pnl()
            total_pl_pct += pnl

            # Calculate P/L in SOL
            entry_value = position.amount * position.entry_price
            current_value = position.amount * position.current_price
            total_value += current_value
            pl_sol = current_value - entry_value
            total_pl_sol += pl_sol

            # Format values
            pl_color = "green" if pl_sol >= 0 else "red"
            pl_sign = "+" if pl_sol >= 0 else ""

            positions_table.add_row(
                f"[bold magenta]{position.token_symbol}[/bold magenta]",
                f"[cyan]{position.amount:,.0f}[/cyan]",
                f"[yellow]{position.entry_price:.8f}[/yellow]",
                f"[green]{position.current_price:.8f}[/green]",
                f"[{pl_color}]{pl_sign}{pl_sol:.3f}[/{pl_color}]",
                f"[{pl_color}]{pl_sign}{pnl:.2f}%[/{pl_color}]"
            )

        # Add summary row
        avg_pl_pct = total_pl_pct / len(positions) if positions else 0
        summary_color = "green" if total_pl_sol >= 0 else "red"
        summary_sign = "+" if total_pl_sol >= 0 else ""

        positions_table.add_section()
        positions_table.add_row(
            "[bold white]TOTAL[/bold white]",
            "",
            "",
            f"[bold green]{total_value:.3f}[/bold green]",
            f"[bold {summary_color}]{summary_sign}{total_pl_sol:.3f}[/bold {summary_color}]",
            f"[bold {summary_color}]{summary_sign}{avg_pl_pct:.2f}%[/bold {summary_color}]"
        )

        positions_panel = Panel(
            positions_table,
            title=f"[bold green]ACTIVE POSITIONS[/bold green] [white]({len(positions)})[/white]",
            border_style="green",
            box=ROUNDED,
            padding=(1, 1)
        )
        console.print(positions_panel)
    else:
        console.print(Panel(
            "[bold white]No active trading positions[/bold white]",
            title="[bold yellow]POSITIONS[/bold yellow]",
            border_style="yellow",
            box=ROUNDED,
            padding=(1, 1)
        ))

    # Activity log section
    log_entries = [
        ("14:32:10", "[bold green]+ BUY[/bold green]", "[magenta]PEPEWHIHAT[/magenta]", "1.5M @ 0.0000000", "(Cost: 1.20 SOL)"),
        ("14:35:55", "[bold red]- SELL[/bold red]", "[magenta]SILLY[/magenta]", "250K @ 0.0001500", "(Return: 0.95 SOL) [red](-21%)[/red]"),
        ("14:38:01", "[dim]i INFO[/dim]", "Scanning for new pairs...", "", ""),
        ("14:40:15", "[bold green]+ BUY[/bold green]", "[magenta]MONIE[/magenta]", "5.0M @ 0.0000005", "(Cost: 2.50 SOL)"),
        ("14:41:30", "[yellow]! WARN[/yellow]", "High network congestion. Fees increased.", "", "")
    ]

    log_table = Table(
        show_header=True,
        header_style="bold white",
        box=ROUNDED,
        border_style="blue",
        padding=(0, 1)
    )
    log_table.add_column("TIME", style="dim", width=10)
    log_table.add_column("ACTION", width=10)
    log_table.add_column("TOKEN", style="magenta", width=15)
    log_table.add_column("DETAILS", style="cyan")
    log_table.add_column("RESULT", style="green")

    for entry in log_entries:
        log_table.add_row(*entry)

    log_panel = Panel(
        log_table,
        title="[bold blue]RECENT ACTIVITY[/bold blue]",
        border_style="blue",
        box=ROUNDED,
        padding=(1, 1)
    )
    console.print(log_panel)

    # Add a footer with helpful information
    footer = Panel(
        "[bold white]Press any key to return to main menu[/bold white]",
        border_style="dim",
        box=ROUNDED,
        padding=(0, 1),
        width=40
    )
    console.print(footer)

    input()
    return True


def show_settings():
    """Display and modify bot settings."""
    console.print(Panel(
        "[bold white]Configure Trading Bot Settings[/bold white]",
        title="[bold cyan]SETTINGS[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))

    while True:
        # Create settings table with improved styling
        settings_table = Table(
            show_header=True,
            header_style="bold white",
            box=ROUNDED,
            border_style="cyan",
            padding=(0, 1)
        )
        settings_table.add_column("#", style="cyan", width=5, justify="center")
        settings_table.add_column("SETTING", style="yellow")
        settings_table.add_column("VALUE", style="green")
        settings_table.add_column("DESCRIPTION", style="dim white")

        # Add settings to the table
        settings = [
            ("1", "RPC URL", get_config_value("rpc_url", "https://api.mainnet-beta.solana.com"), "Solana RPC endpoint"),
            ("2", "Slippage", f"{get_config_value('slippage_bps', 50) / 100}%", "Trading slippage tolerance"),
            ("3", "Stop Loss", f"{get_config_value('stop_loss_percentage', 5.0)}%", "Default stop loss percentage"),
            ("4", "Take Profit", f"{get_config_value('take_profit_percentage', 20.0)}%", "Default take profit percentage"),
            ("5", "Sniping Enabled", str(get_config_value("sniping_enabled", False)), "Enable token sniping"),
            ("6", "Copy Trading", str(get_config_value("copy_trading_enabled", False)), "Enable copy trading"),
            ("7", "Sentiment Analysis", str(get_config_value("sentiment_analysis_enabled", False)), "Enable sentiment analysis"),
            ("8", "Technical Analysis", str(get_config_value("technical_analysis_enabled", False)), "Enable technical analysis"),
            ("9", "Log Level", get_config_value("log_level", "INFO"), "Logging verbosity"),
            ("A", "Auto-Buy", str(get_config_value("auto_buy_enabled", False)), "Enable auto-buy"),
            ("B", "Price Alerts", str(get_config_value("price_alerts_enabled", False)), "Enable price alerts"),
            ("C", "Wallet Monitoring", str(get_config_value("wallet_monitoring_enabled", False)), "Enable wallet monitoring"),
            ("D", "Limit Orders", str(get_config_value("limit_orders_enabled", False)), "Enable limit orders"),
            ("E", "DCA Orders", str(get_config_value("dca_enabled", False)), "Enable DCA orders"),
            ("F", "Token Analytics", str(get_config_value("token_analytics_enabled", False)), "Enable token analytics"),
        ]

        for setting in settings:
            settings_table.add_row(*setting)

        console.print(settings_table)
        console.print("\n[bold]Enter setting number/letter to change, or 'Q' to return to main menu:[/bold] ", end="")

        choice = input().strip().upper()

        if choice == "Q":
            break

        # Find the selected setting
        selected_setting = None
        for setting in settings:
            if setting[0].upper() == choice:
                selected_setting = setting
                break

        if selected_setting:
            setting_key = selected_setting[1]
            current_value = selected_setting[2]

            console.print(f"\nChanging setting: [yellow]{setting_key}[/yellow]")
            console.print(f"Current value: [green]{current_value}[/green]")
            console.print("Enter new value: ", end="")

            new_value = input().strip()

            # Map the setting name to the config key
            config_key_map = {
                "RPC URL": "rpc_url",
                "Slippage": "slippage_bps",
                "Stop Loss": "stop_loss_percentage",
                "Take Profit": "take_profit_percentage",
                "Sniping Enabled": "sniping_enabled",
                "Copy Trading": "copy_trading_enabled",
                "Sentiment Analysis": "sentiment_analysis_enabled",
                "Technical Analysis": "technical_analysis_enabled",
                "Log Level": "log_level",
                "Auto-Buy": "auto_buy_enabled",
                "Price Alerts": "price_alerts_enabled",
                "Wallet Monitoring": "wallet_monitoring_enabled",
                "Limit Orders": "limit_orders_enabled",
                "DCA Orders": "dca_enabled",
                "Token Analytics": "token_analytics_enabled",
            }

            config_key = config_key_map.get(setting_key)

            if config_key:
                # Convert value to appropriate type
                if config_key.endswith("_enabled"):
                    # Boolean settings
                    if new_value.lower() in ["true", "yes", "y", "1"]:
                        new_value = True
                    else:
                        new_value = False
                elif config_key in ["slippage_bps"]:
                    # Convert percentage to basis points
                    try:
                        new_value = float(new_value.strip("%")) * 100
                    except ValueError:
                        console.print("[bold red]Invalid value. Please enter a number.[/bold red]")
                        input("Press Enter to continue...")
                        continue
                elif config_key in ["stop_loss_percentage", "take_profit_percentage"]:
                    # Percentage values
                    try:
                        new_value = float(new_value.strip("%"))
                    except ValueError:
                        console.print("[bold red]Invalid value. Please enter a number.[/bold red]")
                        input("Press Enter to continue...")
                        continue

                # Update the config
                update_config(config_key, new_value)
                console.print(f"[bold green]Setting updated: {setting_key} = {new_value}[/bold green]")

                # Apply the setting if needed
                if config_key == "sniping_enabled":
                    if new_value:
                        pool_monitor.start_monitoring()
                    else:
                        pool_monitor.stop_monitoring()
                elif config_key == "copy_trading_enabled":
                    copy_trading.set_enabled(new_value)
                elif config_key == "sentiment_analysis_enabled":
                    sentiment_analyzer.set_enabled(new_value)
                elif config_key == "technical_analysis_enabled":
                    technical_analyzer.set_enabled(new_value)
                elif config_key == "price_alerts_enabled":
                    price_alert_manager.set_enabled(new_value)
                elif config_key == "wallet_monitoring_enabled":
                    wallet_monitor.set_enabled(new_value)
                elif config_key == "limit_orders_enabled":
                    limit_order_manager.set_enabled(new_value)
                elif config_key == "dca_enabled":
                    dca_manager.set_enabled(new_value)
                elif config_key == "auto_buy_enabled":
                    auto_buy_manager.set_enabled(new_value)
                elif config_key == "token_analytics_enabled":
                    token_analytics.set_enabled(new_value)

                input("Press Enter to continue...")
            else:
                console.print("[bold red]Error: Could not map setting to config key.[/bold red]")
                input("Press Enter to continue...")
        else:
            console.print("[bold red]Invalid option. Please try again.[/bold red]")
            input("Press Enter to continue...")

    return True


def show_logs():
    """Display recent log entries."""
    global LOG_LINES
    console.print(Panel(
        "[bold white]System and Trading Activity Logs[/bold white]",
        title="[bold cyan]LOGS[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))

    # Check if log file exists
    if not os.path.exists(LOG_FILE):
        console.print(f"[bold red]Log file not found: {LOG_FILE}[/bold red]")
        input("Press Enter to continue...")
        return True

    try:
        # Read the last LOG_LINES lines from the log file
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            log_lines = lines[-LOG_LINES:] if len(lines) > LOG_LINES else lines

        # Create a table for the logs with improved styling
        log_table = Table(
            show_header=True,
            header_style="bold white",
            box=ROUNDED,
            border_style="blue",
            padding=(0, 1)
        )
        log_table.add_column("TIMESTAMP", style="dim", width=12)
        log_table.add_column("LEVEL", style="cyan", width=10)
        log_table.add_column("MESSAGE", style="white")

        # Parse and add log lines to the table
        for line in log_lines:
            try:
                # Simple parsing for log lines
                parts = line.strip().split(" ", 2)
                if len(parts) >= 3:
                    timestamp = parts[0]
                    level = parts[1]
                    message = parts[2]

                    # Color the level based on severity with improved styling
                    level_style = "green"
                    level_icon = "i"
                    if level == "WARNING":
                        level_style = "yellow"
                        level_icon = "!"
                    elif level == "ERROR" or level == "CRITICAL":
                        level_style = "bold red"
                        level_icon = "X"
                    elif level == "INFO":
                        level_style = "cyan"
                        level_icon = "i"
                    elif level == "DEBUG":
                        level_style = "dim"
                        level_icon = "D"

                    log_table.add_row(
                        timestamp,
                        f"[{level_style}]{level_icon} {level}[/{level_style}]",
                        message
                    )
                else:
                    # If we can't parse it, just show the raw line
                    log_table.add_row("", "", line.strip())
            except Exception as e:
                # If there's an error parsing, just show the raw line
                log_table.add_row("", "", line.strip())

        # Display the logs in a panel with improved styling
        log_panel = Panel(
            log_table,
            title=f"[bold blue]SYSTEM LOGS[/bold blue] [white](Last {len(log_lines)} entries)[/white]",
            border_style="blue",
            box=ROUNDED,
            padding=(1, 1)
        )
        console.print(log_panel)

        # Options with improved styling
        options_table = Table(show_header=False, box=None, padding=(0, 2))
        options_table.add_column("Key", style="cyan", width=5)
        options_table.add_column("Action", style="white")

        options_table.add_row("[bold cyan]1[/bold cyan]", "[white]Refresh logs[/white]")
        options_table.add_row("[bold cyan]2[/bold cyan]", "[white]View more lines[/white]")
        options_table.add_row("[bold cyan]Q[/bold cyan]", "[white]Return to main menu[/white]")

        options_panel = Panel(
            options_table,
            title="[bold cyan]OPTIONS[/bold cyan]",
            border_style="cyan",
            box=ROUNDED,
            padding=(1, 2),
            width=40
        )
        console.print(options_panel)

        console.print("[bold cyan]>[/bold cyan] ", end="")
        choice = input().strip().upper()

        if choice == "1":
            # Refresh logs (recursively call this function)
            return show_logs()
        elif choice == "2":
            # Increase the number of lines to show
            LOG_LINES += 20
            return show_logs()
        elif choice == "Q":
            return True
        else:
            console.print("[bold red]Invalid option. Returning to main menu.[/bold red]")
            input("Press Enter to continue...")
            return True

    except Exception as e:
        console.print(f"[bold red]Error reading log file: {e}[/bold red]")
        input("Press Enter to continue...")
        return True


def connect_wallet():
    """Connect to a Solana wallet."""
    global wallet_connected, wallet_pubkey, current_keypair

    console.print("[bold cyan]Connect Wallet[/bold cyan]\n")

    # Options for wallet connection
    console.print("[bold]Select wallet connection method:[/bold]")
    console.print("[cyan]1[/cyan] Connect existing wallet")
    console.print("[cyan]2[/cyan] Create new wallet")
    console.print("[cyan]3[/cyan] Import from private key")
    console.print("[cyan]Q[/cyan] Return to main menu")
    console.print("\n[bold]Enter choice:[/bold] ", end="")

    choice = input().strip().upper()

    if choice == "1":
        # List available wallets
        wallets = wallet_manager.list_wallets()

        if not wallets:
            console.print("[bold yellow]No wallets found. Please create a new wallet.[/bold yellow]")
            input("Press Enter to continue...")
            return connect_wallet()

        console.print("\n[bold]Available wallets:[/bold]")
        for i, wallet in enumerate(wallets, 1):
            console.print(f"[cyan]{i}[/cyan] {wallet}")

        console.print("\n[bold]Select wallet (number) or 'Q' to cancel:[/bold] ", end="")
        wallet_choice = input().strip().upper()

        if wallet_choice == "Q":
            return True

        try:
            wallet_index = int(wallet_choice) - 1
            if 0 <= wallet_index < len(wallets):
                selected_wallet = wallets[wallet_index]

                # Get password for wallet
                console.print("\n[bold]Enter wallet password:[/bold] ", end="")
                password = input().strip()

                try:
                    # Load the wallet
                    keypair = wallet_manager.load_wallet(selected_wallet, password)
                    wallet_pubkey = str(keypair.public_key)
                    current_keypair = keypair
                    wallet_connected = True

                    console.print(f"\n[bold green]Wallet connected: {wallet_pubkey[:6]}...{wallet_pubkey[-6:]}[/bold green]")
                    input("Press Enter to continue...")
                    return True
                except Exception as e:
                    console.print(f"\n[bold red]Error loading wallet: {e}[/bold red]")
                    input("Press Enter to continue...")
                    return True
            else:
                console.print("\n[bold red]Invalid wallet selection.[/bold red]")
                input("Press Enter to continue...")
                return True
        except ValueError:
            console.print("\n[bold red]Invalid input. Please enter a number.[/bold red]")
            input("Press Enter to continue...")
            return True

    elif choice == "2":
        # Create new wallet
        console.print("\n[bold]Creating new wallet...[/bold]")
        console.print("[bold]Enter a name for the wallet:[/bold] ", end="")
        wallet_name = input().strip()

        if not wallet_name:
            console.print("\n[bold red]Wallet name cannot be empty.[/bold red]")
            input("Press Enter to continue...")
            return True

        console.print("\n[bold]Enter a password to encrypt the wallet:[/bold] ", end="")
        password = input().strip()

        if not password:
            console.print("\n[bold red]Password cannot be empty.[/bold red]")
            input("Press Enter to continue...")
            return True

        console.print("\n[bold]Confirm password:[/bold] ", end="")
        confirm_password = input().strip()

        if password != confirm_password:
            console.print("\n[bold red]Passwords do not match.[/bold red]")
            input("Press Enter to continue...")
            return True

        try:
            # Create the wallet
            keypair = wallet_manager.create_wallet(wallet_name, password)
            wallet_pubkey = str(keypair.public_key)
            current_keypair = keypair
            wallet_connected = True

            console.print(f"\n[bold green]Wallet created and connected: {wallet_pubkey[:6]}...{wallet_pubkey[-6:]}[/bold green]")
            console.print("[bold yellow]IMPORTANT: Please back up your wallet! The private key is encrypted and stored locally.[/bold yellow]")
            input("Press Enter to continue...")
            return True
        except Exception as e:
            console.print(f"\n[bold red]Error creating wallet: {e}[/bold red]")
            input("Press Enter to continue...")
            return True

    elif choice == "3":
        # Import from private key
        console.print("\n[bold]Import wallet from private key[/bold]")
        console.print("[bold]Enter wallet name:[/bold] ", end="")
        wallet_name = input().strip()

        if not wallet_name:
            console.print("\n[bold red]Wallet name cannot be empty.[/bold red]")
            input("Press Enter to continue...")
            return True

        console.print("\n[bold]Enter private key (base58):[/bold] ", end="")
        private_key = input().strip()

        if not private_key:
            console.print("\n[bold red]Private key cannot be empty.[/bold red]")
            input("Press Enter to continue...")
            return True

        console.print("\n[bold]Enter a password to encrypt the wallet:[/bold] ", end="")
        password = input().strip()

        if not password:
            console.print("\n[bold red]Password cannot be empty.[/bold red]")
            input("Press Enter to continue...")
            return True

        console.print("\n[bold]Confirm password:[/bold] ", end="")
        confirm_password = input().strip()

        if password != confirm_password:
            console.print("\n[bold red]Passwords do not match.[/bold red]")
            input("Press Enter to continue...")
            return True

        try:
            # Import the wallet
            keypair = wallet_manager.import_wallet(wallet_name, private_key, password)
            wallet_pubkey = str(keypair.public_key)
            current_keypair = keypair
            wallet_connected = True

            console.print(f"\n[bold green]Wallet imported and connected: {wallet_pubkey[:6]}...{wallet_pubkey[-6:]}[/bold green]")
            input("Press Enter to continue...")
            return True
        except Exception as e:
            console.print(f"\n[bold red]Error importing wallet: {e}[/bold red]")
            input("Press Enter to continue...")
            return True

    elif choice == "Q":
        return True

    else:
        console.print("\n[bold red]Invalid option. Please try again.[/bold red]")
        input("Press Enter to continue...")
        return connect_wallet()


def buy_token():
    """Buy a token."""
    global wallet_connected

    if not wallet_connected:
        console.print("[bold red]Wallet not connected. Please connect a wallet first.[/bold red]")
        input("Press Enter to continue...")
        return True

    console.print("[bold cyan]Buy Token[/bold cyan]\n")

    console.print("[bold]Enter token address or symbol:[/bold] ", end="")
    token = input().strip()

    if not token:
        console.print("[bold red]Token cannot be empty.[/bold red]")
        input("Press Enter to continue...")
        return True

    console.print("\n[bold]Enter amount in SOL:[/bold] ", end="")
    amount_str = input().strip()

    try:
        amount = float(amount_str)
        if amount <= 0:
            raise ValueError("Amount must be greater than 0")
    except ValueError:
        console.print("[bold red]Invalid amount. Please enter a positive number.[/bold red]")
        input("Press Enter to continue...")
        return True

    # Confirm the transaction
    console.print(f"\n[bold]You are about to buy {token} for {amount} SOL.[/bold]")
    console.print("[bold]Confirm? (Y/N):[/bold] ", end="")
    confirm = input().strip().upper()

    if confirm != "Y":
        console.print("\n[bold yellow]Transaction cancelled.[/bold yellow]")
        input("Press Enter to continue...")
        return True

    console.print("\n[bold]Processing transaction...[/bold]")

    try:
        # Execute the buy
        result = position_manager.buy_token(
            token_mint=token,
            amount_sol=amount,
            token_symbol=token
        )

        if result["success"]:
            console.print(f"\n[bold green]Transaction successful![/bold green]")
            console.print(f"Bought {result.get('amount', 'unknown')} {token} for {amount} SOL")
        else:
            console.print(f"\n[bold red]Transaction failed: {result.get('error', 'Unknown error')}[/bold red]")

        input("Press Enter to continue...")
        return True
    except Exception as e:
        console.print(f"\n[bold red]Error executing transaction: {e}[/bold red]")
        input("Press Enter to continue...")
        return True


def sell_token():
    """Sell a token."""
    global wallet_connected

    if not wallet_connected:
        console.print("[bold red]Wallet not connected. Please connect a wallet first.[/bold red]")
        input("Press Enter to continue...")
        return True

    console.print("[bold cyan]Sell Token[/bold cyan]\n")

    # Get positions
    positions = position_manager.get_all_positions()

    if not positions:
        console.print("[bold yellow]No positions found. Nothing to sell.[/bold yellow]")
        input("Press Enter to continue...")
        return True

    # Display positions
    console.print("[bold]Your positions:[/bold]")
    for i, position in enumerate(positions, 1):
        console.print(f"[cyan]{i}[/cyan] {position.token_symbol} - {position.amount:,.0f} tokens (Value: {position.amount * position.current_price:.3f} SOL)")

    console.print("\n[bold]Select position to sell (number) or 'Q' to cancel:[/bold] ", end="")
    position_choice = input().strip().upper()

    if position_choice == "Q":
        return True

    try:
        position_index = int(position_choice) - 1
        if 0 <= position_index < len(positions):
            selected_position = positions[position_index]

            console.print(f"\n[bold]Selected: {selected_position.token_symbol}[/bold]")
            console.print(f"Amount: {selected_position.amount:,.0f} tokens")
            console.print(f"Current value: {selected_position.amount * selected_position.current_price:.3f} SOL")

            console.print("\n[bold]Sell percentage (1-100) or 'Q' to cancel:[/bold] ", end="")
            percentage_str = input().strip().upper()

            if percentage_str == "Q":
                return True

            try:
                percentage = float(percentage_str)
                if percentage <= 0 or percentage > 100:
                    raise ValueError("Percentage must be between 1 and 100")

                # Calculate amount to sell
                amount_to_sell = selected_position.amount * (percentage / 100)

                # Confirm the transaction
                console.print(f"\n[bold]You are about to sell {percentage}% of {selected_position.token_symbol} ({amount_to_sell:,.0f} tokens).[/bold]")
                console.print("[bold]Confirm? (Y/N):[/bold] ", end="")
                confirm = input().strip().upper()

                if confirm != "Y":
                    console.print("\n[bold yellow]Transaction cancelled.[/bold yellow]")
                    input("Press Enter to continue...")
                    return True

                console.print("\n[bold]Processing transaction...[/bold]")

                try:
                    # Execute the sell
                    result = position_manager.sell_token(
                        token_mint=selected_position.token_mint,
                        percentage=percentage
                    )

                    if result["success"]:
                        console.print(f"\n[bold green]Transaction successful![/bold green]")
                        console.print(f"Sold {amount_to_sell:,.0f} {selected_position.token_symbol} for {result.get('sol_amount', 0):.3f} SOL")
                    else:
                        console.print(f"\n[bold red]Transaction failed: {result.get('error', 'Unknown error')}[/bold red]")

                    input("Press Enter to continue...")
                    return True
                except Exception as e:
                    console.print(f"\n[bold red]Error executing transaction: {e}[/bold red]")
                    input("Press Enter to continue...")
                    return True
            except ValueError:
                console.print("\n[bold red]Invalid percentage. Please enter a number between 1 and 100.[/bold red]")
                input("Press Enter to continue...")
                return True
        else:
            console.print("\n[bold red]Invalid position selection.[/bold red]")
            input("Press Enter to continue...")
            return True
    except ValueError:
        console.print("\n[bold red]Invalid input. Please enter a number.[/bold red]")
        input("Press Enter to continue...")
        return True


def check_balance():
    """Check wallet balance."""
    global wallet_connected, wallet_pubkey

    if not wallet_connected:
        console.print("[bold red]Wallet not connected. Please connect a wallet first.[/bold red]")
        input("Press Enter to continue...")
        return True

    console.print("[bold cyan]Wallet Balance[/bold cyan]\n")

    try:
        # Get SOL balance
        # Using solana_client directly without importing PublicKey
        sol_balance = solana_client.get_balance(wallet_pubkey)

        console.print(f"[bold]Wallet:[/bold] {wallet_pubkey}")
        console.print(f"[bold]SOL Balance:[/bold] [green]{sol_balance:.5f} SOL[/green]")

        # Get token balances
        token_balances = solana_client.get_token_accounts_by_owner(wallet_pubkey)

        if token_balances:
            console.print("\n[bold]Token Balances:[/bold]")

            token_table = Table(show_header=True, header_style="bold")
            token_table.add_column("Token", style="magenta")
            token_table.add_column("Balance", style="cyan")
            token_table.add_column("Value (SOL)", style="green")

            for token in token_balances:
                token_symbol = token.get("symbol", token.get("mint", "Unknown")[:8])
                token_balance = token.get("balance", 0)
                token_value = token.get("value_sol", 0)

                token_table.add_row(
                    f"[magenta]{token_symbol}[/magenta]",
                    f"[cyan]{token_balance:,.0f}[/cyan]",
                    f"[green]{token_value:.3f}[/green]"
                )

            console.print(token_table)
        else:
            console.print("\n[bold yellow]No token balances found.[/bold yellow]")

        input("\nPress Enter to continue...")
        return True
    except Exception as e:
        console.print(f"\n[bold red]Error checking balance: {e}[/bold red]")
        input("Press Enter to continue...")
        return True


def exit_app():
    """Exit the application."""
    return False


def not_implemented(feature_name):
    """Display a message for features that are not yet implemented."""
    console.print(f"[bold yellow]The {feature_name} feature is not yet implemented.[/bold yellow]")
    input("Press Enter to continue...")
    return True


# Register all functions
def register_all_functions():
    """Register all CLI functions."""
    register_function("start_bot", start_bot)
    register_function("show_dashboard", show_dashboard)
    register_function("show_settings", show_settings)
    register_function("show_logs", show_logs)
    register_function("stop_bot", stop_bot)
    register_function("connect_wallet", connect_wallet)
    register_function("buy_token", buy_token)
    register_function("sell_token", sell_token)
    register_function("check_balance", check_balance)
    register_function("exit_app", exit_app)

    # Register withdraw and limit_orders functions
    register_function("withdraw", withdraw)
    register_function("limit_orders", limit_orders)

    # Register dca_orders as a placeholder function for now
    register_function("dca_orders", lambda: not_implemented("dca_orders"))

    # These functions would be implemented similarly to the ones above
    # For now, we'll just create placeholder functions that show "Not implemented yet"
    for func_name in ["price_alerts", "wallet_monitor", "auto_buy"]:
        register_function(func_name, lambda func_name=func_name: not_implemented(func_name))

    # Register token analytics function
    register_function("token_analytics", token_analytics_function)

    # Import and register risk management functions
    from src.cli.risk_functions import manage_risk, check_portfolio
    register_function("risk", manage_risk)
    register_function("portfolio", check_portfolio)

    # Import and register gas optimization functions
    from src.cli.gas_functions import manage_gas, check_fees
    register_function("gas", manage_gas)
    register_function("fees", check_fees)

    # Register rapid trading functions
    register_function("rapid", rapid_trading_menu)
    register_function("rbuy", rapid_buy)
    register_function("rsell", rapid_sell)
    register_function("rcancel", rapid_cancel)
    register_function("rstatus", rapid_status)


def token_analytics_function():
    """Display token analytics including social sentiment, holder metrics, and market data."""
    if not wallet_connected:
        console.print("[bold red]No wallet connected. Please connect a wallet first.[/bold red]")
        input("Press Enter to continue...")
        return True

    console.print(Panel(
        "[bold white]Token Analytics - Social Sentiment, Holder Metrics, and Market Data[/bold white]",
        title="[bold cyan]TOKEN ANALYTICS[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))

    # Check if token analytics is enabled
    if not get_config_value("token_analytics_enabled", False):
        console.print("[bold yellow]Token analytics is currently disabled.[/bold yellow]")
        console.print("Would you like to enable it? (y/n): ", end="")
        choice = input().strip().lower()
        if choice in ["y", "yes"]:
            token_analytics.set_enabled(True)
            update_config("token_analytics_enabled", True)
            console.print("[bold green]Token analytics enabled![/bold green]")
        else:
            console.print("[bold yellow]Token analytics remains disabled.[/bold yellow]")
            input("Press Enter to continue...")
            return True

    # Get positions to analyze
    positions = position_manager.get_all_positions()

    if not positions:
        console.print("[bold yellow]No positions found to analyze.[/bold yellow]")
        console.print("Would you like to enter a token address manually? (y/n): ", end="")
        choice = input().strip().lower()
        if choice not in ["y", "yes"]:
            input("Press Enter to continue...")
            return True

        console.print("\nEnter token mint address: ", end="")
        token_mint = input().strip()

        # Validate token mint address
        if not token_mint or len(token_mint) < 32:
            console.print("[bold red]Invalid token mint address.[/bold red]")
            input("Press Enter to continue...")
            return True

        # Get token info
        token_info = jupiter_api.get_token_info(token_mint)
        if not token_info:
            console.print("[bold red]Could not find token information.[/bold red]")
            input("Press Enter to continue...")
            return True

        # Analyze token
        console.print(f"\n[bold]Analyzing token: [magenta]{token_info.get('symbol', token_mint[:8])}[/magenta][/bold]")
        console.print("This may take a moment...\n")

        # Get token analytics
        analytics = token_analytics.get_token_analytics(token_mint, force_update=True)

        # Display analytics
        _display_token_analytics(token_mint, analytics)
    else:
        # Show menu of positions to analyze
        console.print("\n[bold]Select a token to analyze:[/bold]")
        for i, position in enumerate(positions, 1):
            console.print(f"[cyan]{i}.[/cyan] [magenta]{position.token_symbol}[/magenta] ({position.token_mint[:8]}...)")

        console.print("\nEnter token number (or 'a' for all): ", end="")
        choice = input().strip().lower()

        if choice == "a":
            # Analyze all tokens
            for position in positions:
                console.print(f"\n[bold]Analyzing token: [magenta]{position.token_symbol}[/magenta][/bold]")
                console.print("This may take a moment...\n")

                # Get token analytics
                analytics = token_analytics.get_token_analytics(position.token_mint, force_update=True)

                # Display analytics
                _display_token_analytics(position.token_mint, analytics)

                console.print("\nPress Enter to continue to next token or 'q' to quit: ", end="")
                if input().strip().lower() == "q":
                    break
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(positions):
                    position = positions[index]
                    console.print(f"\n[bold]Analyzing token: [magenta]{position.token_symbol}[/magenta][/bold]")
                    console.print("This may take a moment...\n")

                    # Get token analytics
                    analytics = token_analytics.get_token_analytics(position.token_mint, force_update=True)

                    # Display analytics
                    _display_token_analytics(position.token_mint, analytics)
                else:
                    console.print("[bold red]Invalid selection.[/bold red]")
            except ValueError:
                console.print("[bold red]Invalid input.[/bold red]")

    input("\nPress Enter to continue...")
    return True


def _display_token_analytics(token_mint: str, analytics: Dict[str, Any]) -> None:
    """Display token analytics in a formatted way."""
    if not analytics:
        console.print("[bold red]No analytics data available for this token.[/bold red]")
        return

    # Get token info
    token_name = analytics.get("token_name", "Unknown")
    token_symbol = analytics.get("token_symbol", "Unknown")

    # Create market data table
    market_table = Table(
        show_header=True,
        header_style="bold white",
        box=ROUNDED,
        border_style="green",
        padding=(0, 1)
    )
    market_table.add_column("METRIC", style="cyan")
    market_table.add_column("VALUE", style="green", justify="right")

    # Add market data
    market_table.add_row("Price", f"{analytics.get('price', 0):.8f} SOL")
    market_table.add_row("Market Cap", f"{analytics.get('market_cap', 0):,.2f} SOL")
    market_table.add_row("24h Volume", f"{analytics.get('volume_24h', 0):,.2f} SOL")
    market_table.add_row("Liquidity", f"{analytics.get('liquidity', 0):,.2f} SOL")
    market_table.add_row("24h Change", f"{analytics.get('price_change_24h', 0) * 100:.2f}%")
    market_table.add_row("Age", f"{analytics.get('age_days', 0)} days")

    # Create holder metrics table
    holder_table = Table(
        show_header=True,
        header_style="bold white",
        box=ROUNDED,
        border_style="blue",
        padding=(0, 1)
    )
    holder_table.add_column("METRIC", style="cyan")
    holder_table.add_column("VALUE", style="blue", justify="right")

    # Add holder metrics
    holder_table.add_row("Holder Count", f"{analytics.get('holder_count', 0):,}")
    holder_table.add_row("Top 10 Concentration", f"{analytics.get('top10_concentration', 0):.2f}%")
    holder_table.add_row("Avg Holding Time", f"{analytics.get('avg_holding_time_days', 0):.1f} days")

    # Create social sentiment table
    sentiment_table = Table(
        show_header=True,
        header_style="bold white",
        box=ROUNDED,
        border_style="magenta",
        padding=(0, 1)
    )
    sentiment_table.add_column("METRIC", style="cyan")
    sentiment_table.add_column("VALUE", style="magenta", justify="right")

    # Add social sentiment data
    twitter_sentiment = analytics.get("twitter_sentiment_24h", 0)
    sentiment_color = "green" if twitter_sentiment > 0 else "red" if twitter_sentiment < 0 else "yellow"
    sentiment_table.add_row("Twitter Mentions (24h)", f"{analytics.get('twitter_mentions_24h', 0):,}")
    sentiment_table.add_row("Twitter Sentiment", f"[{sentiment_color}]{twitter_sentiment:.2f}[/{sentiment_color}]")
    sentiment_table.add_row("Reddit Mentions (24h)", f"{analytics.get('reddit_mentions_24h', 0):,}")
    sentiment_table.add_row("Sentiment Trend", f"{analytics.get('sentiment_trend', 0):.2f}")

    # Create developer activity table
    dev_table = Table(
        show_header=True,
        header_style="bold white",
        box=ROUNDED,
        border_style="yellow",
        padding=(0, 1)
    )
    dev_table.add_column("METRIC", style="cyan")
    dev_table.add_column("VALUE", style="yellow", justify="right")

    # Add developer activity data
    dev_table.add_row("GitHub Stars", f"{analytics.get('github_stars', 0):,}")
    dev_table.add_row("GitHub Forks", f"{analytics.get('github_forks', 0):,}")
    dev_table.add_row("Recent Commits (30d)", f"{analytics.get('github_commits_30d', 0):,}")
    dev_table.add_row("Activity Score", f"{analytics.get('dev_activity_score', 0):.1f}/100")

    # Display token info and tables
    console.print(f"\n[bold]Token: [magenta]{token_name} ({token_symbol})[/magenta][/bold]")
    console.print(f"Mint: {token_mint}")
    console.print(f"Last Updated: {analytics.get('last_updated', 'Unknown')}\n")

    # Display tables in a 2x2 grid
    console.print(Panel(market_table, title="[bold green]MARKET DATA[/bold green]", border_style="green", box=ROUNDED))
    console.print(Panel(holder_table, title="[bold blue]HOLDER METRICS[/bold blue]", border_style="blue", box=ROUNDED))
    console.print(Panel(sentiment_table, title="[bold magenta]SOCIAL SENTIMENT[/bold magenta]", border_style="magenta", box=ROUNDED))
    console.print(Panel(dev_table, title="[bold yellow]DEVELOPER ACTIVITY[/bold yellow]", border_style="yellow", box=ROUNDED))


def withdraw():
    """Withdraw SOL or tokens to an external wallet."""
    if not wallet_connected:
        console.print("[bold red]No wallet connected. Please connect a wallet first.[/bold red]")
        input("Press Enter to continue...")
        return True

    console.print(Panel(
        "[bold white]Withdraw SOL or Tokens to External Wallet[/bold white]",
        title="[bold cyan]WITHDRAW[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))

    # Show wallet balance
    sol_balance = solana_client.get_sol_balance(wallet_pubkey)
    console.print(f"[bold white]Current SOL Balance:[/bold white] [green]{sol_balance:.5f} SOL[/green]\n")

    # Get token balances
    token_balances = solana_client.get_token_balances(wallet_pubkey)

    # Show token balances if any
    if token_balances:
        tokens_table = Table(
            show_header=True,
            header_style="bold white",
            box=ROUNDED,
            border_style="cyan",
            padding=(0, 1)
        )
        tokens_table.add_column("#", style="cyan", width=5, justify="center")
        tokens_table.add_column("TOKEN", style="magenta")
        tokens_table.add_column("BALANCE", style="green", justify="right")
        tokens_table.add_column("MINT ADDRESS", style="dim white")

        for i, (token_mint, token_data) in enumerate(token_balances.items(), 1):
            tokens_table.add_row(
                str(i),
                token_data.get("symbol", "Unknown"),
                f"{token_data.get('balance', 0):,.2f}",
                f"{token_mint[:8]}...{token_mint[-8:]}"
            )

        console.print(tokens_table)
    else:
        console.print("[yellow]No token balances found.[/yellow]\n")

    # Ask what to withdraw
    console.print("[bold]What would you like to withdraw?[/bold]")
    console.print("1. SOL")
    console.print("2. Token")
    console.print("Q. Return to main menu")
    console.print("\nEnter your choice: ", end="")

    choice = input().strip().upper()

    if choice == "Q":
        return True

    if choice == "1":
        # Withdraw SOL
        console.print("\n[bold]Withdraw SOL[/bold]")
        console.print(f"Available balance: [green]{sol_balance:.5f} SOL[/green]")
        console.print("Enter amount to withdraw (or 'max' for maximum): ", end="")

        amount_input = input().strip().lower()

        if amount_input == "max":
            # Leave 0.01 SOL for fees
            amount = max(0, sol_balance - 0.01)
        else:
            try:
                amount = float(amount_input)
            except ValueError:
                console.print("[bold red]Invalid amount. Please enter a number.[/bold red]")
                input("Press Enter to continue...")
                return True

        if amount <= 0:
            console.print("[bold red]Amount must be greater than 0.[/bold red]")
            input("Press Enter to continue...")
            return True

        if amount > sol_balance:
            console.print("[bold red]Amount exceeds available balance.[/bold red]")
            input("Press Enter to continue...")
            return True

        # Get destination address
        console.print("Enter destination wallet address: ", end="")
        destination = input().strip()

        # Confirm withdrawal
        console.print(f"\n[bold yellow]Confirm withdrawal of {amount:.5f} SOL to {destination}[/bold yellow]")
        console.print("Type 'confirm' to proceed: ", end="")

        confirmation = input().strip().lower()

        if confirmation != "confirm":
            console.print("[yellow]Withdrawal cancelled.[/yellow]")
            input("Press Enter to continue...")
            return True

        # Execute withdrawal
        console.print("\n[bold]Processing withdrawal...[/bold]")
        result = withdraw_manager.withdraw_sol(amount, destination)

        if result["success"]:
            console.print(f"[bold green]Successfully withdrew {amount:.5f} SOL to {destination}[/bold green]")
            console.print(f"Transaction signature: {result['signature']}")
        else:
            console.print(f"[bold red]Withdrawal failed: {result['error']}[/bold red]")

        input("Press Enter to continue...")
        return True

    elif choice == "2":
        # Withdraw token
        if not token_balances:
            console.print("[bold red]No tokens available to withdraw.[/bold red]")
            input("Press Enter to continue...")
            return True

        console.print("\n[bold]Withdraw Token[/bold]")
        console.print("Enter token number from the list above: ", end="")

        try:
            token_index = int(input().strip()) - 1
            if token_index < 0 or token_index >= len(token_balances):
                raise ValueError("Invalid token index")
        except ValueError:
            console.print("[bold red]Invalid token selection.[/bold red]")
            input("Press Enter to continue...")
            return True

        # Get selected token
        token_mint = list(token_balances.keys())[token_index]
        token_data = token_balances[token_mint]
        token_symbol = token_data.get("symbol", "Unknown")
        token_balance = token_data.get("balance", 0)

        console.print(f"\nSelected token: [magenta]{token_symbol}[/magenta]")
        console.print(f"Available balance: [green]{token_balance:,.2f} {token_symbol}[/green]")
        console.print("Enter amount to withdraw (or 'max' for maximum): ", end="")

        amount_input = input().strip().lower()

        if amount_input == "max":
            amount = token_balance
        else:
            try:
                amount = float(amount_input)
            except ValueError:
                console.print("[bold red]Invalid amount. Please enter a number.[/bold red]")
                input("Press Enter to continue...")
                return True

        if amount <= 0:
            console.print("[bold red]Amount must be greater than 0.[/bold red]")
            input("Press Enter to continue...")
            return True

        if amount > token_balance:
            console.print("[bold red]Amount exceeds available balance.[/bold red]")
            input("Press Enter to continue...")
            return True

        # Get destination address
        console.print("Enter destination wallet address: ", end="")
        destination = input().strip()

        # Confirm withdrawal
        console.print(f"\n[bold yellow]Confirm withdrawal of {amount:,.2f} {token_symbol} to {destination}[/bold yellow]")
        console.print("Type 'confirm' to proceed: ", end="")

        confirmation = input().strip().lower()

        if confirmation != "confirm":
            console.print("[yellow]Withdrawal cancelled.[/yellow]")
            input("Press Enter to continue...")
            return True

        # Execute withdrawal
        console.print("\n[bold]Processing withdrawal...[/bold]")
        result = withdraw_manager.withdraw_token(token_mint, amount, destination)

        if result["success"]:
            console.print(f"[bold green]Successfully withdrew {amount:,.2f} {token_symbol} to {destination}[/bold green]")
            console.print(f"Transaction signature: {result['signature']}")
        else:
            console.print(f"[bold red]Withdrawal failed: {result['error']}[/bold red]")

        input("Press Enter to continue...")
        return True

    else:
        console.print("[bold red]Invalid choice.[/bold red]")
        input("Press Enter to continue...")
        return True


def limit_orders():
    """Manage limit orders for buying and selling tokens."""
    if not wallet_connected:
        console.print("[bold red]No wallet connected. Please connect a wallet first.[/bold red]")
        input("Press Enter to continue...")
        return True

    console.print(Panel(
        "[bold white]Limit Orders Management[/bold white]",
        title="[bold cyan]LIMIT ORDERS[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))

    while True:
        # Get active orders
        active_orders = limit_order_manager.get_orders(status="active")

        # Show active orders
        console.print("[bold white]Active Limit Orders:[/bold white]")

        if active_orders:
            orders_table = Table(
                show_header=True,
                header_style="bold white",
                box=ROUNDED,
                border_style="cyan",
                padding=(0, 1)
            )
            orders_table.add_column("#", style="cyan", width=5, justify="center")
            orders_table.add_column("TOKEN", style="magenta")
            orders_table.add_column("TYPE", style="yellow")
            orders_table.add_column("TARGET PRICE", style="green", justify="right")
            orders_table.add_column("AMOUNT", style="cyan", justify="right")
            orders_table.add_column("CREATED", style="dim white")

            for i, (order_id, order) in enumerate(active_orders.items(), 1):
                order_type = order["type"]
                amount_str = f"{order['amount']:.3f} SOL" if order_type == "buy" else f"{order['amount']}%"
                created_at = datetime.fromisoformat(order["created_at"]).strftime("%Y-%m-%d %H:%M")

                orders_table.add_row(
                    str(i),
                    order["token_symbol"],
                    order_type.upper(),
                    f"{order['target_price']:.8f}",
                    amount_str,
                    created_at
                )

            console.print(orders_table)
        else:
            console.print("[yellow]No active limit orders.[/yellow]\n")

        # Show menu
        console.print("\n[bold]Limit Orders Menu:[/bold]")
        console.print("1. Create Buy Limit Order")
        console.print("2. Create Sell Limit Order")
        console.print("3. Cancel Limit Order")
        console.print("4. View Order History")
        console.print("5. Clear Inactive Orders")
        console.print("Q. Return to Main Menu")
        console.print("\nEnter your choice: ", end="")

        choice = input().strip().upper()

        if choice == "Q":
            break

        if choice == "1":
            # Create buy limit order
            create_buy_limit_order()
        elif choice == "2":
            # Create sell limit order
            create_sell_limit_order()
        elif choice == "3":
            # Cancel limit order
            cancel_limit_order(active_orders)
        elif choice == "4":
            # View order history
            view_order_history()
        elif choice == "5":
            # Clear inactive orders
            clear_inactive_orders()
        else:
            console.print("[bold red]Invalid choice.[/bold red]")
            input("Press Enter to continue...")

    return True


def create_buy_limit_order():
    """Create a buy limit order."""
    console.print("\n[bold]Create Buy Limit Order[/bold]")

    # Get SOL balance
    sol_balance = solana_client.get_sol_balance(wallet_pubkey)
    console.print(f"Available SOL: [green]{sol_balance:.5f} SOL[/green]")

    # Get token mint
    console.print("Enter token mint address: ", end="")
    token_mint = input().strip()

    # Validate token mint
    try:
        # Get token info
        token_info = jupiter_api.get_token_info(token_mint)
        if not token_info:
            console.print("[bold red]Invalid token mint address or token not found.[/bold red]")
            input("Press Enter to continue...")
            return

        token_symbol = token_info.get("symbol", token_mint[:8])
        console.print(f"Token: [magenta]{token_symbol}[/magenta]")

        # Get current price
        current_price = jupiter_api.get_token_price(token_mint)
        if current_price is None:
            console.print("[bold red]Could not get current price for this token.[/bold red]")
            input("Press Enter to continue...")
            return

        console.print(f"Current price: [green]{current_price:.8f} SOL[/green]")

        # Get target price
        console.print("Enter target price (in SOL): ", end="")
        target_price_str = input().strip()

        try:
            target_price = float(target_price_str)
            if target_price <= 0:
                raise ValueError("Price must be greater than 0")
        except ValueError:
            console.print("[bold red]Invalid price. Please enter a valid number.[/bold red]")
            input("Press Enter to continue...")
            return

        # Get amount in SOL
        console.print("Enter amount to spend (in SOL): ", end="")
        amount_str = input().strip()

        try:
            amount = float(amount_str)
            if amount <= 0:
                raise ValueError("Amount must be greater than 0")
            if amount > sol_balance:
                raise ValueError("Amount exceeds available balance")
        except ValueError as e:
            console.print(f"[bold red]{str(e)}[/bold red]")
            input("Press Enter to continue...")
            return

        # Get expiry (optional)
        console.print("Enter expiry in hours (leave blank for no expiry): ", end="")
        expiry_str = input().strip()

        expiry = None
        if expiry_str:
            try:
                expiry_hours = float(expiry_str)
                if expiry_hours <= 0:
                    raise ValueError("Expiry must be greater than 0")
                expiry = datetime.now() + timedelta(hours=expiry_hours)
            except ValueError:
                console.print("[bold red]Invalid expiry. Please enter a valid number.[/bold red]")
                input("Press Enter to continue...")
                return

        # Confirm order
        console.print(f"\n[bold yellow]Confirm buy limit order:[/bold yellow]")
        console.print(f"Token: [magenta]{token_symbol}[/magenta]")
        console.print(f"Current price: [green]{current_price:.8f} SOL[/green]")
        console.print(f"Target price: [green]{target_price:.8f} SOL[/green]")
        console.print(f"Amount: [cyan]{amount:.5f} SOL[/cyan]")
        if expiry:
            console.print(f"Expiry: [dim]{expiry.strftime('%Y-%m-%d %H:%M')}[/dim]")
        else:
            console.print("Expiry: [dim]None[/dim]")

        console.print("\nType 'confirm' to create this order: ", end="")
        confirmation = input().strip().lower()

        if confirmation != "confirm":
            console.print("[yellow]Order creation cancelled.[/yellow]")
            input("Press Enter to continue...")
            return

        # Create order
        try:
            order = limit_order_manager.create_limit_order(
                token_mint=token_mint,
                token_symbol=token_symbol,
                order_type="buy",
                target_price=target_price,
                amount=amount,
                expiry=expiry
            )

            console.print(f"[bold green]Buy limit order created successfully![/bold green]")
            console.print(f"Order ID: {order['id']}")

            # Ensure monitoring is running
            if not limit_order_manager.monitoring_thread or not limit_order_manager.monitoring_thread.is_alive():
                limit_order_manager.start_monitoring_thread()
        except Exception as e:
            console.print(f"[bold red]Error creating order: {str(e)}[/bold red]")

        input("Press Enter to continue...")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        input("Press Enter to continue...")


def create_sell_limit_order():
    """Create a sell limit order."""
    console.print("\n[bold]Create Sell Limit Order[/bold]")

    # Get token balances
    token_balances = solana_client.get_token_balances(wallet_pubkey)

    if not token_balances:
        console.print("[bold red]No tokens available to sell.[/bold red]")
        input("Press Enter to continue...")
        return

    # Show token balances
    tokens_table = Table(
        show_header=True,
        header_style="bold white",
        box=ROUNDED,
        border_style="cyan",
        padding=(0, 1)
    )
    tokens_table.add_column("#", style="cyan", width=5, justify="center")
    tokens_table.add_column("TOKEN", style="magenta")
    tokens_table.add_column("BALANCE", style="green", justify="right")
    tokens_table.add_column("MINT ADDRESS", style="dim white")

    for i, (token_mint, token_data) in enumerate(token_balances.items(), 1):
        tokens_table.add_row(
            str(i),
            token_data.get("symbol", "Unknown"),
            f"{token_data.get('balance', 0):,.2f}",
            f"{token_mint[:8]}...{token_mint[-8:]}"
        )

    console.print(tokens_table)

    # Select token
    console.print("\nEnter token number from the list above: ", end="")

    try:
        token_index = int(input().strip()) - 1
        if token_index < 0 or token_index >= len(token_balances):
            raise ValueError("Invalid token index")
    except ValueError:
        console.print("[bold red]Invalid token selection.[/bold red]")
        input("Press Enter to continue...")
        return

    # Get selected token
    token_mint = list(token_balances.keys())[token_index]
    token_data = token_balances[token_mint]
    token_symbol = token_data.get("symbol", "Unknown")
    token_balance = token_data.get("balance", 0)

    console.print(f"\nSelected token: [magenta]{token_symbol}[/magenta]")
    console.print(f"Available balance: [green]{token_balance:,.2f} {token_symbol}[/green]")

    # Get current price
    current_price = jupiter_api.get_token_price(token_mint)
    if current_price is None:
        console.print("[bold red]Could not get current price for this token.[/bold red]")
        input("Press Enter to continue...")
        return

    console.print(f"Current price: [green]{current_price:.8f} SOL[/green]")

    # Get target price
    console.print("Enter target price (in SOL): ", end="")
    target_price_str = input().strip()

    try:
        target_price = float(target_price_str)
        if target_price <= 0:
            raise ValueError("Price must be greater than 0")
    except ValueError:
        console.print("[bold red]Invalid price. Please enter a valid number.[/bold red]")
        input("Press Enter to continue...")
        return

    # Get percentage to sell
    console.print("Enter percentage to sell (1-100): ", end="")
    percentage_str = input().strip()

    try:
        percentage = float(percentage_str)
        if percentage <= 0 or percentage > 100:
            raise ValueError("Percentage must be between 1 and 100")
    except ValueError as e:
        console.print(f"[bold red]{str(e)}[/bold red]")
        input("Press Enter to continue...")
        return

    # Get expiry (optional)
    console.print("Enter expiry in hours (leave blank for no expiry): ", end="")
    expiry_str = input().strip()

    expiry = None
    if expiry_str:
        try:
            expiry_hours = float(expiry_str)
            if expiry_hours <= 0:
                raise ValueError("Expiry must be greater than 0")
            expiry = datetime.now() + timedelta(hours=expiry_hours)
        except ValueError:
            console.print("[bold red]Invalid expiry. Please enter a valid number.[/bold red]")
            input("Press Enter to continue...")
            return

    # Confirm order
    console.print(f"\n[bold yellow]Confirm sell limit order:[/bold yellow]")
    console.print(f"Token: [magenta]{token_symbol}[/magenta]")
    console.print(f"Current price: [green]{current_price:.8f} SOL[/green]")
    console.print(f"Target price: [green]{target_price:.8f} SOL[/green]")
    console.print(f"Percentage to sell: [cyan]{percentage:.1f}%[/cyan]")
    if expiry:
        console.print(f"Expiry: [dim]{expiry.strftime('%Y-%m-%d %H:%M')}[/dim]")
    else:
        console.print("Expiry: [dim]None[/dim]")

    console.print("\nType 'confirm' to create this order: ", end="")
    confirmation = input().strip().lower()

    if confirmation != "confirm":
        console.print("[yellow]Order creation cancelled.[/yellow]")
        input("Press Enter to continue...")
        return

    # Create order
    try:
        order = limit_order_manager.create_limit_order(
            token_mint=token_mint,
            token_symbol=token_symbol,
            order_type="sell",
            target_price=target_price,
            amount=percentage,
            expiry=expiry
        )

        console.print(f"[bold green]Sell limit order created successfully![/bold green]")
        console.print(f"Order ID: {order['id']}")

        # Ensure monitoring is running
        if not limit_order_manager.monitoring_thread or not limit_order_manager.monitoring_thread.is_alive():
            limit_order_manager.start_monitoring_thread()
    except Exception as e:
        console.print(f"[bold red]Error creating order: {str(e)}[/bold red]")

    input("Press Enter to continue...")


def cancel_limit_order(active_orders):
    """Cancel a limit order."""
    if not active_orders:
        console.print("[bold yellow]No active orders to cancel.[/bold yellow]")
        input("Press Enter to continue...")
        return

    console.print("\n[bold]Cancel Limit Order[/bold]")
    console.print("Enter order number to cancel (from the list above): ", end="")

    try:
        order_index = int(input().strip()) - 1
        if order_index < 0 or order_index >= len(active_orders):
            raise ValueError("Invalid order index")
    except ValueError:
        console.print("[bold red]Invalid order selection.[/bold red]")
        input("Press Enter to continue...")
        return

    # Get selected order
    order_id = list(active_orders.keys())[order_index]
    order = active_orders[order_id]

    # Confirm cancellation
    console.print(f"\n[bold yellow]Confirm cancellation of order:[/bold yellow]")
    console.print(f"Token: [magenta]{order['token_symbol']}[/magenta]")
    console.print(f"Type: [yellow]{order['type'].upper()}[/yellow]")
    console.print(f"Target price: [green]{order['target_price']:.8f} SOL[/green]")

    console.print("\nType 'confirm' to cancel this order: ", end="")
    confirmation = input().strip().lower()

    if confirmation != "confirm":
        console.print("[yellow]Cancellation aborted.[/yellow]")
        input("Press Enter to continue...")
        return

    # Cancel order
    result = limit_order_manager.cancel_order(order_id)

    if result:
        console.print(f"[bold green]Order cancelled successfully![/bold green]")
    else:
        console.print(f"[bold red]Failed to cancel order.[/bold red]")

    input("Press Enter to continue...")


def view_order_history():
    """View order history."""
    console.print("\n[bold]Order History[/bold]")

    # Get all orders
    all_orders = limit_order_manager.get_orders()

    if not all_orders:
        console.print("[bold yellow]No order history found.[/bold yellow]")
        input("Press Enter to continue...")
        return

    # Create table
    history_table = Table(
        show_header=True,
        header_style="bold white",
        box=ROUNDED,
        border_style="cyan",
        padding=(0, 1)
    )
    history_table.add_column("TOKEN", style="magenta")
    history_table.add_column("TYPE", style="yellow")
    history_table.add_column("TARGET PRICE", style="green", justify="right")
    history_table.add_column("STATUS", style="cyan")
    history_table.add_column("CREATED", style="dim white")

    # Add orders to table
    for order_id, order in all_orders.items():
        status = order["status"]
        status_style = {
            "active": "green",
            "executed": "blue",
            "cancelled": "yellow",
            "expired": "dim",
            "failed": "red"
        }.get(status, "white")

        created_at = datetime.fromisoformat(order["created_at"]).strftime("%Y-%m-%d %H:%M")

        history_table.add_row(
            order["token_symbol"],
            order["type"].upper(),
            f"{order['target_price']:.8f}",
            f"[{status_style}]{status.upper()}[/{status_style}]",
            created_at
        )

    console.print(history_table)
    input("Press Enter to continue...")


def clear_inactive_orders():
    """Clear inactive orders."""
    console.print("\n[bold]Clear Inactive Orders[/bold]")
    console.print("This will remove all executed, cancelled, expired, and failed orders.")
    console.print("Type 'confirm' to proceed: ", end="")

    confirmation = input().strip().lower()

    if confirmation != "confirm":
        console.print("[yellow]Operation cancelled.[/yellow]")
        input("Press Enter to continue...")
        return

    # Clear inactive orders
    count = limit_order_manager.clear_inactive_orders()

    console.print(f"[bold green]Cleared {count} inactive orders.[/bold green]")
    input("Press Enter to continue...")


def rapid_trading_menu():
    """Display rapid trading menu and handle rapid trading operations."""
    from src.trading.rapid_executor import rapid_executor
    from src.trading.realtime_feed import realtime_feed

    while True:
        clear_screen()
        console.print(Panel(
            "[bold cyan]RAPID TRADING SYSTEM[/bold cyan]\n\n"
            "[white]High-frequency trading with minimal latency[/white]",
            title="Rapid Trading",
            border_style="cyan"
        ))

        # Show system status
        executor_stats = rapid_executor.get_performance_stats()
        feed_stats = realtime_feed.get_performance_stats()

        status_table = Table(title="System Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Metrics", style="yellow")

        status_table.add_row(
            "Rapid Executor",
            "Running" if rapid_executor.running else "Stopped",
            f"Success Rate: {executor_stats['success_rate']:.1%}, "
            f"Avg Time: {executor_stats['average_execution_time']:.3f}s"
        )

        status_table.add_row(
            "Real-time Feed",
            "Connected" if feed_stats['connected'] else "Disconnected",
            f"Latency: {feed_stats['average_latency_ms']:.1f}ms, "
            f"Messages: {feed_stats['message_count']}"
        )

        console.print(status_table)

        # Menu options
        console.print("\n[bold cyan]Options:[/bold cyan]")
        console.print("1. Start/Stop Rapid Executor")
        console.print("2. Start/Stop Real-time Feed")
        console.print("3. Rapid Buy (rbuy)")
        console.print("4. Rapid Sell (rsell)")
        console.print("5. View Active Orders")
        console.print("6. Performance Statistics")
        console.print("7. Back to Main Menu")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            if rapid_executor.running:
                rapid_executor.stop()
                console.print("[green]Rapid executor stopped[/green]")
            else:
                rapid_executor.start()
                console.print("[green]Rapid executor started[/green]")
            input("Press Enter to continue...")

        elif choice == "2":
            if realtime_feed.connected:
                realtime_feed.stop()
                console.print("[green]Real-time feed stopped[/green]")
            else:
                realtime_feed.start()
                console.print("[green]Real-time feed started[/green]")
            input("Press Enter to continue...")

        elif choice == "3":
            rapid_buy()

        elif choice == "4":
            rapid_sell()

        elif choice == "5":
            show_active_rapid_orders()

        elif choice == "6":
            show_rapid_performance_stats()

        elif choice == "7":
            break

        else:
            console.print("[red]Invalid choice[/red]")
            input("Press Enter to continue...")

    return True


def rapid_buy():
    """Execute a rapid buy order."""
    from src.trading.rapid_executor import rapid_executor

    console.print(Panel(
        "[bold cyan]RAPID BUY ORDER[/bold cyan]",
        border_style="cyan"
    ))

    try:
        # Get token mint
        token_mint = input("Enter token mint address: ").strip()
        if not token_mint:
            console.print("[red]Token mint address is required[/red]")
            input("Press Enter to continue...")
            return True

        # Get amount
        amount_str = input("Enter SOL amount to spend: ").strip()
        try:
            amount = float(amount_str)
            if amount <= 0:
                raise ValueError("Amount must be positive")
        except ValueError:
            console.print("[red]Invalid amount[/red]")
            input("Press Enter to continue...")
            return True

        # Get priority (optional)
        priority_str = input("Enter priority (1-5, default 1): ").strip()
        priority = 1
        if priority_str:
            try:
                priority = int(priority_str)
                if priority < 1 or priority > 5:
                    priority = 1
            except ValueError:
                priority = 1

        # Get price limit (optional)
        price_limit = None
        price_limit_str = input("Enter max price limit (optional): ").strip()
        if price_limit_str:
            try:
                price_limit = float(price_limit_str)
            except ValueError:
                console.print("[yellow]Invalid price limit, proceeding without limit[/yellow]")

        # Submit order
        order_id = rapid_executor.submit_rapid_buy(
            token_mint=token_mint,
            amount_sol=amount,
            priority=priority,
            price_limit=price_limit
        )

        console.print(f"[green]Rapid buy order submitted: {order_id}[/green]")
        console.print(f"Token: {token_mint}")
        console.print(f"Amount: {amount} SOL")
        console.print(f"Priority: {priority}")
        if price_limit:
            console.print(f"Price Limit: {price_limit}")

    except Exception as e:
        console.print(f"[red]Error submitting rapid buy order: {e}[/red]")

    input("Press Enter to continue...")
    return True


def rapid_sell():
    """Execute a rapid sell order."""
    from src.trading.rapid_executor import rapid_executor
    from src.trading.position_manager import position_manager

    console.print(Panel(
        "[bold cyan]RAPID SELL ORDER[/bold cyan]",
        border_style="cyan"
    ))

    try:
        # Show current positions
        positions = position_manager.get_all_positions()
        if not positions:
            console.print("[yellow]No positions available to sell[/yellow]")
            input("Press Enter to continue...")
            return True

        console.print("\n[bold cyan]Current Positions:[/bold cyan]")
        for i, (token_mint, position) in enumerate(positions.items(), 1):
            console.print(f"{i}. {position.token_name} ({token_mint[:8]}...)")
            console.print(f"   Amount: {position.amount:.6f}")
            console.print(f"   Current Price: {position.current_price:.8f} SOL")

        # Get position selection
        choice = input("\nSelect position number or enter token mint: ").strip()

        token_mint = None
        position = None

        try:
            # Try as position number
            pos_num = int(choice)
            if 1 <= pos_num <= len(positions):
                token_mint = list(positions.keys())[pos_num - 1]
                position = positions[token_mint]
        except ValueError:
            # Try as token mint
            if choice in positions:
                token_mint = choice
                position = positions[token_mint]

        if not token_mint or not position:
            console.print("[red]Invalid selection[/red]")
            input("Press Enter to continue...")
            return True

        # Get amount to sell
        amount_str = input(f"Enter amount to sell (max: {position.amount:.6f}): ").strip()
        try:
            amount = float(amount_str)
            if amount <= 0 or amount > position.amount:
                raise ValueError("Invalid amount")
        except ValueError:
            console.print("[red]Invalid amount[/red]")
            input("Press Enter to continue...")
            return True

        # Get priority
        priority_str = input("Enter priority (1-5, default 1): ").strip()
        priority = 1
        if priority_str:
            try:
                priority = int(priority_str)
                if priority < 1 or priority > 5:
                    priority = 1
            except ValueError:
                priority = 1

        # Get price limit (optional)
        price_limit = None
        price_limit_str = input("Enter min price limit (optional): ").strip()
        if price_limit_str:
            try:
                price_limit = float(price_limit_str)
            except ValueError:
                console.print("[yellow]Invalid price limit, proceeding without limit[/yellow]")

        # Submit order
        order_id = rapid_executor.submit_rapid_sell(
            token_mint=token_mint,
            amount_tokens=amount,
            priority=priority,
            price_limit=price_limit
        )

        console.print(f"[green]Rapid sell order submitted: {order_id}[/green]")
        console.print(f"Token: {position.token_name}")
        console.print(f"Amount: {amount}")
        console.print(f"Priority: {priority}")
        if price_limit:
            console.print(f"Price Limit: {price_limit}")

    except Exception as e:
        console.print(f"[red]Error submitting rapid sell order: {e}[/red]")

    input("Press Enter to continue...")
    return True


def rapid_cancel():
    """Cancel a rapid order."""
    from src.trading.rapid_executor import rapid_executor

    console.print(Panel(
        "[bold cyan]CANCEL RAPID ORDER[/bold cyan]",
        border_style="cyan"
    ))

    try:
        order_id = input("Enter order ID to cancel: ").strip()
        if not order_id:
            console.print("[red]Order ID is required[/red]")
            input("Press Enter to continue...")
            return True

        success = rapid_executor.cancel_order(order_id)
        if success:
            console.print(f"[green]Order {order_id} cancelled successfully[/green]")
        else:
            console.print(f"[red]Failed to cancel order {order_id} (not found or already executing)[/red]")

    except Exception as e:
        console.print(f"[red]Error cancelling order: {e}[/red]")

    input("Press Enter to continue...")
    return True


def rapid_status():
    """Show status of rapid orders."""
    from src.trading.rapid_executor import rapid_executor

    console.print(Panel(
        "[bold cyan]RAPID ORDER STATUS[/bold cyan]",
        border_style="cyan"
    ))

    try:
        order_id = input("Enter order ID (or press Enter for all active orders): ").strip()

        if order_id:
            # Show specific order
            status = rapid_executor.get_order_status(order_id)
            console.print(f"\nOrder ID: {order_id}")
            console.print(f"Status: {status['status']}")

            if status['status'] == 'pending':
                order = status['order']
                console.print(f"Action: {order.action}")
                console.print(f"Token: {order.token_mint}")
                console.print(f"Amount: {order.amount}")
                console.print(f"Priority: {order.priority}")
                console.print(f"Submitted: {time.strftime('%H:%M:%S', time.localtime(order.timestamp))}")
            elif status['status'] == 'completed':
                result = status['result']
                console.print(f"Success: {result.get('success', False)}")
                if result.get('success'):
                    console.print(f"Transaction: {result.get('tx_signature', 'N/A')}")
                    console.print(f"Execution Time: {result.get('execution_time', 0):.3f}s")
                else:
                    console.print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            # Show all active orders
            stats = rapid_executor.get_performance_stats()
            console.print(f"\nActive Orders: {stats['pending_orders']}")
            console.print(f"Completed Orders: {stats['completed_orders']}")
            console.print(f"Success Rate: {stats['success_rate']:.1%}")
            console.print(f"Average Execution Time: {stats['average_execution_time']:.3f}s")

    except Exception as e:
        console.print(f"[red]Error getting order status: {e}[/red]")

    input("Press Enter to continue...")
    return True


def show_active_rapid_orders():
    """Show all active rapid orders."""
    from src.trading.rapid_executor import rapid_executor

    console.print(Panel(
        "[bold cyan]ACTIVE RAPID ORDERS[/bold cyan]",
        border_style="cyan"
    ))

    try:
        # This would need to be implemented in rapid_executor
        # For now, show basic stats
        stats = rapid_executor.get_performance_stats()

        table = Table(title="Order Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Pending Orders", str(stats['pending_orders']))
        table.add_row("Completed Orders", str(stats['completed_orders']))
        table.add_row("Success Rate", f"{stats['success_rate']:.1%}")
        table.add_row("Avg Execution Time", f"{stats['average_execution_time']:.3f}s")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error showing active orders: {e}[/red]")

    input("Press Enter to continue...")


def show_rapid_performance_stats():
    """Show detailed performance statistics for rapid trading."""
    from src.trading.rapid_executor import rapid_executor
    from src.trading.realtime_feed import realtime_feed

    console.print(Panel(
        "[bold cyan]RAPID TRADING PERFORMANCE[/bold cyan]",
        border_style="cyan"
    ))

    try:
        executor_stats = rapid_executor.get_performance_stats()
        feed_stats = realtime_feed.get_performance_stats()

        # Executor statistics
        exec_table = Table(title="Execution Engine")
        exec_table.add_column("Metric", style="cyan")
        exec_table.add_column("Value", style="green")

        exec_table.add_row("Total Orders", str(executor_stats['total_orders']))
        exec_table.add_row("Successful Orders", str(executor_stats['successful_orders']))
        exec_table.add_row("Success Rate", f"{executor_stats['success_rate']:.1%}")
        exec_table.add_row("Average Execution Time", f"{executor_stats['average_execution_time']:.3f}s")
        exec_table.add_row("Pending Orders", str(executor_stats['pending_orders']))

        console.print(exec_table)

        # Feed statistics
        feed_table = Table(title="Real-time Data Feed")
        feed_table.add_column("Metric", style="cyan")
        feed_table.add_column("Value", style="green")

        feed_table.add_row("Connection Status", "Connected" if feed_stats['connected'] else "Disconnected")
        feed_table.add_row("Messages Received", str(feed_stats['message_count']))
        feed_table.add_row("Average Latency", f"{feed_stats['average_latency_ms']:.1f}ms")
        feed_table.add_row("Subscribed Tokens", str(feed_stats['subscribed_tokens']))
        feed_table.add_row("Cached Prices", str(feed_stats['cached_prices']))

        console.print(feed_table)

    except Exception as e:
        console.print(f"[red]Error showing performance stats: {e}[/red]")

    input("Press Enter to continue...")


def not_implemented(feature_name):
    """Placeholder for not implemented features."""
    console.print(f"[bold yellow]The {feature_name} feature is not implemented yet.[/bold yellow]")
    input("Press Enter to continue...")
    return True


def register_all_functions():
    """Register all CLI functions."""
    # Register core functions
    register_function("start", start_bot)
    register_function("stop", stop_bot)
    register_function("dashboard", show_dashboard)
    register_function("settings", show_settings)
    register_function("connect", connect_wallet)
    register_function("disconnect", disconnect_wallet)
    register_function("exit", exit_bot)
    register_function("help", show_help)
    register_function("logs", show_logs)

    # Register trading functions
    register_function("buy", buy_token)
    register_function("sell", sell_token)
    register_function("positions", show_positions)
    register_function("snipe", snipe_token)
    register_function("copy", copy_trading_function)
    register_function("sentiment", sentiment_analysis_function)
    register_function("strategy", strategy_generator_function)
    register_function("technical", technical_analysis_function)
    register_function("alerts", price_alerts_function)
    register_function("monitor", wallet_monitor_function)
    register_function("limit", limit_orders_function)
    register_function("dca", dca_orders_function)
    register_function("auto", auto_buy_function)
    register_function("withdraw", withdraw_function)
    register_function("token_analytics", token_analytics_function)

    # Register enhanced features
    register_function("enhanced_copy_trading", enhanced_copy_trading_menu)
    register_function("enhanced_portfolio", enhanced_portfolio_menu)
    register_function("advanced_alerts", advanced_alerts_menu)
    register_function("smart_copy_trading", smart_copy_trading_menu)
    register_function("multi_dex_hunting", multi_dex_hunting_menu)

    # Register Phase 2 features
    from src.cli.enhanced_features_cli import (
        advanced_risk_analytics, smart_order_management,
        performance_attribution, ai_pool_analysis
    )
    register_function("advanced_risk_analytics", advanced_risk_analytics)
    register_function("smart_order_management", smart_order_management)
    register_function("performance_attribution", performance_attribution)
    register_function("ai_pool_analysis", ai_pool_analysis)

    # Import and register risk management functions
    from src.cli.risk_functions import manage_risk, check_portfolio
    register_function("risk", manage_risk)
    register_function("portfolio", check_portfolio)

    # Import and register gas optimization functions
    from src.cli.gas_functions import manage_gas, check_fees
    register_function("gas", manage_gas)
    register_function("fees", check_fees)

    # Import and register refinement functions
    from src.cli.refinement_functions import manage_refinement, run_risk_refinement, run_gas_refinement, check_auto_refinement
    register_function("refinement", manage_refinement)
    register_function("refine_risk", run_risk_refinement)
    register_function("refine_gas", run_gas_refinement)

    # Register Phase 3 features
    from src.cli.enhanced_features_cli import (
        dynamic_portfolio_optimization_menu, enhanced_ai_pool_analysis_menu,
        advanced_benchmarking_menu
    )
    register_function("dynamic_portfolio_optimization", dynamic_portfolio_optimization_menu)
    register_function("enhanced_ai_pool_analysis", enhanced_ai_pool_analysis_menu)
    register_function("advanced_benchmarking", advanced_benchmarking_menu)

    # Register Phase 4 features
    from src.cli.phase4_functions import (
        live_trading_engine, advanced_ai_predictions, cross_chain_management,
        enterprise_api, production_monitoring
    )
    register_function("live_trading_engine", live_trading_engine)
    register_function("advanced_ai_predictions", advanced_ai_predictions)
    register_function("cross_chain_management", cross_chain_management)
    register_function("enterprise_api", enterprise_api)
    register_function("production_monitoring", production_monitoring)
