"""
Command-line interface for the Solana Memecoin Trading Bot.
Provides a CLI (Command Line Interface) for interacting with the bot.
"""

import os
import sys
import time
import io
from typing import Dict, List, Callable, Any, Optional
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
from src.charts.chart_generator import chart_generator
from src.mobile.mobile_app import mobile_app_manager
from src.ml.token_evaluator import token_evaluator
from src.community.strategy_sharing import strategy_sharing
from src.wallet.withdraw import withdraw_manager

# Get logger for this module
logger = get_logger(__name__)

# Create console for rich output
console = Console()

# Bot state
bot_running = False
wallet_connected = False
wallet_pubkey = ""
current_keypair = None

# Menu options dictionary
# Will be updated to use letters after numbers 1-9 are exhausted
menu_options = {
    "1": {"label": "Start Bot", "function": "start_bot"},
    "2": {"label": "Dashboard", "function": "show_dashboard"},
    "3": {"label": "Settings", "function": "show_settings"},
    "4": {"label": "Logs", "function": "show_logs"},
    "5": {"label": "Stop Bot", "function": "stop_bot"},
    "6": {"label": "Connect Wallet", "function": "connect_wallet"},
    "7": {"label": "Buy Token", "function": "buy_token"},
    "8": {"label": "Sell Token", "function": "sell_token"},
    "9": {"label": "Check Balance", "function": "check_balance"},
    "A": {"label": "Withdraw", "function": "withdraw"},
    "B": {"label": "Limit Orders", "function": "limit_orders"},
    "C": {"label": "DCA Orders", "function": "dca_orders"},
    "D": {"label": "Price Alerts", "function": "price_alerts"},
    "E": {"label": "Wallet Monitor", "function": "wallet_monitor"},
    "F": {"label": "Auto-Buy", "function": "auto_buy"},
    "G": {"label": "Rapid Trading", "function": "rapid"},
    "H": {"label": "Enhanced Copy Trading", "function": "enhanced_copy_trading"},
    "I": {"label": "Portfolio Management", "function": "enhanced_portfolio"},
    "J": {"label": "Advanced Alerts", "function": "advanced_alerts"},
    "K": {"label": "Smart Copy Trading", "function": "smart_copy_trading"},
    "L": {"label": "Multi-DEX Pool Hunter", "function": "multi_dex_hunting"},
    "M": {"label": "Advanced Risk Analytics", "function": "advanced_risk_analytics"},
    "N": {"label": "Smart Order Management", "function": "smart_order_management"},
    "O": {"label": "Performance Attribution", "function": "performance_attribution"},
    "P": {"label": "AI Pool Analysis", "function": "ai_pool_analysis"},
    "Q": {"label": "Dynamic Portfolio Optimization", "function": "dynamic_portfolio_optimization"},
    "R": {"label": "Enhanced AI Pool Analysis", "function": "enhanced_ai_pool_analysis"},
    "S": {"label": "Advanced Benchmarking", "function": "advanced_benchmarking"},
    "T": {"label": "Live Trading Engine", "function": "live_trading_engine"},
    "U": {"label": "Advanced AI Predictions", "function": "advanced_ai_predictions"},
    "V": {"label": "Cross-Chain Management", "function": "cross_chain_management"},
    "W": {"label": "Enterprise API", "function": "enterprise_api"},
    "X": {"label": "Production Monitoring", "function": "production_monitoring"},
    "Y": {"label": "Exit", "function": "exit_app"}
}

# Function mapping
function_map = {}


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the application header."""
    console.print("\n")
    header = Panel(
        "[bold cyan]SolSniperX[/bold cyan] [white]v1.0[/white]\n[bold white]Solana Memecoin Trading Bot[/bold white]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
        title="[bold cyan]>> SOLSNIPER X <<[/bold cyan]",
        subtitle="[dim]Powered by Solana[/dim]"
    )
    console.print(header)


def print_wallet_status():
    """Print the wallet connection status."""
    global wallet_connected, wallet_pubkey

    if wallet_connected and wallet_pubkey:
        wallet_panel = Panel(
            f"[bold white]{wallet_pubkey[:6]}...{wallet_pubkey[-6:]}[/bold white]",
            title="[bold green]WALLET CONNECTED[/bold green]",
            border_style="green",
            box=ROUNDED,
            padding=(0, 1)
        )
    else:
        wallet_panel = Panel(
            "[bold white]Please connect a wallet to trade[/bold white]",
            title="[bold yellow]WALLET STATUS[/bold yellow]",
            border_style="yellow",
            box=ROUNDED,
            padding=(0, 1)
        )
    console.print(wallet_panel)


def print_menu():
    """Print the main menu options."""
    # Group menu options by category
    categories = {
        "Bot Control": ["1", "5"],  # Start Bot, Stop Bot
        "Information": ["2", "3", "4", "9"],  # Dashboard, Settings, Logs, Check Balance
        "Trading": ["7", "8", "A", "B", "C", "F", "G"],  # Buy, Sell, Withdraw, Limit Orders, DCA Orders, Auto-Buy, Rapid Trading
        "Enhanced Features": ["H", "I", "J", "K", "L", "M", "N", "O", "P"],  # Enhanced Copy Trading, Portfolio Management, Advanced Alerts, Smart Copy Trading, Multi-DEX Pool Hunter, Advanced Risk Analytics, Smart Order Management, Performance Attribution, AI Pool Analysis
        "Phase 3 - Advanced Optimization": ["Q", "R", "S"],  # Dynamic Portfolio Optimization, Enhanced AI Pool Analysis, Advanced Benchmarking
        "Phase 4 - Production Features": ["T", "U", "V", "W", "X"],  # Live Trading Engine, Advanced AI Predictions, Cross-Chain Management, Enterprise API, Production Monitoring
        "Monitoring": ["D", "E"],  # Price Alerts, Wallet Monitor
        "System": ["6", "Y"]  # Connect Wallet, Exit
    }

    # Create a panel for the menu
    menu_panel = Panel(
        "",  # We'll fill this in later
        title="[bold cyan]MENU OPTIONS[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    )

    # Build the menu content as a string
    menu_content = ""

    # Process each category
    for category, keys in categories.items():
        menu_content += f"\n[bold yellow]{category}[/bold yellow]\n"

        # Add options for this category
        for key in keys:
            if key in menu_options:
                option = menu_options[key]

                # Disable wallet-dependent options if wallet is not connected
                disabled = False
                if option["function"] in ["buy_token", "sell_token", "check_balance", "withdraw"] and not wallet_connected:
                    disabled = True

                # Disable Start Bot if already running, and Stop Bot if not running
                if (option["function"] == "start_bot" and bot_running) or (option["function"] == "stop_bot" and not bot_running):
                    disabled = True

                # Add the option with appropriate styling
                if disabled:
                    menu_content += f"  [cyan]{key}[/cyan]  [dim]{option['label']}[/dim]\n"
                else:
                    menu_content += f"  [bold cyan]{key}[/bold cyan]  [white]{option['label']}[/white]\n"

        # Add a blank line between categories
        menu_content += "\n"

    # Update the panel content
    menu_panel.renderable = Text.from_markup(menu_content)

    # Print the menu panel
    console.print(menu_panel)


def get_user_input():
    """Get user input for menu selection."""
    try:
        prompt_panel = Panel(
            "[bold white]Type a command key and press Enter[/bold white]",
            title="[bold cyan]COMMAND[/bold cyan]",
            border_style="cyan",
            box=ROUNDED,
            padding=(0, 1),
            width=40
        )
        console.print(prompt_panel)
        console.print("[bold cyan]>[/bold cyan] ", end="")
        choice = input().strip().upper()
        return choice
    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user[/bold red]")
        return "Q"  # Return exit option on Ctrl+C


def handle_menu_selection(choice):
    """Handle the user's menu selection."""
    global menu_options, function_map

    if choice in menu_options:
        function_name = menu_options[choice]["function"]

        # Check if the function exists in the function map
        if function_name in function_map:
            # Call the function
            return function_map[function_name]()
        else:
            console.print(f"[bold red]Error: Function '{function_name}' not implemented yet.[/bold red]")
            input("Press Enter to continue...")
            return True
    else:
        console.print("[bold red]Invalid option. Please try again.[/bold red]")
        input("Press Enter to continue...")
        return True


def update_menu_options():
    """Update menu options to use letters after numbers 1-9 are exhausted."""
    global menu_options

    # This function is called when the menu is first created
    # The menu_options dictionary is already set up with numbers and letters
    # No need to modify it here
    pass


def register_function(name, func):
    """Register a function in the function map."""
    global function_map
    function_map[name] = func


def run_cli():
    """Run the CLI application."""
    global bot_running, wallet_connected, wallet_pubkey, current_keypair

    # Initialize the CLI
    update_menu_options()

    # Main application loop
    running = True
    while running:
        clear_screen()
        print_header()
        print_wallet_status()
        print_menu()
        choice = get_user_input()
        running = handle_menu_selection(choice)

    # Clean up before exiting
    if bot_running:
        # Call the stop_bot function to ensure clean shutdown
        if "stop_bot" in function_map:
            function_map["stop_bot"]()

    console.print("[bold green]Thank you for using SolSniperX! Goodbye![/bold green]")


if __name__ == "__main__":
    # This allows testing the CLI directly
    from src.cli.cli_functions import register_all_functions
    register_all_functions()
    run_cli()
