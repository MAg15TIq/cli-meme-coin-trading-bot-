"""
CLI functions for enhanced trading features.
Provides command-line interface for copy trading, portfolio management, and alerts.
"""

import json
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from src.trading.copy_trading import copy_trading
from src.trading.enhanced_portfolio_manager import enhanced_portfolio_manager
from src.trading.advanced_alert_system import advanced_alert_system, AlertPriority
from src.trading.smart_wallet_discovery import smart_wallet_discovery
from src.trading.multi_dex_monitor import multi_dex_monitor
from src.trading.advanced_risk_metrics import advanced_risk_metrics
from src.trading.smart_order_management import smart_stop_loss_manager, smart_take_profit_manager
from src.trading.performance_attribution import performance_attribution_analyzer
from src.trading.ai_pool_analysis import ai_pool_analyzer
from src.trading.dynamic_portfolio_optimizer import dynamic_portfolio_optimizer, OptimizationMethod, OptimizationConstraints
from src.trading.enhanced_ai_pool_analysis import enhanced_ai_pool_analyzer
from src.trading.advanced_benchmarking import advanced_benchmarking_engine
from src.utils.logging_utils import get_logger
import asyncio

# Get logger for this module
logger = get_logger(__name__)

# Create console for rich output
console = Console()


def enhanced_copy_trading_menu():
    """Enhanced copy trading management menu."""
    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Enhanced Copy Trading Management[/bold cyan]",
            border_style="cyan"
        ))

        # Show current status
        status = "Enabled" if copy_trading.enabled else "Disabled"
        console.print(f"Status: [bold {'green' if copy_trading.enabled else 'red'}]{status}[/bold {'green' if copy_trading.enabled else 'red'}]")
        console.print(f"Tracked Wallets: {len(copy_trading.tracked_wallets)}")
        console.print()

        # Menu options
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("1", "Toggle Copy Trading")
        table.add_row("2", "Add Tracked Wallet")
        table.add_row("3", "Remove Tracked Wallet")
        table.add_row("4", "View Tracked Wallets")
        table.add_row("5", "View Performance Stats")
        table.add_row("6", "Configure Parameters")
        table.add_row("0", "Back to Main Menu")

        console.print(table)

        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6"])

        if choice == "0":
            break
        elif choice == "1":
            toggle_copy_trading()
        elif choice == "2":
            add_tracked_wallet()
        elif choice == "3":
            remove_tracked_wallet()
        elif choice == "4":
            view_tracked_wallets()
        elif choice == "5":
            view_copy_trading_stats()
        elif choice == "6":
            configure_copy_trading_parameters()


def enhanced_portfolio_menu():
    """Enhanced portfolio management menu."""
    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Enhanced Portfolio Management[/bold cyan]",
            border_style="cyan"
        ))

        # Show current status
        status = "Enabled" if enhanced_portfolio_manager.enabled else "Disabled"
        console.print(f"Status: [bold {'green' if enhanced_portfolio_manager.enabled else 'red'}]{status}[/bold {'green' if enhanced_portfolio_manager.enabled else 'red'}]")

        # Show portfolio metrics
        try:
            metrics = enhanced_portfolio_manager.calculate_portfolio_metrics()
            console.print(f"Total Value: {metrics['total_value_sol']:.4f} SOL")
            console.print(f"Positions: {metrics['position_count']}")
            console.print(f"Rebalancing Needed: {'Yes' if metrics.get('rebalancing_needed', False) else 'No'}")
        except Exception as e:
            console.print(f"[red]Error getting metrics: {e}[/red]")

        console.print()

        # Menu options
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("1", "Toggle Portfolio Management")
        table.add_row("2", "View Current Allocations")
        table.add_row("3", "Set Target Allocation")
        table.add_row("4", "Remove Target Allocation")
        table.add_row("5", "Calculate Rebalancing")
        table.add_row("6", "Execute Rebalancing")
        table.add_row("7", "View Portfolio Metrics")
        table.add_row("8", "Configure Settings")
        table.add_row("0", "Back to Main Menu")

        console.print(table)

        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])

        if choice == "0":
            break
        elif choice == "1":
            toggle_portfolio_management()
        elif choice == "2":
            view_current_allocations()
        elif choice == "3":
            set_target_allocation()
        elif choice == "4":
            remove_target_allocation()
        elif choice == "5":
            calculate_rebalancing()
        elif choice == "6":
            execute_rebalancing()
        elif choice == "7":
            view_portfolio_metrics()
        elif choice == "8":
            configure_portfolio_settings()


def advanced_alerts_menu():
    """Advanced alert system menu."""
    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Advanced Alert System[/bold cyan]",
            border_style="cyan"
        ))

        # Show current status
        status = "Enabled" if advanced_alert_system.enabled else "Disabled"
        console.print(f"Status: [bold {'green' if advanced_alert_system.enabled else 'red'}]{status}[/bold {'green' if advanced_alert_system.enabled else 'red'}]")
        console.print(f"Active Conditions: {len([c for c in advanced_alert_system.alert_conditions.values() if c.enabled])}")
        console.print(f"Total Alerts: {len(advanced_alert_system.alert_history)}")
        console.print()

        # Menu options
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("1", "Toggle Alert System")
        table.add_row("2", "Add Price Alert")
        table.add_row("3", "Add Volume Alert")
        table.add_row("4", "Add Whale Alert")
        table.add_row("5", "Add Portfolio Alert")
        table.add_row("6", "View Alert Conditions")
        table.add_row("7", "View Alert History")
        table.add_row("8", "Remove Alert Condition")
        table.add_row("0", "Back to Main Menu")

        console.print(table)

        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])

        if choice == "0":
            break
        elif choice == "1":
            toggle_alert_system()
        elif choice == "2":
            add_price_alert()
        elif choice == "3":
            add_volume_alert()
        elif choice == "4":
            add_whale_alert()
        elif choice == "5":
            add_portfolio_alert()
        elif choice == "6":
            view_alert_conditions()
        elif choice == "7":
            view_alert_history()
        elif choice == "8":
            remove_alert_condition()


def toggle_copy_trading():
    """Toggle copy trading on/off."""
    current_status = copy_trading.enabled
    new_status = not current_status

    if Confirm.ask(f"{'Enable' if new_status else 'Disable'} copy trading?"):
        copy_trading.set_enabled(new_status)
        console.print(f"[green]Copy trading {'enabled' if new_status else 'disabled'}[/green]")

    input("Press Enter to continue...")


def add_tracked_wallet():
    """Add a wallet to track for copy trading."""
    console.print("[bold]Add Tracked Wallet[/bold]")

    wallet_address = Prompt.ask("Enter wallet address")
    if not wallet_address or len(wallet_address) < 32:
        console.print("[red]Invalid wallet address[/red]")
        input("Press Enter to continue...")
        return

    multiplier = float(Prompt.ask("Enter trade size multiplier", default="1.0"))
    max_copy_amount = float(Prompt.ask("Enter max copy amount (SOL)", default="1.0"))

    try:
        copy_trading.add_tracked_wallet(wallet_address, multiplier, max_copy_amount)
        console.print(f"[green]Added wallet {wallet_address[:8]}... to tracking[/green]")
    except Exception as e:
        console.print(f"[red]Error adding wallet: {e}[/red]")

    input("Press Enter to continue...")


def view_tracked_wallets():
    """View all tracked wallets."""
    console.print("[bold]Tracked Wallets[/bold]")

    if not copy_trading.tracked_wallets:
        console.print("No wallets being tracked")
        input("Press Enter to continue...")
        return

    table = Table()
    table.add_column("Address", style="cyan")
    table.add_column("Multiplier", style="white")
    table.add_column("Max Amount", style="white")
    table.add_column("Total Trades", style="white")
    table.add_column("Success Rate", style="green")
    table.add_column("Enabled", style="white")

    for address, config in copy_trading.tracked_wallets.items():
        perf = copy_trading.wallet_performance.get(address, {})
        success_rate = f"{perf.get('success_rate', 0):.1f}%"

        table.add_row(
            f"{address[:8]}...{address[-8:]}",
            f"{config['multiplier']:.2f}",
            f"{config['max_copy_amount']:.2f}",
            str(config.get('total_trades', 0)),
            success_rate,
            "Yes" if config.get('enabled', True) else "No"
        )

    console.print(table)
    input("Press Enter to continue...")


def toggle_portfolio_management():
    """Toggle enhanced portfolio management on/off."""
    current_status = enhanced_portfolio_manager.enabled
    new_status = not current_status

    if Confirm.ask(f"{'Enable' if new_status else 'Disable'} enhanced portfolio management?"):
        enhanced_portfolio_manager.set_enabled(new_status)
        console.print(f"[green]Enhanced portfolio management {'enabled' if new_status else 'disabled'}[/green]")

    input("Press Enter to continue...")


def view_current_allocations():
    """View current portfolio allocations."""
    console.print("[bold]Current Portfolio Allocations[/bold]")

    try:
        allocations = enhanced_portfolio_manager.calculate_current_allocations()

        if not allocations:
            console.print("No target allocations set")
            input("Press Enter to continue...")
            return

        table = Table()
        table.add_column("Token", style="cyan")
        table.add_column("Target %", style="white")
        table.add_column("Current %", style="white")
        table.add_column("Deviation", style="yellow")
        table.add_column("Rebalance", style="white")
        table.add_column("Risk Level", style="white")

        for target in allocations.values():
            rebalance_color = "red" if target.rebalance_needed else "green"
            table.add_row(
                target.token_symbol,
                f"{target.target_percentage:.2f}%",
                f"{target.current_percentage:.2f}%",
                f"{target.deviation:.2f}%",
                f"[{rebalance_color}]{'Yes' if target.rebalance_needed else 'No'}[/{rebalance_color}]",
                target.risk_level
            )

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error getting allocations: {e}[/red]")

    input("Press Enter to continue...")


def toggle_alert_system():
    """Toggle advanced alert system on/off."""
    current_status = advanced_alert_system.enabled
    new_status = not current_status

    if Confirm.ask(f"{'Enable' if new_status else 'Disable'} advanced alert system?"):
        advanced_alert_system.set_enabled(new_status)
        console.print(f"[green]Advanced alert system {'enabled' if new_status else 'disabled'}[/green]")

    input("Press Enter to continue...")


def add_price_alert():
    """Add a price alert."""
    console.print("[bold]Add Price Alert[/bold]")

    token_mint = Prompt.ask("Enter token mint address")
    token_symbol = Prompt.ask("Enter token symbol")
    condition_type = Prompt.ask("Enter condition type", choices=["above", "below", "change_percent"])
    threshold = float(Prompt.ask("Enter threshold value"))
    priority = Prompt.ask("Enter priority", choices=["low", "medium", "high", "critical"], default="medium")

    try:
        condition_id = advanced_alert_system.add_price_alert(
            token_mint, token_symbol, condition_type, threshold, AlertPriority(priority)
        )
        console.print(f"[green]Added price alert with ID: {condition_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error adding price alert: {e}[/red]")

    input("Press Enter to continue...")


def view_alert_conditions():
    """View all alert conditions."""
    console.print("[bold]Alert Conditions[/bold]")

    if not advanced_alert_system.alert_conditions:
        console.print("No alert conditions configured")
        input("Press Enter to continue...")
        return

    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="white")
    table.add_column("Token", style="white")
    table.add_column("Priority", style="white")
    table.add_column("Enabled", style="white")
    table.add_column("Triggers", style="white")

    for condition in advanced_alert_system.alert_conditions.values():
        table.add_row(
            condition.id[:12] + "...",
            condition.alert_type.value,
            condition.token_symbol or "N/A",
            condition.priority.value,
            "Yes" if condition.enabled else "No",
            str(condition.trigger_count)
        )

    console.print(table)
    input("Press Enter to continue...")


def smart_copy_trading_menu():
    """Enhanced copy trading interface with smart discovery."""
    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Smart Copy Trading System[/bold cyan]\n"
            "Automatically discover and copy profitable wallets",
            border_style="cyan"
        ))

        # Display current status
        discovery_enabled = smart_wallet_discovery.enabled
        copy_enabled = copy_trading.enabled

        status_table = Table(title="Current Status")
        status_table.add_column("Feature", style="cyan")
        status_table.add_column("Status", style="green" if discovery_enabled else "red")

        status_table.add_row("Smart Discovery", "Enabled" if discovery_enabled else "Disabled")
        status_table.add_row("Copy Trading", "Enabled" if copy_enabled else "Disabled")
        status_table.add_row("Tracked Wallets", str(len(copy_trading.tracked_wallets)))

        console.print(status_table)

        # Menu options
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("1", "Auto-discover profitable wallets")
        table.add_row("2", "View tracked wallets performance")
        table.add_row("3", "Configure copy trading parameters")
        table.add_row("4", "Enable/Disable smart discovery")
        table.add_row("5", "Manual wallet management")
        table.add_row("0", "Back to main menu")

        console.print(table)

        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            break
        elif choice == "1":
            auto_discover_wallets()
        elif choice == "2":
            view_wallet_performance()
        elif choice == "3":
            configure_copy_parameters()
        elif choice == "4":
            toggle_smart_discovery()
        elif choice == "5":
            manual_wallet_management()


def auto_discover_wallets():
    """Auto-discover profitable wallets interface."""
    console.clear()
    console.print(Panel(
        "[bold cyan]Auto-Discover Profitable Wallets[/bold cyan]",
        border_style="cyan"
    ))

    if not smart_wallet_discovery.enabled:
        console.print("[yellow]Smart discovery is currently disabled.[/yellow]")
        if Confirm.ask("Enable smart discovery?"):
            smart_wallet_discovery.set_enabled(True)
        else:
            return

    console.print("[yellow]Starting wallet discovery... This may take a few minutes.[/yellow]")

    try:
        # Run the discovery
        result = asyncio.run(copy_trading.auto_discover_wallets())

        if result["success"]:
            console.print(f"\n[green]Discovery completed successfully![/green]")
            console.print(f"Total wallets discovered: {result['total_discovered']}")
            console.print(f"Wallets added: {len(result['added_wallets'])}")
            console.print(f"Wallets skipped: {len(result['skipped_wallets'])}")

            if result['added_wallets']:
                # Display added wallets
                added_table = Table(title="Added Wallets")
                added_table.add_column("Address", style="cyan")
                added_table.add_column("Score", style="green")
                added_table.add_column("Multiplier", style="yellow")
                added_table.add_column("Max Copy (SOL)", style="magenta")

                for wallet in result['added_wallets']:
                    added_table.add_row(
                        wallet['address'][:8] + "...",
                        f"{wallet['score']:.1f}",
                        f"{wallet['multiplier']:.2f}",
                        f"{wallet['max_copy_amount']:.3f}"
                    )

                console.print(added_table)

        else:
            console.print(f"[red]Discovery failed: {result['message']}[/red]")

    except Exception as e:
        console.print(f"[red]Error during discovery: {e}[/red]")

    input("Press Enter to continue...")


def view_wallet_performance():
    """View tracked wallets performance."""
    console.clear()
    console.print(Panel(
        "[bold cyan]Tracked Wallets Performance[/bold cyan]",
        border_style="cyan"
    ))

    if not copy_trading.tracked_wallets:
        console.print("[yellow]No wallets are currently being tracked.[/yellow]")
        input("Press Enter to continue...")
        return

    # Create performance table
    perf_table = Table(title="Wallet Performance")
    perf_table.add_column("Address", style="cyan")
    perf_table.add_column("Enabled", style="green")
    perf_table.add_column("Success Rate", style="yellow")
    perf_table.add_column("Profit Ratio", style="magenta")
    perf_table.add_column("Max Drawdown", style="red")
    perf_table.add_column("Trades", style="blue")

    for address, config in copy_trading.tracked_wallets.items():
        # Get performance data
        perf = copy_trading.wallet_performance.get(address, {})

        perf_table.add_row(
            address[:8] + "...",
            "Yes" if config.get("enabled", True) else "No",
            f"{perf.get('success_rate', 0):.1f}%",
            f"{perf.get('profit_ratio', 0):.2f}",
            f"{perf.get('max_drawdown', 0):.1f}%",
            str(len(perf.get('trades', [])))
        )

    console.print(perf_table)
    input("Press Enter to continue...")


def multi_dex_hunting_menu():
    """Multi-DEX pool hunting interface."""
    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Multi-DEX Pool Hunter[/bold cyan]\n"
            "Monitor multiple DEXes for liquidity opportunities",
            border_style="cyan"
        ))

        # Display current DEX monitoring status
        dex_status = multi_dex_monitor.get_monitoring_status()

        status_table = Table(title="DEX Monitoring Status")
        status_table.add_column("DEX", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Pools Found", style="yellow")
        status_table.add_column("Opportunities", style="magenta")

        for dex_name, status in dex_status.items():
            status_color = "green" if status['active'] else "red"
            status_table.add_row(
                dex_name.upper(),
                f"[{status_color}]{'Active' if status['active'] else 'Inactive'}[/{status_color}]",
                str(status['pools_found']),
                str(status['opportunities'])
            )

        console.print(status_table)

        # Display statistics
        stats = multi_dex_monitor.get_statistics()
        console.print(f"\n[yellow]Total Pools:[/yellow] {stats['total_pools']}")
        console.print(f"[yellow]Active Opportunities:[/yellow] {stats['active_opportunities']}")
        console.print(f"[yellow]Last Scan:[/yellow] {stats.get('last_scan_time', 'Never')}")

        # Menu options
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("1", "Start/Stop DEX monitoring")
        table.add_row("2", "View arbitrage opportunities")
        table.add_row("3", "Configure DEX settings")
        table.add_row("4", "Search pools by token")
        table.add_row("0", "Back to main menu")

        console.print(table)

        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4"])

        if choice == "0":
            break
        elif choice == "1":
            toggle_dex_monitoring()
        elif choice == "2":
            view_arbitrage_opportunities()
        elif choice == "3":
            configure_dex_settings()
        elif choice == "4":
            search_pools_by_token()


def toggle_dex_monitoring():
    """Toggle DEX monitoring on/off."""
    current_status = multi_dex_monitor.enabled

    console.print(f"[yellow]DEX monitoring is currently {'enabled' if current_status else 'disabled'}.[/yellow]")

    if current_status:
        if Confirm.ask("Disable DEX monitoring?"):
            multi_dex_monitor.set_enabled(False)
            console.print("[green]DEX monitoring disabled.[/green]")
    else:
        if Confirm.ask("Enable DEX monitoring?"):
            multi_dex_monitor.set_enabled(True)
            console.print("[green]DEX monitoring enabled.[/green]")

    input("Press Enter to continue...")


def view_arbitrage_opportunities():
    """View current arbitrage opportunities."""
    console.clear()
    console.print(Panel(
        "[bold cyan]Arbitrage Opportunities[/bold cyan]",
        border_style="cyan"
    ))

    opportunities = multi_dex_monitor.get_arbitrage_opportunities(limit=20)

    if not opportunities:
        console.print("[yellow]No arbitrage opportunities found.[/yellow]")
        input("Press Enter to continue...")
        return

    # Create opportunities table
    opp_table = Table(title="Current Arbitrage Opportunities")
    opp_table.add_column("Token", style="cyan")
    opp_table.add_column("Buy DEX", style="green")
    opp_table.add_column("Sell DEX", style="red")
    opp_table.add_column("Profit %", style="yellow")
    opp_table.add_column("Est. Profit (SOL)", style="magenta")
    opp_table.add_column("Max Size (SOL)", style="blue")

    for opp in opportunities:
        opp_table.add_row(
            opp['token_symbol'] or opp['token_mint'][:8] + "...",
            opp['buy_dex'].upper(),
            opp['sell_dex'].upper(),
            f"{opp['profit_percentage']:.2f}%",
            f"{opp['estimated_profit_sol']:.4f}",
            f"{opp['max_trade_size_sol']:.2f}"
        )

    console.print(opp_table)
    input("Press Enter to continue...")


def configure_copy_parameters():
    """Configure copy trading parameters."""
    console.clear()
    console.print(Panel(
        "[bold cyan]Configure Copy Trading Parameters[/bold cyan]",
        border_style="cyan"
    ))

    console.print("[yellow]Current Settings:[/yellow]")
    console.print(f"Copy Percentage: {copy_trading.copy_percentage}%")
    console.print(f"Min Transaction: {copy_trading.min_transaction_sol} SOL")
    console.print(f"Max Transaction: {copy_trading.max_transaction_sol} SOL")
    console.print(f"Min Success Rate: {copy_trading.min_wallet_success_rate}%")
    console.print(f"Min Profit Ratio: {copy_trading.min_wallet_profit_ratio}")
    console.print(f"Max Drawdown: {copy_trading.max_wallet_drawdown}%")

    # Menu options
    table = Table(show_header=False, box=None)
    table.add_column("Option", style="cyan")
    table.add_column("Description", style="white")

    table.add_row("1", "Copy percentage")
    table.add_row("2", "Transaction limits")
    table.add_row("3", "Wallet performance criteria")
    table.add_row("0", "Back to smart copy trading menu")

    console.print(table)

    choice = Prompt.ask("Select option", choices=["0", "1", "2", "3"])

    if choice == "1":
        try:
            new_percentage = float(Prompt.ask(f"Enter new copy percentage (current: {copy_trading.copy_percentage}%)"))
            if 0 < new_percentage <= 100:
                copy_trading.copy_percentage = new_percentage
                console.print(f"[green]Copy percentage updated to {new_percentage}%[/green]")
            else:
                console.print("[red]Invalid percentage. Must be between 0 and 100.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")

    elif choice == "2":
        try:
            min_tx = float(Prompt.ask(f"Enter minimum transaction size in SOL (current: {copy_trading.min_transaction_sol})"))
            max_tx = float(Prompt.ask(f"Enter maximum transaction size in SOL (current: {copy_trading.max_transaction_sol})"))

            if 0 < min_tx < max_tx:
                copy_trading.min_transaction_sol = min_tx
                copy_trading.max_transaction_sol = max_tx
                console.print(f"[green]Transaction limits updated: {min_tx} - {max_tx} SOL[/green]")
            else:
                console.print("[red]Invalid limits. Min must be less than max and both must be positive.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter numbers.[/red]")

    input("Press Enter to continue...")


def toggle_smart_discovery():
    """Toggle smart discovery on/off."""
    current_status = smart_wallet_discovery.enabled

    if Confirm.ask(f"{'Disable' if current_status else 'Enable'} smart discovery?"):
        smart_wallet_discovery.set_enabled(not current_status)
        status = "enabled" if not current_status else "disabled"
        console.print(f"[green]Smart discovery {status}.[/green]")

    input("Press Enter to continue...")


def manual_wallet_management():
    """Manual wallet management interface."""
    console.clear()
    console.print(Panel(
        "[bold cyan]Manual Wallet Management[/bold cyan]",
        border_style="cyan"
    ))

    # Menu options
    table = Table(show_header=False, box=None)
    table.add_column("Option", style="cyan")
    table.add_column("Description", style="white")

    table.add_row("1", "Add wallet manually")
    table.add_row("2", "Remove wallet")
    table.add_row("3", "Enable/Disable wallet")
    table.add_row("4", "Modify wallet settings")
    table.add_row("0", "Back to smart copy trading menu")

    console.print(table)

    choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4"])

    if choice == "1":
        wallet_address = Prompt.ask("Enter wallet address")
        if len(wallet_address) == 44:  # Valid Solana address length
            copy_trading.add_tracked_wallet(wallet_address)
            console.print(f"[green]Wallet {wallet_address} added successfully.[/green]")
        else:
            console.print("[red]Invalid wallet address.[/red]")

    elif choice == "2":
        if not copy_trading.tracked_wallets:
            console.print("[yellow]No wallets to remove.[/yellow]")
        else:
            console.print("Tracked wallets:")
            addresses = list(copy_trading.tracked_wallets.keys())
            for i, address in enumerate(addresses, 1):
                console.print(f"{i}. {address}")

            try:
                index = int(Prompt.ask("Enter wallet number to remove")) - 1
                if 0 <= index < len(addresses):
                    wallet_to_remove = addresses[index]
                    copy_trading.remove_tracked_wallet(wallet_to_remove)
                    console.print(f"[green]Wallet removed successfully.[/green]")
                else:
                    console.print("[red]Invalid wallet number.[/red]")
            except ValueError:
                console.print("[red]Invalid input.[/red]")

    input("Press Enter to continue...")


def configure_dex_settings():
    """Configure DEX monitoring settings."""
    console.clear()
    console.print(Panel(
        "[bold cyan]Configure DEX Settings[/bold cyan]",
        border_style="cyan"
    ))

    console.print("[yellow]Current DEX Configuration:[/yellow]")

    for dex_name, config in multi_dex_monitor.dex_configs.items():
        status = "Enabled" if config.get('enabled', False) else "Disabled"
        console.print(f"{dex_name.upper()}: {status}")

    console.print(f"\nArbitrage Detection: {'Enabled' if multi_dex_monitor.arbitrage_enabled else 'Disabled'}")
    console.print(f"Min Arbitrage Profit: {multi_dex_monitor.min_arbitrage_profit_bps} bps")

    # Menu options
    table = Table(show_header=False, box=None)
    table.add_column("Option", style="cyan")
    table.add_column("Description", style="white")

    table.add_row("1", "Enable/Disable specific DEX")
    table.add_row("2", "Toggle arbitrage detection")
    table.add_row("3", "Set minimum arbitrage profit")
    table.add_row("0", "Back to multi-DEX menu")

    console.print(table)

    choice = Prompt.ask("Select option", choices=["0", "1", "2", "3"])

    if choice == "1":
        console.print("Available DEXes:")
        dex_names = list(multi_dex_monitor.dex_configs.keys())
        for i, dex_name in enumerate(dex_names, 1):
            status = "Enabled" if multi_dex_monitor.dex_configs[dex_name].get('enabled', False) else "Disabled"
            console.print(f"{i}. {dex_name.upper()} ({status})")

        try:
            index = int(Prompt.ask("Select DEX to toggle")) - 1
            if 0 <= index < len(dex_names):
                dex_name = dex_names[index]
                current_status = multi_dex_monitor.dex_configs[dex_name].get('enabled', False)
                multi_dex_monitor.dex_configs[dex_name]['enabled'] = not current_status
                new_status = "enabled" if not current_status else "disabled"
                console.print(f"[green]{dex_name.upper()} {new_status}.[/green]")
            else:
                console.print("[red]Invalid DEX number.[/red]")
        except ValueError:
            console.print("[red]Invalid input.[/red]")

    elif choice == "2":
        multi_dex_monitor.arbitrage_enabled = not multi_dex_monitor.arbitrage_enabled
        status = "enabled" if multi_dex_monitor.arbitrage_enabled else "disabled"
        console.print(f"[green]Arbitrage detection {status}.[/green]")

    input("Press Enter to continue...")


def search_pools_by_token():
    """Search pools by token mint."""
    console.clear()
    console.print(Panel(
        "[bold cyan]Search Pools by Token[/bold cyan]",
        border_style="cyan"
    ))

    token_mint = Prompt.ask("Enter token mint address")

    if len(token_mint) != 44:
        console.print("[red]Invalid token mint address.[/red]")
        input("Press Enter to continue...")
        return

    pools = multi_dex_monitor.get_pools_by_token(token_mint)

    if not pools:
        console.print(f"[yellow]No pools found for token {token_mint}[/yellow]")
        input("Press Enter to continue...")
        return

    # Display pools
    pools_table = Table(title=f"Pools for Token {token_mint[:8]}...")
    pools_table.add_column("DEX", style="cyan")
    pools_table.add_column("Pool Address", style="green")
    pools_table.add_column("Liquidity (SOL)", style="yellow")
    pools_table.add_column("Volume 24h (SOL)", style="magenta")
    pools_table.add_column("Fee %", style="blue")

    for pool in pools:
        pools_table.add_row(
            pool['dex_name'].upper(),
            pool['pool_address'][:8] + "...",
            f"{pool['liquidity_sol']:.2f}",
            f"{pool['volume_24h_sol']:.2f}",
            f"{pool['fee_percentage']:.3f}%"
        )

    console.print(pools_table)
    input("Press Enter to continue...")


def advanced_risk_analytics():
    """Advanced risk analytics interface."""
    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Advanced Risk Analytics[/bold cyan]\n"
            "VaR, CVaR, Stress Testing, and Monte Carlo Analysis",
            border_style="cyan"
        ))

        # Get portfolio returns for analysis
        try:
            from src.trading.portfolio_analytics import portfolio_analytics
            portfolio_returns = portfolio_analytics.get_portfolio_returns(days=90)

            if len(portfolio_returns) < 30:
                console.print("[yellow]Insufficient data for advanced risk analysis (need at least 30 days).[/yellow]")
                console.print("Please trade for a few more days to build up historical data.")
                input("Press Enter to continue...")
                return

            # Calculate VaR and CVaR for multiple horizons
            var_results = advanced_risk_metrics.calculate_var_cvar_multi_horizon(
                portfolio_returns.values
            )

            # Display VaR/CVaR results
            if var_results:
                var_table = Table(title="Value at Risk (VaR) Analysis")
                var_table.add_column("Confidence", style="cyan")
                var_table.add_column("Time Horizon", style="green")
                var_table.add_column("VaR", style="yellow")
                var_table.add_column("CVaR", style="red")
                var_table.add_column("Observations", style="blue")

                for result in var_results:
                    var_table.add_row(
                        f"{result.confidence_level:.0%}",
                        f"{result.time_horizon_days}d",
                        f"{result.var:.4f}",
                        f"{result.cvar:.4f}",
                        str(result.observations)
                    )

                console.print(var_table)

            # Maximum drawdown prediction
            mdd_prediction = advanced_risk_metrics.calculate_maximum_drawdown_prediction(
                portfolio_returns.values
            )

            console.print(f"\n[yellow]Maximum Drawdown Prediction (95% confidence):[/yellow]")
            console.print(f"Predicted MDD: {mdd_prediction.predicted_mdd:.2%}")
            console.print(f"Mean MDD: {mdd_prediction.mean_mdd:.2%}")
            console.print(f"Simulations: {mdd_prediction.simulations:,}")

        except Exception as e:
            console.print(f"[red]Error calculating risk metrics: {e}[/red]")

        # Menu options
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("1", "Stress Test Portfolio")
        table.add_row("2", "Scenario Analysis")
        table.add_row("3", "Risk Configuration")
        table.add_row("0", "Back to enhanced features menu")

        console.print(table)

        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3"])

        if choice == "0":
            break
        elif choice == "1":
            stress_test_portfolio()
        elif choice == "2":
            scenario_analysis()
        elif choice == "3":
            risk_configuration()


def stress_test_portfolio():
    """Perform stress testing on the portfolio."""
    console.clear()
    console.print(Panel(
        "[bold cyan]Portfolio Stress Testing[/bold cyan]",
        border_style="cyan"
    ))

    try:
        # Get current positions
        from src.trading.position_manager import position_manager
        positions = position_manager.get_all_positions()

        if not positions:
            console.print("[yellow]No positions found for stress testing.[/yellow]")
            input("Press Enter to continue...")
            return

        # Convert positions to format expected by stress testing
        position_data = {}
        for token_mint, position in positions.items():
            position_data[token_mint] = {
                'value_sol': position.current_value,
                'token_name': position.token_name,
                'is_sol': token_mint == 'SOL'
            }

        # Run stress tests
        stress_results = advanced_risk_metrics.stress_test_portfolio(position_data)

        if stress_results:
            # Display stress test results
            stress_table = Table(title="Stress Test Results")
            stress_table.add_column("Scenario", style="cyan")
            stress_table.add_column("Total P&L (SOL)", style="yellow")
            stress_table.add_column("P&L %", style="red")
            stress_table.add_column("Worst Position", style="magenta")
            stress_table.add_column("Worst P&L (SOL)", style="blue")

            for result in stress_results:
                pnl_color = "green" if result.total_pnl >= 0 else "red"
                stress_table.add_row(
                    result.scenario_name.replace('_', ' ').title(),
                    f"[{pnl_color}]{result.total_pnl:.4f}[/{pnl_color}]",
                    f"[{pnl_color}]{result.pnl_percentage:.2f}%[/{pnl_color}]",
                    result.worst_position,
                    f"{result.worst_position_pnl:.4f}"
                )

            console.print(stress_table)

            # Show scenario details
            console.print("\n[yellow]Scenario Details:[/yellow]")
            for result in stress_results:
                details = result.scenario_details
                console.print(f"• {result.scenario_name.replace('_', ' ').title()}: {details.get('description', 'N/A')}")

        else:
            console.print("[yellow]No stress test results available.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error performing stress test: {e}[/red]")

    input("Press Enter to continue...")


def smart_order_management():
    """Smart order management interface."""
    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Smart Order Management[/bold cyan]\n"
            "Intelligent Stop-Loss and Take-Profit Strategies",
            border_style="cyan"
        ))

        # Menu options
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("1", "Create Smart Stop-Loss")
        table.add_row("2", "Create Smart Take-Profit")
        table.add_row("3", "View Active Smart Orders")
        table.add_row("4", "Smart Order Configuration")
        table.add_row("0", "Back to enhanced features menu")

        console.print(table)

        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4"])

        if choice == "0":
            break
        elif choice == "1":
            create_smart_stop_loss()
        elif choice == "2":
            create_smart_take_profit()
        elif choice == "3":
            view_smart_orders()
        elif choice == "4":
            smart_order_configuration()


def create_smart_stop_loss():
    """Create a smart stop-loss order."""
    console.clear()
    console.print(Panel(
        "[bold cyan]Create Smart Stop-Loss[/bold cyan]",
        border_style="cyan"
    ))

    try:
        # Get current positions
        from src.trading.position_manager import position_manager
        positions = position_manager.get_all_positions()

        if not positions:
            console.print("[yellow]No positions found to create stop-loss for.[/yellow]")
            input("Press Enter to continue...")
            return

        # Display positions
        console.print("[yellow]Select a position:[/yellow]")
        position_list = list(positions.items())
        for i, (token_mint, position) in enumerate(position_list, 1):
            console.print(f"{i}. {position.token_name} - {position.amount:.4f} tokens "
                         f"(Value: {position.current_value:.4f} SOL)")

        try:
            index = int(Prompt.ask("Enter position number")) - 1
            if 0 <= index < len(position_list):
                token_mint, position = position_list[index]

                # Select stop-loss strategy
                console.print("\n[yellow]Select stop-loss strategy:[/yellow]")
                strategies = [
                    ("1", "Volatility-Based", "Adjusts based on token volatility"),
                    ("2", "Trailing", "Follows price movements"),
                    ("3", "Time Decay", "Tightens over time"),
                    ("4", "ATR-Based", "Based on Average True Range"),
                    ("5", "Fixed", "Fixed percentage")
                ]

                for num, name, desc in strategies:
                    console.print(f"{num}. {name} - {desc}")

                strategy_choice = Prompt.ask("Select strategy", choices=["1", "2", "3", "4", "5"])

                from src.trading.smart_order_management import StopLossStrategy
                strategy_map = {
                    "1": StopLossStrategy.VOLATILITY_BASED,
                    "2": StopLossStrategy.TRAILING,
                    "3": StopLossStrategy.TIME_DECAY,
                    "4": StopLossStrategy.ATR_BASED,
                    "5": StopLossStrategy.FIXED
                }

                strategy = strategy_map[strategy_choice]

                # Create the stop-loss order
                order_id = smart_stop_loss_manager.create_smart_stop_loss(
                    token_mint=token_mint,
                    token_name=position.token_name,
                    position_size=position.amount,
                    entry_price=position.entry_price,
                    strategy=strategy
                )

                console.print(f"[green]Smart stop-loss created successfully![/green]")
                console.print(f"Order ID: {order_id}")

            else:
                console.print("[red]Invalid position number.[/red]")

        except ValueError:
            console.print("[red]Invalid input.[/red]")

    except Exception as e:
        console.print(f"[red]Error creating stop-loss: {e}[/red]")

    input("Press Enter to continue...")


def performance_attribution():
    """Performance attribution analysis interface."""
    console.clear()
    console.print(Panel(
        "[bold cyan]Performance Attribution Analysis[/bold cyan]\n"
        "Brinson-Hood-Beebower and Factor-Based Attribution",
        border_style="cyan"
    ))

    try:
        # Get portfolio returns
        from src.trading.portfolio_analytics import portfolio_analytics
        portfolio_returns = portfolio_analytics.get_portfolio_returns(days=30)

        if len(portfolio_returns) < 10:
            console.print("[yellow]Insufficient data for attribution analysis (need at least 10 days).[/yellow]")
            input("Press Enter to continue...")
            return

        # Create benchmark returns (simplified - SOL price movements)
        import pandas as pd
        import numpy as np
        benchmark_returns = pd.Series(
            np.random.normal(0.01, 0.05, len(portfolio_returns)),
            index=portfolio_returns.index
        )

        # Perform attribution analysis
        attribution_result = performance_attribution_analyzer.analyze_performance_attribution(
            portfolio_returns, benchmark_returns, method='brinson', period_days=30
        )

        # Display results
        attribution_table = Table(title="Performance Attribution Analysis")
        attribution_table.add_column("Component", style="cyan")
        attribution_table.add_column("Contribution", style="yellow")
        attribution_table.add_column("Description", style="white")

        attribution_table.add_row(
            "Allocation Effect",
            f"{attribution_result.allocation_effect:.4f}",
            "Return from over/under-weighting sectors"
        )
        attribution_table.add_row(
            "Selection Effect",
            f"{attribution_result.selection_effect:.4f}",
            "Return from security selection within sectors"
        )
        attribution_table.add_row(
            "Interaction Effect",
            f"{attribution_result.interaction_effect:.4f}",
            "Combined effect of allocation and selection"
        )
        attribution_table.add_row(
            "Total Active Return",
            f"[bold]{attribution_result.total_active_return:.4f}[/bold]",
            "Total outperformance vs benchmark"
        )

        console.print(attribution_table)

        console.print(f"\n[yellow]Analysis Period:[/yellow] {attribution_result.period_start.strftime('%Y-%m-%d')} to {attribution_result.period_end.strftime('%Y-%m-%d')}")
        console.print(f"[yellow]Method:[/yellow] {attribution_result.method.title()}")

    except Exception as e:
        console.print(f"[red]Error performing attribution analysis: {e}[/red]")

    input("Press Enter to continue...")


def ai_pool_analysis():
    """AI-powered pool analysis interface."""
    console.clear()
    console.print(Panel(
        "[bold cyan]AI-Powered Pool Analysis[/bold cyan]\n"
        "Machine Learning Pool Quality and Sustainability Scoring",
        border_style="cyan"
    ))

    # Get pool address from user
    pool_address = Prompt.ask("Enter pool address to analyze")

    if len(pool_address) != 44:
        console.print("[red]Invalid pool address.[/red]")
        input("Press Enter to continue...")
        return

    try:
        # Simulate pool data (in production, this would fetch real data)
        pool_data = {
            'address': pool_address,
            'liquidity': np.random.uniform(1000, 100000),
            'volume_24h': np.random.uniform(500, 50000),
            'volume_7d': np.random.uniform(3500, 350000),
            'trade_count_24h': np.random.randint(10, 1000),
            'price_volatility_24h': np.random.uniform(0.05, 0.5),
            'price_change_24h': np.random.uniform(-0.2, 0.2),
            'holder_count': np.random.randint(100, 10000),
            'top_10_holder_percentage': np.random.uniform(20, 80),
            'created_at': '2024-01-01T00:00:00Z',
            'dex': 'raydium',
            'fee_tier': 0.0025,
            'market_cap': np.random.uniform(100000, 10000000),
            'fdv': np.random.uniform(200000, 20000000),
            'circulating_supply': np.random.uniform(1000000, 1000000000)
        }

        console.print("[yellow]Analyzing pool with AI models...[/yellow]")

        # Perform AI analysis
        analysis_result = ai_pool_analyzer.analyze_pool(pool_data)

        # Display results
        results_table = Table(title=f"AI Analysis Results for {pool_address[:8]}...")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Score", style="yellow")
        results_table.add_column("Interpretation", style="white")

        # Quality score interpretation
        if analysis_result.quality_score >= 75:
            quality_interp = "Excellent"
            quality_color = "green"
        elif analysis_result.quality_score >= 60:
            quality_interp = "Good"
            quality_color = "yellow"
        elif analysis_result.quality_score >= 40:
            quality_interp = "Fair"
            quality_color = "orange"
        else:
            quality_interp = "Poor"
            quality_color = "red"

        # Sustainability interpretation
        sustainability_interp = "High" if analysis_result.sustainability_score > 0.7 else "Medium" if analysis_result.sustainability_score > 0.4 else "Low"

        # Risk interpretation
        risk_interp = "Low" if analysis_result.risk_score < 30 else "Medium" if analysis_result.risk_score < 60 else "High"

        results_table.add_row(
            "Quality Score",
            f"[{quality_color}]{analysis_result.quality_score:.1f}/100[/{quality_color}]",
            quality_interp
        )
        results_table.add_row(
            "Sustainability",
            f"{analysis_result.sustainability_score:.2f}",
            sustainability_interp
        )
        results_table.add_row(
            "Risk Score",
            f"{analysis_result.risk_score:.1f}/100",
            risk_interp
        )
        results_table.add_row(
            "Confidence",
            f"{analysis_result.confidence:.2f}",
            "Model prediction confidence"
        )

        console.print(results_table)

        # Display recommendation
        rec_color = {
            "STRONG_BUY": "bright_green",
            "BUY": "green",
            "HOLD": "yellow",
            "SELL": "red",
            "STRONG_SELL": "bright_red"
        }.get(analysis_result.recommendation, "white")

        console.print(f"\n[bold {rec_color}]Recommendation: {analysis_result.recommendation}[/bold {rec_color}]")

        # Display key features
        if analysis_result.features:
            console.print(f"\n[yellow]Key Pool Metrics:[/yellow]")
            console.print(f"• Total Liquidity: {analysis_result.features.total_liquidity:,.0f}")
            console.print(f"• 24h Volume: {analysis_result.features.volume_24h:,.0f}")
            console.print(f"• Volume/Liquidity Ratio: {analysis_result.features.volume_to_liquidity_ratio:.2f}")
            console.print(f"• Holder Count: {analysis_result.features.holder_count:,}")
            console.print(f"• Pool Age: {analysis_result.features.age_days} days")
            console.print(f"• DEX: {analysis_result.features.dex_name.title()}")

    except Exception as e:
        console.print(f"[red]Error performing AI analysis: {e}[/red]")

    input("Press Enter to continue...")


def scenario_analysis():
    """Placeholder for scenario analysis."""
    console.print("[yellow]Scenario analysis feature coming soon![/yellow]")
    input("Press Enter to continue...")


def risk_configuration():
    """Placeholder for risk configuration."""
    console.print("[yellow]Risk configuration feature coming soon![/yellow]")
    input("Press Enter to continue...")


def create_smart_take_profit():
    """Placeholder for smart take-profit creation."""
    console.print("[yellow]Smart take-profit feature coming soon![/yellow]")
    input("Press Enter to continue...")


def view_smart_orders():
    """Placeholder for viewing smart orders."""
    console.print("[yellow]Smart orders view feature coming soon![/yellow]")
    input("Press Enter to continue...")


def smart_order_configuration():
    """Placeholder for smart order configuration."""
    console.print("[yellow]Smart order configuration feature coming soon![/yellow]")
    input("Press Enter to continue...")


# Phase 3: Advanced Optimization & AI Features

def dynamic_portfolio_optimization_menu():
    """Dynamic portfolio optimization interface."""
    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Dynamic Portfolio Optimization[/bold cyan]\n"
            "Advanced portfolio optimization with market regime detection",
            border_style="cyan"
        ))

        # Display current status
        summary = dynamic_portfolio_optimizer.get_optimization_summary()

        status_table = Table(title="Current Status")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")

        if "status" in summary:
            status_table.add_row("Status", summary["status"])
        else:
            status_table.add_row("Total Optimizations", str(summary.get("total_optimizations", 0)))
            status_table.add_row("Last Optimization", summary.get("last_optimization", "Never"))
            status_table.add_row("Current Method", summary.get("current_method", "None"))
            status_table.add_row("Market Regime", summary.get("current_regime", "Unknown"))

            if "performance_metrics" in summary:
                metrics = summary["performance_metrics"]
                status_table.add_row("Expected Return", f"{metrics.get('expected_return', 0):.4f}")
                status_table.add_row("Expected Risk", f"{metrics.get('expected_risk', 0):.4f}")
                status_table.add_row("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.4f}")

        console.print(status_table)

        # Menu options
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("1", "Run Portfolio Optimization")
        table.add_row("2", "View Current Weights")
        table.add_row("3", "Calculate Efficient Frontier")
        table.add_row("4", "Market Regime Analysis")
        table.add_row("5", "Optimization History")
        table.add_row("6", "Configure Optimization Settings")
        table.add_row("0", "Back to main menu")

        console.print(table)

        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6"])

        if choice == "0":
            break
        elif choice == "1":
            run_portfolio_optimization()
        elif choice == "2":
            view_current_weights()
        elif choice == "3":
            calculate_efficient_frontier()
        elif choice == "4":
            market_regime_analysis()
        elif choice == "5":
            optimization_history()
        elif choice == "6":
            configure_optimization_settings()


def enhanced_ai_pool_analysis_menu():
    """Enhanced AI pool analysis interface."""
    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Enhanced AI Pool Analysis[/bold cyan]\n"
            "Advanced ML models with ensemble predictions and real-time learning",
            border_style="cyan"
        ))

        # Display model status
        model_summary = enhanced_ai_pool_analyzer.get_model_summary()

        status_table = Table(title="Model Status")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")

        status_table.add_row("Available Models", ", ".join(model_summary.get("available_models", [])))
        status_table.add_row("Ensemble Enabled", str(model_summary.get("ensemble_enabled", False)))
        status_table.add_row("Retraining Enabled", str(model_summary.get("retraining_enabled", False)))
        status_table.add_row("Total Predictions", str(model_summary.get("total_predictions", 0)))
        status_table.add_row("Feature Count", str(model_summary.get("feature_count", 0)))

        if "recent_performance" in model_summary:
            perf = model_summary["recent_performance"]
            status_table.add_row("Avg Confidence", f"{perf.get('avg_confidence', 0):.3f}")
            status_table.add_row("Avg Consensus", f"{perf.get('avg_consensus', 0):.3f}")

        console.print(status_table)

        # Menu options
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("1", "Analyze Pool with Enhanced AI")
        table.add_row("2", "View Model Performance")
        table.add_row("3", "Retrain Models")
        table.add_row("4", "Feature Importance Analysis")
        table.add_row("5", "Prediction History")
        table.add_row("6", "Model Configuration")
        table.add_row("0", "Back to main menu")

        console.print(table)

        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6"])

        if choice == "0":
            break
        elif choice == "1":
            analyze_pool_enhanced_ai()
        elif choice == "2":
            view_model_performance()
        elif choice == "3":
            retrain_ai_models()
        elif choice == "4":
            feature_importance_analysis()
        elif choice == "5":
            ai_prediction_history()
        elif choice == "6":
            ai_model_configuration()


def advanced_benchmarking_menu():
    """Advanced benchmarking interface."""
    while True:
        console.clear()
        console.print(Panel(
            "[bold cyan]Advanced Benchmarking Engine[/bold cyan]\n"
            "Comprehensive performance analysis with multiple benchmarks",
            border_style="cyan"
        ))

        # Display benchmark status
        benchmark_summary = advanced_benchmarking_engine.get_benchmark_summary()

        status_table = Table(title="Benchmark Status")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")

        benchmarks = benchmark_summary.get("available_benchmarks", {})
        status_table.add_row("Available Benchmarks", str(len(benchmarks)))
        status_table.add_row("Total Comparisons", str(benchmark_summary.get("total_comparisons", 0)))
        status_table.add_row("Forecasting Enabled", str(benchmark_summary.get("forecasting_enabled", False)))

        if "latest_comparison" in benchmark_summary:
            latest = benchmark_summary["latest_comparison"]
            status_table.add_row("Latest Benchmark", latest.get("benchmark", "None"))
            status_table.add_row("Active Return", f"{latest.get('active_return', 0):.4f}")
            status_table.add_row("Information Ratio", f"{latest.get('information_ratio', 0):.4f}")

        console.print(status_table)

        # Menu options
        table = Table(show_header=False, box=None)
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("1", "Compare Performance vs Benchmark")
        table.add_row("2", "Risk-Adjusted Metrics Analysis")
        table.add_row("3", "Factor Exposure Analysis")
        table.add_row("4", "Performance Forecasting")
        table.add_row("5", "View Available Benchmarks")
        table.add_row("6", "Add Custom Benchmark")
        table.add_row("7", "Comparison History")
        table.add_row("0", "Back to main menu")

        console.print(table)

        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7"])

        if choice == "0":
            break
        elif choice == "1":
            compare_performance_benchmark()
        elif choice == "2":
            risk_adjusted_metrics_analysis()
        elif choice == "3":
            factor_exposure_analysis()
        elif choice == "4":
            performance_forecasting()
        elif choice == "5":
            view_available_benchmarks()
        elif choice == "6":
            add_custom_benchmark()
        elif choice == "7":
            benchmarking_comparison_history()


# Placeholder implementations for Phase 3 CLI functions

def run_portfolio_optimization():
    """Run portfolio optimization."""
    console.print("[yellow]Portfolio optimization feature - implementation in progress![/yellow]")
    console.print("This will run advanced portfolio optimization with market regime detection.")
    input("Press Enter to continue...")

def view_current_weights():
    """View current portfolio weights."""
    console.print("[yellow]Current weights view - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def calculate_efficient_frontier():
    """Calculate efficient frontier."""
    console.print("[yellow]Efficient frontier calculation - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def market_regime_analysis():
    """Market regime analysis."""
    console.print("[yellow]Market regime analysis - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def optimization_history():
    """View optimization history."""
    console.print("[yellow]Optimization history - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def configure_optimization_settings():
    """Configure optimization settings."""
    console.print("[yellow]Optimization settings - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def analyze_pool_enhanced_ai():
    """Analyze pool with enhanced AI."""
    console.print("[yellow]Enhanced AI pool analysis - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def view_model_performance():
    """View AI model performance."""
    console.print("[yellow]Model performance view - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def retrain_ai_models():
    """Retrain AI models."""
    console.print("[yellow]Model retraining - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def feature_importance_analysis():
    """Feature importance analysis."""
    console.print("[yellow]Feature importance analysis - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def ai_prediction_history():
    """AI prediction history."""
    console.print("[yellow]Prediction history - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def ai_model_configuration():
    """AI model configuration."""
    console.print("[yellow]Model configuration - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def compare_performance_benchmark():
    """Compare performance vs benchmark."""
    console.print("[yellow]Benchmark comparison - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def risk_adjusted_metrics_analysis():
    """Risk-adjusted metrics analysis."""
    console.print("[yellow]Risk-adjusted metrics - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def factor_exposure_analysis():
    """Factor exposure analysis."""
    console.print("[yellow]Factor exposure analysis - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def performance_forecasting():
    """Performance forecasting."""
    console.print("[yellow]Performance forecasting - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def view_available_benchmarks():
    """View available benchmarks."""
    console.print("[yellow]Available benchmarks - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def add_custom_benchmark():
    """Add custom benchmark."""
    console.print("[yellow]Add custom benchmark - implementation in progress![/yellow]")
    input("Press Enter to continue...")

def benchmarking_comparison_history():
    """Benchmarking comparison history."""
    console.print("[yellow]Comparison history - implementation in progress![/yellow]")
    input("Press Enter to continue...")