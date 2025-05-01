"""
Gas optimization functions for the CLI interface of the Solana Memecoin Trading Bot.
"""

from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED
from datetime import datetime

from src.cli.cli_interface import console, wallet_connected
from src.solana.gas_optimizer import gas_optimizer, TransactionType, TransactionPriority
from config import get_config_value, update_config


def manage_gas():
    """Configure gas optimization settings."""
    console.print(Panel(
        "[bold white]Gas Optimization Configuration[/bold white]",
        title="[bold cyan]GAS OPTIMIZATION[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))

    # Check if gas optimization is enabled
    if not get_config_value("fee_optimization_enabled", True):
        console.print("[bold yellow]Gas optimization is currently disabled.[/bold yellow]")
        console.print("Would you like to enable it? (y/n): ", end="")
        choice = input().strip().lower()
        if choice in ["y", "yes"]:
            update_config("fee_optimization_enabled", True)
            console.print("[bold green]Gas optimization enabled![/bold green]")
        else:
            console.print("[bold yellow]Gas optimization remains disabled.[/bold yellow]")
            input("Press Enter to continue...")
            return True

    while True:
        # Create settings table
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
            ("1", "Minimum Priority Fee", str(get_config_value("min_priority_fee", "1000")), "Minimum priority fee in micro-lamports"),
            ("2", "Low Priority Percentile", str(get_config_value("low_priority_percentile", "25")), "Percentile for low priority transactions"),
            ("3", "Medium Priority Percentile", str(get_config_value("medium_priority_percentile", "50")), "Percentile for medium priority transactions"),
            ("4", "High Priority Percentile", str(get_config_value("high_priority_percentile", "75")), "Percentile for high priority transactions"),
            ("5", "Urgent Priority Percentile", str(get_config_value("urgent_priority_percentile", "90")), "Percentile for urgent transactions"),
            ("6", "Buy Fee Multiplier", str(get_config_value("buy_fee_multiplier", "1.0")), "Fee multiplier for buy transactions"),
            ("7", "Sell Fee Multiplier", str(get_config_value("sell_fee_multiplier", "1.0")), "Fee multiplier for sell transactions"),
            ("8", "Snipe Fee Multiplier", str(get_config_value("snipe_fee_multiplier", "2.0")), "Fee multiplier for snipe transactions"),
            ("9", "Swap Fee Multiplier", str(get_config_value("swap_fee_multiplier", "1.2")), "Fee multiplier for swap transactions"),
            ("A", "Limit Order Fee Multiplier", str(get_config_value("limit_order_fee_multiplier", "0.8")), "Fee multiplier for limit order transactions"),
            ("B", "Withdraw Fee Multiplier", str(get_config_value("withdraw_fee_multiplier", "1.5")), "Fee multiplier for withdraw transactions"),
            ("C", "Time-Based Fee Adjustment", str(get_config_value("time_based_fee_adjustment", "False")), "Adjust fees based on time of day"),
            ("D", "Urgent Sell Fee Boost", str(get_config_value("urgent_sell_fee_boost", "True")), "Boost fees for urgent sell transactions"),
            ("E", "Compute Unit Limit", str(get_config_value("compute_unit_limit", "200000")), "Default compute unit limit"),
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
            if setting[0] == choice:
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
                "Minimum Priority Fee": "min_priority_fee",
                "Low Priority Percentile": "low_priority_percentile",
                "Medium Priority Percentile": "medium_priority_percentile",
                "High Priority Percentile": "high_priority_percentile",
                "Urgent Priority Percentile": "urgent_priority_percentile",
                "Buy Fee Multiplier": "buy_fee_multiplier",
                "Sell Fee Multiplier": "sell_fee_multiplier",
                "Snipe Fee Multiplier": "snipe_fee_multiplier",
                "Swap Fee Multiplier": "swap_fee_multiplier",
                "Limit Order Fee Multiplier": "limit_order_fee_multiplier",
                "Withdraw Fee Multiplier": "withdraw_fee_multiplier",
                "Time-Based Fee Adjustment": "time_based_fee_adjustment",
                "Urgent Sell Fee Boost": "urgent_sell_fee_boost",
                "Compute Unit Limit": "compute_unit_limit",
            }

            config_key = config_key_map.get(setting_key)

            if config_key:
                # Convert value to appropriate type
                if config_key in ["time_based_fee_adjustment", "urgent_sell_fee_boost"]:
                    # Boolean settings
                    if new_value.lower() in ["true", "yes", "y", "1"]:
                        new_value = True
                    else:
                        new_value = False
                elif config_key in ["min_priority_fee", "low_priority_percentile", "medium_priority_percentile", 
                                   "high_priority_percentile", "urgent_priority_percentile", "compute_unit_limit"]:
                    # Integer values
                    try:
                        new_value = int(new_value)
                    except ValueError:
                        console.print("[bold red]Invalid value. Please enter a number.[/bold red]")
                        input("Press Enter to continue...")
                        continue
                else:
                    # Float values
                    try:
                        new_value = float(new_value)
                    except ValueError:
                        console.print("[bold red]Invalid value. Please enter a number.[/bold red]")
                        input("Press Enter to continue...")
                        continue

                # Update the config
                update_config(config_key, new_value)
                console.print(f"[bold green]Setting updated: {setting_key} = {new_value}[/bold green]")
                input("Press Enter to continue...")
            else:
                console.print("[bold red]Error: Could not map setting to config key.[/bold red]")
                input("Press Enter to continue...")
        else:
            console.print("[bold red]Invalid option. Please try again.[/bold red]")
            input("Press Enter to continue...")

    return True


def check_fees():
    """Check fee statistics and recommendations."""
    console.print(Panel(
        "[bold white]Fee Statistics and Recommendations[/bold white]",
        title="[bold cyan]FEE ANALYSIS[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))

    # Check if gas optimization is enabled
    if not get_config_value("fee_optimization_enabled", True):
        console.print("[bold yellow]Gas optimization is currently disabled.[/bold yellow]")
        console.print("Would you like to enable it? (y/n): ", end="")
        choice = input().strip().lower()
        if choice in ["y", "yes"]:
            update_config("fee_optimization_enabled", True)
            console.print("[bold green]Gas optimization enabled![/bold green]")
        else:
            console.print("[bold yellow]Gas optimization remains disabled. Cannot perform fee analysis.[/bold yellow]")
            input("Press Enter to continue...")
            return True

    # Get current fees
    current_fees = gas_optimizer._get_recent_fees()
    
    if not current_fees:
        console.print("[bold yellow]No recent fee data available. Collecting new data...[/bold yellow]")
        fee_data = gas_optimizer.collect_recent_fees()
        current_fees = fee_data.get("fees", {})
        
        if not current_fees:
            console.print("[bold red]Failed to collect fee data.[/bold red]")
            input("Press Enter to continue...")
            return True

    # Display current fees
    console.print("[bold]Current Network Fees (micro-lamports):[/bold]")
    
    fees_table = Table(
        show_header=True,
        header_style="bold white",
        box=ROUNDED,
        border_style="cyan",
        padding=(0, 1)
    )
    fees_table.add_column("PERCENTILE", style="yellow")
    fees_table.add_column("FEE", style="green", justify="right")
    fees_table.add_column("PRIORITY", style="cyan")
    
    percentiles = ["10", "25", "50", "75", "90", "99"]
    priority_map = {
        "25": "Low",
        "50": "Medium",
        "75": "High",
        "90": "Urgent"
    }
    
    for percentile in percentiles:
        fee = current_fees.get(percentile, "N/A")
        priority = priority_map.get(percentile, "")
        fees_table.add_row(
            percentile,
            str(fee),
            f"[bold cyan]{priority}[/bold cyan]" if priority else ""
        )
    
    console.print(fees_table)
    
    # Get network congestion
    congestion = gas_optimizer._get_network_congestion()
    congestion_level = "Low"
    congestion_color = "green"
    
    if congestion > 0.8:
        congestion_level = "High"
        congestion_color = "red"
    elif congestion > 0.5:
        congestion_level = "Medium"
        congestion_color = "yellow"
    elif congestion > 0.3:
        congestion_level = "Low-Medium"
        congestion_color = "cyan"
    
    console.print(f"\n[bold]Network Congestion:[/bold] [{congestion_color}]{congestion_level} ({congestion:.2f})[/{congestion_color}]")
    
    # Get fee statistics
    console.print("\n[bold]Collecting fee statistics...[/bold]")
    fee_stats = gas_optimizer.get_fee_stats(hours=24)
    
    if "error" in fee_stats:
        console.print(f"[bold red]Error getting fee statistics: {fee_stats['error']}[/bold red]")
    else:
        # Display fee statistics
        stats_table = Table(
            show_header=True,
            header_style="bold white",
            box=ROUNDED,
            border_style="blue",
            padding=(0, 1)
        )
        stats_table.add_column("PERCENTILE", style="yellow")
        stats_table.add_column("MIN", style="green", justify="right")
        stats_table.add_column("MAX", style="red", justify="right")
        stats_table.add_column("AVG", style="cyan", justify="right")
        stats_table.add_column("MEDIAN", style="magenta", justify="right")
        
        for percentile, stats in fee_stats["percentiles"].items():
            stats_table.add_row(
                percentile,
                str(stats["min"]),
                str(stats["max"]),
                f"{stats['avg']:.2f}",
                str(stats["median"])
            )
        
        console.print(Panel(
            stats_table,
            title=f"[bold blue]FEE STATISTICS (LAST {fee_stats['time_period_hours']} HOURS)[/bold blue]",
            border_style="blue",
            box=ROUNDED,
            padding=(1, 1)
        ))
    
    # Get transaction timing recommendations
    console.print("\n[bold]Transaction Timing Recommendations:[/bold]")
    
    timing_table = Table(
        show_header=True,
        header_style="bold white",
        box=ROUNDED,
        border_style="green",
        padding=(0, 1)
    )
    timing_table.add_column("URGENCY", style="yellow")
    timing_table.add_column("WAIT TIME", style="cyan", justify="right")
    timing_table.add_column("OPTIMAL TIME", style="green")
    timing_table.add_column("SAVINGS", style="magenta", justify="right")
    
    for urgency in ["immediate", "high", "normal", "low"]:
        timing = gas_optimizer.optimize_transaction_timing(urgency)
        
        if "error" in timing:
            timing_table.add_row(
                urgency.capitalize(),
                "Error",
                timing["error"],
                "N/A"
            )
        else:
            wait_minutes = timing["recommended_wait_minutes"]
            optimal_time = datetime.fromisoformat(timing["optimal_execution_time"]).strftime("%H:%M:%S")
            savings = timing["estimated_fee_savings_percent"]
            
            timing_table.add_row(
                urgency.capitalize(),
                f"{wait_minutes} min",
                optimal_time,
                f"{savings}%"
            )
    
    console.print(timing_table)
    
    # Display recommended fees for different transaction types
    console.print("\n[bold]Recommended Fees by Transaction Type:[/bold]")
    
    tx_table = Table(
        show_header=True,
        header_style="bold white",
        box=ROUNDED,
        border_style="magenta",
        padding=(0, 1)
    )
    tx_table.add_column("TRANSACTION TYPE", style="yellow")
    tx_table.add_column("LOW PRIORITY", style="green", justify="right")
    tx_table.add_column("MEDIUM PRIORITY", style="cyan", justify="right")
    tx_table.add_column("HIGH PRIORITY", style="magenta", justify="right")
    tx_table.add_column("URGENT", style="red", justify="right")
    
    for tx_type in [TransactionType.BUY, TransactionType.SELL, TransactionType.SNIPE, 
                   TransactionType.SWAP, TransactionType.LIMIT_ORDER, TransactionType.WITHDRAW]:
        tx_name = tx_type.value.capitalize()
        
        low_fee = gas_optimizer.get_priority_fee(TransactionPriority.LOW, tx_type)
        medium_fee = gas_optimizer.get_priority_fee(TransactionPriority.MEDIUM, tx_type)
        high_fee = gas_optimizer.get_priority_fee(TransactionPriority.HIGH, tx_type)
        urgent_fee = gas_optimizer.get_priority_fee(TransactionPriority.URGENT, tx_type)
        
        tx_table.add_row(
            tx_name,
            str(low_fee),
            str(medium_fee),
            str(high_fee),
            str(urgent_fee)
        )
    
    console.print(tx_table)
    
    input("\nPress Enter to continue...")
    return True
