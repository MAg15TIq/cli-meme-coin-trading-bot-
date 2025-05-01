"""
Refinement functions for the CLI interface of the Solana Memecoin Trading Bot.
"""

from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED
from rich.prompt import Confirm
from datetime import datetime

from src.cli.cli_interface import console, wallet_connected
from src.trading.risk_manager import risk_manager
from src.solana.gas_optimizer import gas_optimizer
from src.utils.performance_tracker import performance_tracker
from config import get_config_value, update_config


def manage_refinement():
    """Configure and manage parameter refinement."""
    console.print(Panel(
        "[bold white]Parameter Refinement Configuration[/bold white]",
        title="[bold cyan]PARAMETER REFINEMENT[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))
    
    # Display current settings
    settings_table = Table(box=ROUNDED, show_header=True, header_style="bold cyan")
    settings_table.add_column("SETTING", style="white")
    settings_table.add_column("VALUE", style="green")
    
    auto_refinement = get_config_value("auto_refinement_enabled", False)
    risk_interval = get_config_value("risk_refinement_interval_days", "7")
    gas_interval = get_config_value("gas_refinement_interval_days", "3")
    
    settings_table.add_row("Auto-Refinement", "Enabled" if auto_refinement else "Disabled")
    settings_table.add_row("Risk Refinement Interval", f"{risk_interval} days")
    settings_table.add_row("Gas Refinement Interval", f"{gas_interval} days")
    
    # Add last refinement times
    last_risk = float(get_config_value("last_risk_refinement", "0"))
    last_gas = float(get_config_value("last_gas_refinement", "0"))
    
    if last_risk > 0:
        last_risk_time = datetime.fromtimestamp(last_risk).strftime("%Y-%m-%d %H:%M:%S")
        settings_table.add_row("Last Risk Refinement", last_risk_time)
    else:
        settings_table.add_row("Last Risk Refinement", "Never")
    
    if last_gas > 0:
        last_gas_time = datetime.fromtimestamp(last_gas).strftime("%Y-%m-%d %H:%M:%S")
        settings_table.add_row("Last Gas Refinement", last_gas_time)
    else:
        settings_table.add_row("Last Gas Refinement", "Never")
    
    console.print(settings_table)
    
    # Display performance metrics
    metrics_table = Table(box=ROUNDED, show_header=True, header_style="bold cyan")
    metrics_table.add_column("METRIC", style="white")
    metrics_table.add_column("VALUE", style="yellow")
    
    metrics_table.add_row("Win Rate", f"{performance_tracker.win_rate:.2%}")
    metrics_table.add_row("Average P/L", f"{performance_tracker.avg_profit_loss:.2f}%")
    metrics_table.add_row("Max Drawdown", f"{performance_tracker.max_drawdown:.2f}%")
    metrics_table.add_row("Transaction Success Rate", f"{performance_tracker.transaction_success_rate:.2%}")
    metrics_table.add_row("Closed Trades", str(len([t for t in performance_tracker.trades if t.get("status") == "closed"])))
    metrics_table.add_row("Total Transactions", str(len(performance_tracker.transactions)))
    
    console.print(metrics_table)
    
    # Menu options
    console.print("\n[bold cyan]Options:[/bold cyan]")
    console.print("1. Toggle Auto-Refinement")
    console.print("2. Set Risk Refinement Interval")
    console.print("3. Set Gas Refinement Interval")
    console.print("4. Run Risk Parameter Refinement Now")
    console.print("5. Run Gas Parameter Refinement Now")
    console.print("6. Clear Performance Data")
    console.print("7. Back to Main Menu")
    
    choice = console.input("\n[bold cyan]Enter your choice (1-7): [/bold cyan]")
    
    if choice == "1":
        # Toggle auto-refinement
        new_value = not auto_refinement
        update_config("auto_refinement_enabled", new_value)
        performance_tracker.auto_refinement_enabled = new_value
        console.print(f"[green]Auto-refinement {'enabled' if new_value else 'disabled'}[/green]")
    
    elif choice == "2":
        # Set risk refinement interval
        new_interval = console.input("[bold cyan]Enter risk refinement interval in days (1-30): [/bold cyan]")
        try:
            interval = int(new_interval)
            if 1 <= interval <= 30:
                update_config("risk_refinement_interval_days", str(interval))
                performance_tracker.risk_refinement_interval_days = interval
                console.print(f"[green]Risk refinement interval set to {interval} days[/green]")
            else:
                console.print("[red]Invalid interval. Must be between 1 and 30 days.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
    
    elif choice == "3":
        # Set gas refinement interval
        new_interval = console.input("[bold cyan]Enter gas refinement interval in days (1-30): [/bold cyan]")
        try:
            interval = int(new_interval)
            if 1 <= interval <= 30:
                update_config("gas_refinement_interval_days", str(interval))
                performance_tracker.gas_refinement_interval_days = interval
                console.print(f"[green]Gas refinement interval set to {interval} days[/green]")
            else:
                console.print("[red]Invalid interval. Must be between 1 and 30 days.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
    
    elif choice == "4":
        # Run risk parameter refinement
        run_risk_refinement()
    
    elif choice == "5":
        # Run gas parameter refinement
        run_gas_refinement()
    
    elif choice == "6":
        # Clear performance data
        clear_data_menu()
    
    elif choice == "7":
        # Back to main menu
        return
    
    # Return to refinement menu
    manage_refinement()


def run_risk_refinement():
    """Run risk parameter refinement."""
    console.print(Panel(
        "[bold white]Risk Parameter Refinement[/bold white]",
        title="[bold cyan]RISK REFINEMENT[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))
    
    # Get refinement data
    refinement_data = performance_tracker.get_risk_refinement_data()
    
    # Check if we have enough data
    if "error" in refinement_data:
        console.print(f"[red]Error getting refinement data: {refinement_data['error']}[/red]")
        return
    
    if refinement_data.get("trade_count", 0) < 5:
        console.print("[yellow]Warning: Not enough trade data for meaningful refinement.[/yellow]")
        console.print(f"[yellow]Only {refinement_data.get('trade_count', 0)} closed trades available. Recommend at least 5 trades.[/yellow]")
        
        if not Confirm.ask("[bold cyan]Continue with refinement anyway?[/bold cyan]"):
            console.print("[yellow]Refinement cancelled.[/yellow]")
            return
    
    # Display refinement data summary
    console.print(f"[cyan]Trade Count:[/cyan] {refinement_data.get('trade_count', 0)}")
    console.print(f"[cyan]Win Rate:[/cyan] {refinement_data.get('win_rate', 0):.2%}")
    console.print(f"[cyan]Average P/L:[/cyan] {refinement_data.get('avg_profit_loss', 0):.2f}%")
    console.print(f"[cyan]Max Drawdown:[/cyan] {refinement_data.get('portfolio_drawdown_max', 0):.2f}%")
    
    # Confirm refinement
    if not Confirm.ask("[bold cyan]Run risk parameter refinement with this data?[/bold cyan]"):
        console.print("[yellow]Refinement cancelled.[/yellow]")
        return
    
    # Run refinement
    console.print("[cyan]Running risk parameter refinement...[/cyan]")
    result = risk_manager.refine_risk_parameters(refinement_data)
    
    if not result.get("success", False):
        console.print(f"[red]Refinement failed: {result.get('reason', result.get('error', 'Unknown error'))}[/red]")
        return
    
    # Display results
    console.print("[green]Risk parameter refinement completed successfully![/green]")
    
    # Display allocation adjustments
    if "allocation_adjustments" in result and result["allocation_adjustments"]:
        console.print("\n[bold cyan]Risk Allocation Adjustments:[/bold cyan]")
        
        adjustments_table = Table(box=ROUNDED, show_header=True, header_style="bold cyan")
        adjustments_table.add_column("RISK LEVEL", style="white")
        adjustments_table.add_column("ORIGINAL", style="yellow")
        adjustments_table.add_column("NEW", style="green")
        adjustments_table.add_column("CHANGE", style="magenta")
        
        for risk_level, adjustment in result["allocation_adjustments"].items():
            change = adjustment.get("change_percent", 0)
            change_str = f"{change:+.2f}%"
            change_style = "green" if change > 0 else "red" if change < 0 else "white"
            
            adjustments_table.add_row(
                risk_level.capitalize(),
                f"{adjustment.get('original', 0):.2f}%",
                f"{adjustment.get('new', 0):.2f}%",
                f"[{change_style}]{change_str}[/{change_style}]"
            )
        
        console.print(adjustments_table)
    
    # Display profile adjustments
    if "profile_adjustments" in result and result["profile_adjustments"]:
        console.print("\n[bold cyan]Risk Profile Adjustments:[/bold cyan]")
        
        profile_table = Table(box=ROUNDED, show_header=True, header_style="bold cyan")
        profile_table.add_column("PROFILE", style="white")
        profile_table.add_column("PARAMETER", style="yellow")
        profile_table.add_column("ORIGINAL", style="yellow")
        profile_table.add_column("NEW", style="green")
        
        for profile, adjustment in result["profile_adjustments"].items():
            profile_table.add_row(
                profile.capitalize(),
                adjustment.get("parameter", ""),
                f"{adjustment.get('original', 0):.2f}%",
                f"{adjustment.get('new', 0):.2f}%"
            )
        
        console.print(profile_table)
    
    # Record refinement
    performance_tracker.record_refinement("risk")
    console.print("\n[green]Refinement recorded. New parameters are now in effect.[/green]")


def run_gas_refinement():
    """Run gas parameter refinement."""
    console.print(Panel(
        "[bold white]Gas Parameter Refinement[/bold white]",
        title="[bold cyan]GAS REFINEMENT[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))
    
    # Get refinement data
    refinement_data = performance_tracker.get_gas_refinement_data()
    
    # Check if we have enough data
    if "error" in refinement_data:
        console.print(f"[red]Error getting refinement data: {refinement_data['error']}[/red]")
        return
    
    if refinement_data.get("transaction_count", 0) < 10:
        console.print("[yellow]Warning: Not enough transaction data for meaningful refinement.[/yellow]")
        console.print(f"[yellow]Only {refinement_data.get('transaction_count', 0)} transactions available. Recommend at least 10 transactions.[/yellow]")
        
        if not Confirm.ask("[bold cyan]Continue with refinement anyway?[/bold cyan]"):
            console.print("[yellow]Refinement cancelled.[/yellow]")
            return
    
    # Display refinement data summary
    console.print(f"[cyan]Transaction Count:[/cyan] {refinement_data.get('transaction_count', 0)}")
    console.print(f"[cyan]Success Rate:[/cyan] {refinement_data.get('success_rate', 0):.2%}")
    
    # Confirm refinement
    if not Confirm.ask("[bold cyan]Run gas parameter refinement with this data?[/bold cyan]"):
        console.print("[yellow]Refinement cancelled.[/yellow]")
        return
    
    # Run refinement
    console.print("[cyan]Running gas parameter refinement...[/cyan]")
    result = gas_optimizer.refine_fee_parameters(refinement_data.get("transactions", []))
    
    if not result.get("success", False):
        console.print(f"[red]Refinement failed: {result.get('reason', result.get('error', 'Unknown error'))}[/red]")
        return
    
    # Display results
    console.print("[green]Gas parameter refinement completed successfully![/green]")
    
    # Display multiplier adjustments
    if "multiplier_adjustments" in result and result["multiplier_adjustments"]:
        console.print("\n[bold cyan]Fee Multiplier Adjustments:[/bold cyan]")
        
        multiplier_table = Table(box=ROUNDED, show_header=True, header_style="bold cyan")
        multiplier_table.add_column("TX TYPE", style="white")
        multiplier_table.add_column("ORIGINAL", style="yellow")
        multiplier_table.add_column("NEW", style="green")
        multiplier_table.add_column("CHANGE", style="magenta")
        multiplier_table.add_column("SUCCESS RATE", style="cyan")
        
        for tx_type, adjustment in result["multiplier_adjustments"].items():
            change = adjustment.get("change_percent", 0)
            change_str = f"{change:+.2f}%"
            change_style = "green" if change > 0 else "red" if change < 0 else "white"
            
            multiplier_table.add_row(
                tx_type.capitalize(),
                f"{adjustment.get('original', 0):.2f}x",
                f"{adjustment.get('new', 0):.2f}x",
                f"[{change_style}]{change_str}[/{change_style}]",
                f"{adjustment.get('success_rate', 0):.2%}"
            )
        
        console.print(multiplier_table)
    
    # Display time adjustments
    if "time_adjustments" in result and result["time_adjustments"]:
        console.print("\n[bold cyan]Time-Based Adjustments:[/bold cyan]")
        
        for time_period, adjustment in result["time_adjustments"].items():
            console.print(f"[yellow]{time_period.upper()}:[/yellow] {adjustment.get('recommendation', '')}")
            
            if "problematic_hours" in adjustment:
                hours_str = ", ".join(str(h) for h in adjustment["problematic_hours"])
                console.print(f"[yellow]Problematic Hours:[/yellow] {hours_str} UTC")
    
    # Record refinement
    performance_tracker.record_refinement("gas")
    console.print("\n[green]Refinement recorded. New parameters are now in effect.[/green]")


def clear_data_menu():
    """Menu for clearing performance data."""
    console.print(Panel(
        "[bold white]Clear Performance Data[/bold white]",
        title="[bold cyan]CLEAR DATA[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))
    
    console.print("[bold yellow]Warning: Clearing data will remove all performance history used for refinement.[/bold yellow]")
    console.print("[bold yellow]This action cannot be undone.[/bold yellow]\n")
    
    console.print("[bold cyan]Options:[/bold cyan]")
    console.print("1. Clear Trade Data")
    console.print("2. Clear Transaction Data")
    console.print("3. Clear Portfolio Metrics")
    console.print("4. Clear All Data")
    console.print("5. Back to Refinement Menu")
    
    choice = console.input("\n[bold cyan]Enter your choice (1-5): [/bold cyan]")
    
    if choice == "1":
        if Confirm.ask("[bold red]Are you sure you want to clear all trade data?[/bold red]"):
            if performance_tracker.clear_data("trades"):
                console.print("[green]Trade data cleared successfully.[/green]")
            else:
                console.print("[red]Error clearing trade data.[/red]")
    
    elif choice == "2":
        if Confirm.ask("[bold red]Are you sure you want to clear all transaction data?[/bold red]"):
            if performance_tracker.clear_data("transactions"):
                console.print("[green]Transaction data cleared successfully.[/green]")
            else:
                console.print("[red]Error clearing transaction data.[/red]")
    
    elif choice == "3":
        if Confirm.ask("[bold red]Are you sure you want to clear all portfolio metrics?[/bold red]"):
            if performance_tracker.clear_data("portfolio"):
                console.print("[green]Portfolio metrics cleared successfully.[/green]")
            else:
                console.print("[red]Error clearing portfolio metrics.[/red]")
    
    elif choice == "4":
        if Confirm.ask("[bold red]Are you sure you want to clear ALL performance data?[/bold red]"):
            if performance_tracker.clear_data():
                console.print("[green]All performance data cleared successfully.[/green]")
            else:
                console.print("[red]Error clearing performance data.[/red]")
    
    elif choice == "5":
        return
    
    # Return to clear data menu unless option 5 was chosen
    if choice != "5":
        clear_data_menu()


def check_auto_refinement():
    """Check if auto-refinement should be performed and run if needed."""
    if not performance_tracker.auto_refinement_enabled:
        return
    
    refinement_checks = performance_tracker.check_auto_refinement()
    
    if refinement_checks["risk"]:
        console.print("[cyan]Auto-refinement: Running scheduled risk parameter refinement...[/cyan]")
        run_risk_refinement()
    
    if refinement_checks["gas"]:
        console.print("[cyan]Auto-refinement: Running scheduled gas parameter refinement...[/cyan]")
        run_gas_refinement()
