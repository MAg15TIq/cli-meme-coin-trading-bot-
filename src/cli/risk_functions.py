"""
Risk management functions for the CLI interface of the Solana Memecoin Trading Bot.
"""

from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED

from src.cli.cli_interface import console, wallet_connected
from src.trading.risk_manager import risk_manager
from src.trading.position_manager import position_manager
from config import get_config_value, update_config


def manage_risk():
    """Configure risk management settings."""
    console.print(Panel(
        "[bold white]Risk Management Configuration[/bold white]",
        title="[bold cyan]RISK MANAGEMENT[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))

    # Check if risk management is enabled
    if not get_config_value("risk_management_enabled", True):
        console.print("[bold yellow]Risk management is currently disabled.[/bold yellow]")
        console.print("Would you like to enable it? (y/n): ", end="")
        choice = input().strip().lower()
        if choice in ["y", "yes"]:
            update_config("risk_management_enabled", True)
            console.print("[bold green]Risk management enabled![/bold green]")
        else:
            console.print("[bold yellow]Risk management remains disabled.[/bold yellow]")
            input("Press Enter to continue...")
            return True

    while True:
        # Get current risk profile
        current_profile = get_config_value("risk_profile", "moderate")

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
            ("1", "Risk Profile", current_profile.capitalize(), "Overall risk profile (conservative, moderate, aggressive)"),
            ("2", "Max Portfolio Allocation", f"{get_config_value(f'{current_profile}_max_allocation_percent', '40.0')}%", "Maximum percentage of portfolio for risky assets"),
            ("3", "Max Position Size", f"{get_config_value(f'{current_profile}_max_position_percent', '5.0')}%", "Maximum percentage of portfolio for a single position"),
            ("4", "Stop Loss Percentage", f"{get_config_value(f'{current_profile}_stop_loss_percent', '10.0')}%", "Default stop-loss percentage for positions"),
            ("5", "Max Drawdown", f"{get_config_value(f'{current_profile}_max_drawdown_percent', '20.0')}%", "Maximum acceptable drawdown before reducing exposure"),
            ("6", "Low Risk Allocation", f"{get_config_value('max_low_risk_allocation_percent', '60.0')}%", "Maximum allocation for low-risk tokens"),
            ("7", "Medium Risk Allocation", f"{get_config_value('max_medium_risk_allocation_percent', '30.0')}%", "Maximum allocation for medium-risk tokens"),
            ("8", "High Risk Allocation", f"{get_config_value('max_high_risk_allocation_percent', '10.0')}%", "Maximum allocation for high-risk tokens"),
            ("9", "Very High Risk Allocation", f"{get_config_value('max_very_high_risk_allocation_percent', '5.0')}%", "Maximum allocation for very high-risk tokens"),
        ]

        for setting in settings:
            settings_table.add_row(*setting)

        console.print(settings_table)
        console.print("\n[bold]Enter setting number to change, or 'Q' to return to main menu:[/bold] ", end="")

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
                "Risk Profile": "risk_profile",
                "Max Portfolio Allocation": f"{current_profile}_max_allocation_percent",
                "Max Position Size": f"{current_profile}_max_position_percent",
                "Stop Loss Percentage": f"{current_profile}_stop_loss_percent",
                "Max Drawdown": f"{current_profile}_max_drawdown_percent",
                "Low Risk Allocation": "max_low_risk_allocation_percent",
                "Medium Risk Allocation": "max_medium_risk_allocation_percent",
                "High Risk Allocation": "max_high_risk_allocation_percent",
                "Very High Risk Allocation": "max_very_high_risk_allocation_percent",
            }

            config_key = config_key_map.get(setting_key)

            if config_key:
                # Convert value to appropriate type
                if config_key == "risk_profile":
                    # Validate risk profile
                    if new_value.lower() not in ["conservative", "moderate", "aggressive"]:
                        console.print("[bold red]Invalid risk profile. Must be 'conservative', 'moderate', or 'aggressive'.[/bold red]")
                        input("Press Enter to continue...")
                        continue
                    new_value = new_value.lower()
                else:
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
                input("Press Enter to continue...")
            else:
                console.print("[bold red]Error: Could not map setting to config key.[/bold red]")
                input("Press Enter to continue...")
        else:
            console.print("[bold red]Invalid option. Please try again.[/bold red]")
            input("Press Enter to continue...")

    return True


def check_portfolio():
    """Check portfolio risk assessment."""
    if not wallet_connected:
        console.print("[bold red]No wallet connected. Please connect a wallet first.[/bold red]")
        input("Press Enter to continue...")
        return True

    console.print(Panel(
        "[bold white]Portfolio Risk Assessment[/bold white]",
        title="[bold cyan]PORTFOLIO RISK[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    ))

    # Check if risk management is enabled
    if not get_config_value("risk_management_enabled", True):
        console.print("[bold yellow]Risk management is currently disabled.[/bold yellow]")
        console.print("Would you like to enable it? (y/n): ", end="")
        choice = input().strip().lower()
        if choice in ["y", "yes"]:
            update_config("risk_management_enabled", True)
            console.print("[bold green]Risk management enabled![/bold green]")
        else:
            console.print("[bold yellow]Risk management remains disabled. Cannot perform risk assessment.[/bold yellow]")
            input("Press Enter to continue...")
            return True

    # Get portfolio risk assessment
    risk_assessment = risk_manager.check_portfolio_risk()

    if "error" in risk_assessment:
        console.print(f"[bold red]Error getting portfolio risk assessment: {risk_assessment['error']}[/bold red]")
        input("Press Enter to continue...")
        return True

    # Display portfolio metrics
    console.print(f"[bold]Portfolio Value:[/bold] [green]{risk_assessment['portfolio_value_sol']:.4f} SOL[/green]")
    console.print(f"[bold]Portfolio Drawdown:[/bold] [{'red' if risk_assessment['drawdown_exceeded'] else 'green'}]{risk_assessment['portfolio_drawdown']:.2f}%[/{'red' if risk_assessment['drawdown_exceeded'] else 'green'}] (Max: {risk_assessment['max_drawdown_percent']:.2f}%)")

    # Display risk allocation
    allocation_table = Table(
        show_header=True,
        header_style="bold white",
        box=ROUNDED,
        border_style="cyan",
        padding=(0, 1)
    )
    allocation_table.add_column("RISK LEVEL", style="yellow")
    allocation_table.add_column("ALLOCATION", style="green", justify="right")
    allocation_table.add_column("PERCENTAGE", style="cyan", justify="right")
    allocation_table.add_column("MAX ALLOWED", style="magenta", justify="right")
    allocation_table.add_column("STATUS", style="bold white", justify="center")

    for risk_level, data in risk_assessment["risk_allocation"].items():
        status = "[red]EXCEEDED[/red]" if data["exceeded"] else "[green]OK[/green]"
        allocation_table.add_row(
            risk_level.capitalize(),
            f"{data['value_sol']:.4f} SOL",
            f"{data['percentage']:.2f}%",
            f"{data['max_percentage']:.2f}%",
            status
        )

    console.print(Panel(
        allocation_table,
        title="[bold cyan]RISK ALLOCATION[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 1)
    ))

    # Display recommendations if any
    if risk_assessment["recommendations"]:
        recommendations_table = Table(
            show_header=True,
            header_style="bold white",
            box=ROUNDED,
            border_style="yellow",
            padding=(0, 1)
        )
        recommendations_table.add_column("TYPE", style="yellow")
        recommendations_table.add_column("MESSAGE", style="white")
        recommendations_table.add_column("ACTION", style="green")

        for rec in risk_assessment["recommendations"]:
            recommendations_table.add_row(
                rec["type"].capitalize(),
                rec["message"],
                rec["action"]
            )

        console.print(Panel(
            recommendations_table,
            title="[bold yellow]RECOMMENDATIONS[/bold yellow]",
            border_style="yellow",
            box=ROUNDED,
            padding=(1, 1)
        ))
    else:
        console.print("[bold green]No risk issues detected. Portfolio is within risk parameters.[/bold green]")

    # Display positions with risk levels
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
        positions_table.add_column("VALUE", style="green", justify="right")
        positions_table.add_column("% OF PORTFOLIO", style="cyan", justify="right")
        positions_table.add_column("RISK LEVEL", style="yellow", justify="center")
        positions_table.add_column("MAX LOSS", style="red", justify="right")

        for position in positions:
            position_value = position.get_value()
            portfolio_percentage = (position_value / risk_assessment["portfolio_value_sol"]) * 100 if risk_assessment["portfolio_value_sol"] > 0 else 0
            
            # Set color based on risk level
            risk_color = "green"
            if position.risk_level.value == "medium":
                risk_color = "yellow"
            elif position.risk_level.value == "high":
                risk_color = "orange"
            elif position.risk_level.value == "very_high":
                risk_color = "red"
                
            positions_table.add_row(
                f"[bold magenta]{position.token_symbol}[/bold magenta]",
                f"{position_value:.4f} SOL",
                f"{portfolio_percentage:.2f}%",
                f"[{risk_color}]{position.risk_level.value.upper()}[/{risk_color}]",
                f"{position.max_loss_sol:.4f} SOL"
            )

        console.print(Panel(
            positions_table,
            title="[bold green]POSITIONS BY RISK LEVEL[/bold green]",
            border_style="green",
            box=ROUNDED,
            padding=(1, 1)
        ))

    input("\nPress Enter to continue...")
    return True
