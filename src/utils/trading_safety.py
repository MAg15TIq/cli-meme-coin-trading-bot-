"""
Trading Safety Module for the Solana Memecoin Trading Bot.
Provides safety checks, warnings, and risk management for live trading.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from config import get_config_value
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)
console = Console()


class TradingSafetyManager:
    """Manager for trading safety checks and warnings."""
    
    def __init__(self):
        """Initialize the trading safety manager."""
        self.safety_checks_enabled = get_config_value("trading_safety_checks_enabled", True)
        self.require_risk_acknowledgment = get_config_value("require_risk_acknowledgment", True)
        self.min_wallet_balance_sol = get_config_value("min_wallet_balance_sol", 0.01)
        
    def check_live_trading_prerequisites(self) -> Tuple[bool, str]:
        """
        Check if all prerequisites for live trading are met.
        
        Returns:
            Tuple of (is_ready, error_message)
        """
        try:
            # Check if wallet is connected
            from src.wallet.wallet import wallet_manager
            if not wallet_manager.current_keypair:
                return False, "No wallet connected. Please connect a wallet before enabling live trading."
            
            # Check wallet balance
            try:
                balance = wallet_manager.get_sol_balance()
                if balance < self.min_wallet_balance_sol:
                    return False, f"Insufficient SOL balance ({balance:.4f}). Need at least {self.min_wallet_balance_sol} SOL for transaction fees."
            except Exception as e:
                logger.warning(f"Could not check wallet balance: {e}")
                return False, f"Could not verify wallet balance: {e}"
            
            # Check if Jupiter API is accessible
            try:
                from src.trading.jupiter_api import jupiter_api
                # Test API connectivity with a simple price check
                test_price = jupiter_api.get_token_price("So11111111111111111111111111111111111111112")
                if test_price <= 0:
                    return False, "Jupiter API connectivity test failed. Cannot get SOL price."
            except Exception as e:
                return False, f"Jupiter API not accessible: {e}"
            
            # Check if risk management is configured
            from src.trading.risk_manager import risk_manager
            if not risk_manager.enabled:
                return False, "Risk management is disabled. Enable risk management before live trading."
            
            return True, "All prerequisites met for live trading."
            
        except Exception as e:
            logger.error(f"Error checking live trading prerequisites: {e}")
            return False, f"Error during safety check: {e}"
    
    def display_startup_warning(self, is_live_mode: bool) -> bool:
        """
        Display startup warning about trading mode.
        
        Args:
            is_live_mode: Whether the bot is starting in live trading mode
            
        Returns:
            True if user confirms to proceed, False otherwise
        """
        if not self.safety_checks_enabled:
            return True
        
        if is_live_mode:
            # Live trading mode warning
            warning_panel = Panel(
                "[bold red]⚠️  LIVE TRADING MODE ACTIVE ⚠️[/bold red]\n\n"
                "[yellow]The bot is configured to execute REAL trades with REAL money![/yellow]\n\n"
                "[bold]Current Configuration:[/bold]\n"
                "• Live Trading: [bold red]ENABLED[/bold red]\n"
                "• Paper Trading: [bold red]DISABLED[/bold red]\n\n"
                "[bold]Safety Features Active:[/bold]\n"
                f"• Daily Loss Limit: {get_config_value('max_daily_loss_limit', 1000.0)} SOL\n"
                f"• Emergency Stop: {'ENABLED' if get_config_value('emergency_stop_enabled', True) else 'DISABLED'}\n"
                f"• Risk Management: {'ENABLED' if get_config_value('risk_management_enabled', True) else 'DISABLED'}\n\n"
                "[bold red]Proceed only if you understand the risks![/bold red]",
                title="🚨 LIVE TRADING WARNING",
                border_style="red"
            )
            console.print(warning_panel)
            
            # Check prerequisites
            is_ready, message = self.check_live_trading_prerequisites()
            if not is_ready:
                console.print(f"\n[bold red]❌ Prerequisites not met:[/bold red] {message}")
                console.print("\n[yellow]Please fix the issues above before proceeding with live trading.[/yellow]")
                console.print("[green]You can switch to paper trading mode from the main menu (option 4).[/green]")
                return False
            
            console.print(f"\n[green]✅ Prerequisites check passed:[/green] {message}")
            
            if self.require_risk_acknowledgment:
                console.print("\n[bold]RISK ACKNOWLEDGMENT REQUIRED:[/bold]")
                console.print("• I understand that live trading involves real financial risk")
                console.print("• I acknowledge that automated trading may result in losses")
                console.print("• I have sufficient funds and risk tolerance for this activity")
                console.print("• I understand the bot's functionality and limitations")
                
                if not Confirm.ask("\n[bold red]Do you acknowledge these risks and want to proceed with live trading?[/bold red]"):
                    console.print("\n[yellow]Live trading cancelled. The bot will not start.[/yellow]")
                    console.print("[green]To use paper trading mode, set 'paper_trading_mode': true in config.py[/green]")
                    return False
            
            # Log the acknowledgment
            logger.critical(f"LIVE TRADING MODE ACKNOWLEDGED by user at {datetime.now()}")
            console.print("\n[bold green]✅ Live trading mode confirmed. Bot will execute real trades.[/bold green]")
            
        else:
            # Paper trading mode info
            info_panel = Panel(
                "[bold green]📊 PAPER TRADING MODE ACTIVE[/bold green]\n\n"
                "[green]The bot is in simulation mode - no real money will be used.[/green]\n\n"
                "[bold]Current Configuration:[/bold]\n"
                "• Live Trading: [bold green]DISABLED[/bold green]\n"
                "• Paper Trading: [bold green]ENABLED[/bold green]\n\n"
                "[bold]Features Available:[/bold]\n"
                "• Strategy testing and development\n"
                "• Risk-free learning and experimentation\n"
                "• Performance analysis and backtesting\n"
                "• All bot features except real trading\n\n"
                "[yellow]Switch to live trading from the main menu when ready.[/yellow]",
                title="📈 PAPER TRADING MODE",
                border_style="green"
            )
            console.print(info_panel)
        
        return True
    
    def display_mode_change_warning(self, from_mode: str, to_mode: str) -> bool:
        """
        Display warning when changing trading modes.
        
        Args:
            from_mode: Current trading mode ('live' or 'paper')
            to_mode: Target trading mode ('live' or 'paper')
            
        Returns:
            True if user confirms the change, False otherwise
        """
        if to_mode == 'live':
            console.print(f"\n[bold red]⚠️  SWITCHING TO LIVE TRADING ⚠️[/bold red]")
            console.print("[yellow]You are about to enable real money trading![/yellow]")
            
            # Check prerequisites again
            is_ready, message = self.check_live_trading_prerequisites()
            if not is_ready:
                console.print(f"\n[bold red]❌ Cannot switch to live trading:[/bold red] {message}")
                return False
            
            console.print(f"\n[green]✅ Prerequisites met:[/green] {message}")
            
            if not Confirm.ask("\n[bold red]Are you sure you want to enable live trading?[/bold red]"):
                return False
                
            # Log the mode change
            logger.critical(f"TRADING MODE CHANGED: {from_mode} -> {to_mode} at {datetime.now()}")
            
        else:  # Switching to paper trading
            console.print(f"\n[bold green]Switching to paper trading (simulation) mode.[/bold green]")
            logger.info(f"TRADING MODE CHANGED: {from_mode} -> {to_mode} at {datetime.now()}")
        
        return True
    
    def get_safety_status(self) -> Dict[str, Any]:
        """
        Get current safety status and configuration.
        
        Returns:
            Dictionary with safety status information
        """
        try:
            is_ready, message = self.check_live_trading_prerequisites()
            
            return {
                "safety_checks_enabled": self.safety_checks_enabled,
                "live_trading_ready": is_ready,
                "prerequisite_message": message,
                "risk_acknowledgment_required": self.require_risk_acknowledgment,
                "min_wallet_balance_sol": self.min_wallet_balance_sol,
                "emergency_stop_enabled": get_config_value("emergency_stop_enabled", True),
                "daily_loss_limit": get_config_value("max_daily_loss_limit", 1000.0),
                "risk_management_enabled": get_config_value("risk_management_enabled", True)
            }
        except Exception as e:
            logger.error(f"Error getting safety status: {e}")
            return {"error": str(e)}


# Global instance
trading_safety = TradingSafetyManager()
