"""
Auto-buy functionality for the Solana Memecoin Trading Bot.
Allows quick buying of tokens by pasting addresses or using a command.
"""

import logging
import re
from typing import Dict, Any, Optional, Union

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.trading.jupiter_api import jupiter_api
from src.trading.position_manager import position_manager
from src.notifications.notification_service import notification_service, NotificationPriority
from src.wallet.wallet import wallet_manager

# Get logger for this module
logger = get_logger(__name__)


class AutoBuyManager:
    """Manager for auto-buy functionality."""
    
    def __init__(self):
        """Initialize the auto-buy manager."""
        self.enabled = get_config_value("auto_buy_enabled", False)
        self.default_amount = float(get_config_value("auto_buy_default_amount", 0.1))
        self.require_confirmation = bool(get_config_value("auto_buy_require_confirmation", True))
        self.solana_address_pattern = re.compile(r'^[1-9A-HJ-NP-Za-km-z]{32,44}$')
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable auto-buy.
        
        Args:
            enabled: Whether auto-buy should be enabled
        """
        self.enabled = enabled
        update_config("auto_buy_enabled", enabled)
        logger.info(f"Auto-buy {'enabled' if enabled else 'disabled'}")
    
    def set_default_amount(self, amount: float) -> None:
        """
        Set the default amount for auto-buy.
        
        Args:
            amount: Default amount in SOL
        """
        if amount <= 0:
            raise ValueError("Amount must be greater than 0")
        
        self.default_amount = amount
        update_config("auto_buy_default_amount", amount)
        logger.info(f"Auto-buy default amount set to {amount} SOL")
    
    def set_require_confirmation(self, require: bool) -> None:
        """
        Set whether confirmation is required for auto-buy.
        
        Args:
            require: Whether confirmation is required
        """
        self.require_confirmation = require
        update_config("auto_buy_require_confirmation", require)
        logger.info(f"Auto-buy confirmation {'required' if require else 'not required'}")
    
    def is_solana_address(self, text: str) -> bool:
        """
        Check if a string is a valid Solana address.
        
        Args:
            text: String to check
            
        Returns:
            True if the string is a valid Solana address, False otherwise
        """
        return bool(self.solana_address_pattern.match(text))
    
    def handle_text_input(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Handle text input for auto-buy.
        
        Args:
            text: Text input
            
        Returns:
            Result of auto-buy if successful, None otherwise
        """
        if not self.enabled:
            logger.debug("Auto-buy is disabled")
            return None
        
        # Check if text is a Solana address
        if self.is_solana_address(text):
            logger.info(f"Detected Solana address: {text}")
            
            # If confirmation is required, return the address for confirmation
            if self.require_confirmation:
                return {
                    "address": text,
                    "requires_confirmation": True
                }
            
            # Otherwise, execute auto-buy
            return self.execute_auto_buy(text, self.default_amount)
        
        # Check if text is a quickbuy command
        if text.lower().startswith("quickbuy "):
            parts = text.split()
            if len(parts) < 2:
                logger.warning("Invalid quickbuy command: missing address")
                return None
            
            address = parts[1]
            
            # Check if address is valid
            if not self.is_solana_address(address):
                logger.warning(f"Invalid Solana address in quickbuy command: {address}")
                return None
            
            # Get amount if provided
            amount = self.default_amount
            if len(parts) >= 3:
                try:
                    amount = float(parts[2])
                except ValueError:
                    logger.warning(f"Invalid amount in quickbuy command: {parts[2]}")
                    return None
            
            # Execute auto-buy
            return self.execute_auto_buy(address, amount)
        
        return None
    
    def execute_auto_buy(self, token_mint: str, amount_sol: float) -> Dict[str, Any]:
        """
        Execute an auto-buy.
        
        Args:
            token_mint: Token mint address
            amount_sol: Amount in SOL
            
        Returns:
            Result of the buy operation
        """
        try:
            # Check if wallet is connected
            if not wallet_manager.current_keypair:
                logger.warning("Cannot execute auto-buy: wallet not connected")
                return {
                    "success": False,
                    "error": "Wallet not connected"
                }
            
            # Get token info
            token_info = jupiter_api.get_token_info(token_mint)
            token_symbol = token_info.get("symbol", token_mint[:8]) if token_info else token_mint[:8]
            
            # Execute buy
            result = position_manager.buy_token(
                token_mint=token_mint,
                amount_sol=amount_sol,
                token_symbol=token_symbol
            )
            
            if result["success"]:
                # Send notification
                notification_service.send_order_alert(
                    message=f"Auto-buy executed: {token_symbol} for {amount_sol} SOL",
                    priority=NotificationPriority.HIGH.value
                )
                
                logger.info(f"Auto-buy executed: {token_symbol} for {amount_sol} SOL")
            else:
                # Send notification
                notification_service.send_order_alert(
                    message=f"Auto-buy failed: {token_symbol} - {result.get('error', 'Unknown error')}",
                    priority=NotificationPriority.HIGH.value
                )
                
                logger.error(f"Auto-buy failed: {token_symbol} - {result.get('error', 'Unknown error')}")
            
            return result
        except Exception as e:
            logger.error(f"Error executing auto-buy: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Create singleton instance
auto_buy_manager = AutoBuyManager()
