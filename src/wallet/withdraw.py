"""
Withdraw functionality for the Solana Memecoin Trading Bot.
Handles withdrawals of SOL and SPL tokens to external wallets.
"""

import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from solders.message import Message
from solders.instruction import Instruction
from solders.system_program import transfer, TransferParams
from solana.rpc.types import TxOpts
from spl.token.instructions import get_associated_token_address, transfer as spl_transfer
from spl.token.constants import TOKEN_PROGRAM_ID

from config import get_config_value
from src.utils.logging_utils import get_logger
from src.wallet.wallet import wallet_manager
from src.wallet.hardware_wallet import hardware_wallet_manager
from src.solana.solana_interact import solana_client
from src.notifications.notification_service import notification_service, NotificationPriority

# Get logger for this module
logger = get_logger(__name__)


class WithdrawManager:
    """Manager for withdrawal operations."""

    def __init__(self):
        """Initialize the withdraw manager."""
        # Security settings
        self.max_withdrawal_amount_sol = float(get_config_value("max_withdrawal_amount_sol", 10.0))
        self.require_confirmation = bool(get_config_value("require_withdrawal_confirmation", True))
        self.withdrawal_history: list[Dict[str, Any]] = []
        
        # Load withdrawal history if available
        self._load_history()
    
    def _load_history(self) -> None:
        """Load withdrawal history from config."""
        self.withdrawal_history = get_config_value("withdrawal_history", [])
    
    def _save_history(self) -> None:
        """Save withdrawal history to config."""
        from config import update_config
        update_config("withdrawal_history", self.withdrawal_history)
    
    def withdraw_sol(self, amount: float, destination: str, keypair: Optional[Keypair] = None) -> Dict[str, Any]:
        """
        Withdraw SOL to an external wallet.
        
        Args:
            amount: Amount of SOL to withdraw
            destination: Destination wallet address
            keypair: Optional keypair to use for signing (if None, uses current wallet)
            
        Returns:
            Dictionary with transaction details
        """
        try:
            # Validate amount
            if amount <= 0:
                return {"success": False, "error": "Amount must be greater than 0"}
            
            if amount > self.max_withdrawal_amount_sol:
                return {"success": False, "error": f"Amount exceeds maximum withdrawal limit of {self.max_withdrawal_amount_sol} SOL"}
            
            # Validate destination
            try:
                destination_pubkey = Pubkey.from_string(destination)
            except Exception as e:
                return {"success": False, "error": f"Invalid destination address: {e}"}
            
            # Get keypair
            if keypair is None:
                if wallet_manager.current_keypair is None:
                    return {"success": False, "error": "No wallet connected"}
                keypair = wallet_manager.current_keypair
            
            # Check balance
            balance = solana_client.get_sol_balance(str(keypair.pubkey()))
            if balance < amount:
                return {"success": False, "error": f"Insufficient balance: {balance} SOL"}
            
            # Calculate lamports
            lamports = int(amount * 1_000_000_000)  # Convert SOL to lamports
            
            # Create transfer instruction
            transfer_params = TransferParams(
                from_pubkey=keypair.pubkey(),
                to_pubkey=destination_pubkey,
                lamports=lamports
            )
            transfer_ix = transfer(transfer_params)
            
            # Create transaction
            recent_blockhash = solana_client.get_recent_blockhash()
            
            # Get priority fee if enabled
            priority_fee = None
            if get_config_value("fee_optimization_enabled", True):
                priority_fee = self._get_priority_fee("withdraw")
            
            # Create transaction
            tx = Transaction()
            tx.add(transfer_ix)
            
            # Set recent blockhash and fee payer
            tx.recent_blockhash = recent_blockhash
            tx.fee_payer = keypair.pubkey()
            
            # Sign transaction
            tx = Transaction.sign_with_keypair(tx, keypair)
            
            # Send transaction
            opts = TxOpts(skip_preflight=False, preflight_commitment="confirmed")
            if priority_fee is not None:
                opts.compute_unit_price = priority_fee
            
            signature = solana_client.send_transaction(tx, opts=opts)
            
            # Record withdrawal
            withdrawal_record = {
                "type": "sol",
                "amount": amount,
                "destination": destination,
                "timestamp": datetime.now().isoformat(),
                "signature": signature,
                "status": "success"
            }
            self.withdrawal_history.append(withdrawal_record)
            self._save_history()
            
            # Send notification
            notification_service.send_wallet_alert(
                message=f"Withdrew {amount} SOL to {destination[:8]}...{destination[-8:]}",
                priority=NotificationPriority.HIGH.value
            )
            
            logger.info(f"Withdrew {amount} SOL to {destination}")
            return {
                "success": True,
                "signature": signature,
                "amount": amount,
                "destination": destination,
                "fee": "~0.000005 SOL"
            }
        except Exception as e:
            logger.error(f"Error withdrawing SOL: {e}")
            
            # Record failed withdrawal
            withdrawal_record = {
                "type": "sol",
                "amount": amount,
                "destination": destination,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "failed"
            }
            self.withdrawal_history.append(withdrawal_record)
            self._save_history()
            
            return {"success": False, "error": str(e)}
    
    def withdraw_token(self, token_mint: str, amount: float, destination: str, 
                      keypair: Optional[Keypair] = None) -> Dict[str, Any]:
        """
        Withdraw SPL token to an external wallet.
        
        Args:
            token_mint: Token mint address
            amount: Amount of tokens to withdraw
            destination: Destination wallet address
            keypair: Optional keypair to use for signing (if None, uses current wallet)
            
        Returns:
            Dictionary with transaction details
        """
        try:
            # Validate amount
            if amount <= 0:
                return {"success": False, "error": "Amount must be greater than 0"}
            
            # Validate token mint
            try:
                token_mint_pubkey = Pubkey.from_string(token_mint)
            except Exception as e:
                return {"success": False, "error": f"Invalid token mint address: {e}"}
            
            # Validate destination
            try:
                destination_pubkey = Pubkey.from_string(destination)
            except Exception as e:
                return {"success": False, "error": f"Invalid destination address: {e}"}
            
            # Get keypair
            if keypair is None:
                if wallet_manager.current_keypair is None:
                    return {"success": False, "error": "No wallet connected"}
                keypair = wallet_manager.current_keypair
            
            # Get token info
            token_info = solana_client.get_token_info(token_mint)
            if not token_info:
                return {"success": False, "error": f"Could not get token info for {token_mint}"}
            
            # Get token decimals
            decimals = token_info.get("decimals", 9)
            
            # Calculate token amount
            token_amount = int(amount * (10 ** decimals))
            
            # Get source token account
            source_token_account = get_associated_token_address(
                owner=keypair.pubkey(),
                mint=token_mint_pubkey
            )
            
            # Check if source token account exists
            source_account_info = solana_client.get_token_account_info(str(source_token_account))
            if not source_account_info:
                return {"success": False, "error": f"Token account not found for {token_mint}"}
            
            # Check balance
            token_balance = int(source_account_info.get("amount", 0))
            if token_balance < token_amount:
                return {"success": False, "error": f"Insufficient token balance: {token_balance / (10 ** decimals)}"}
            
            # Get destination token account
            destination_token_account = get_associated_token_address(
                owner=destination_pubkey,
                mint=token_mint_pubkey
            )
            
            # Check if destination token account exists
            destination_account_info = solana_client.get_token_account_info(str(destination_token_account))
            
            # Create instructions
            instructions = []
            
            # If destination token account doesn't exist, create it
            if not destination_account_info:
                from spl.token.instructions import create_associated_token_account
                create_ata_ix = create_associated_token_account(
                    payer=keypair.pubkey(),
                    owner=destination_pubkey,
                    mint=token_mint_pubkey
                )
                instructions.append(create_ata_ix)
            
            # Create transfer instruction
            transfer_ix = spl_transfer(
                token_program_id=TOKEN_PROGRAM_ID,
                source=source_token_account,
                dest=destination_token_account,
                owner=keypair.pubkey(),
                amount=token_amount
            )
            instructions.append(transfer_ix)
            
            # Create transaction
            recent_blockhash = solana_client.get_recent_blockhash()
            
            # Get priority fee if enabled
            priority_fee = None
            if get_config_value("fee_optimization_enabled", True):
                priority_fee = self._get_priority_fee("withdraw")
            
            # Create transaction
            tx = Transaction()
            for ix in instructions:
                tx.add(ix)
            
            # Set recent blockhash and fee payer
            tx.recent_blockhash = recent_blockhash
            tx.fee_payer = keypair.pubkey()
            
            # Sign transaction
            tx = Transaction.sign_with_keypair(tx, keypair)
            
            # Send transaction
            opts = TxOpts(skip_preflight=False, preflight_commitment="confirmed")
            if priority_fee is not None:
                opts.compute_unit_price = priority_fee
            
            signature = solana_client.send_transaction(tx, opts=opts)
            
            # Get token symbol
            token_symbol = token_info.get("symbol", token_mint[:8])
            
            # Record withdrawal
            withdrawal_record = {
                "type": "token",
                "token_mint": token_mint,
                "token_symbol": token_symbol,
                "amount": amount,
                "destination": destination,
                "timestamp": datetime.now().isoformat(),
                "signature": signature,
                "status": "success"
            }
            self.withdrawal_history.append(withdrawal_record)
            self._save_history()
            
            # Send notification
            notification_service.send_wallet_alert(
                message=f"Withdrew {amount} {token_symbol} to {destination[:8]}...{destination[-8:]}",
                priority=NotificationPriority.HIGH.value
            )
            
            logger.info(f"Withdrew {amount} {token_symbol} to {destination}")
            return {
                "success": True,
                "signature": signature,
                "amount": amount,
                "token": token_symbol,
                "destination": destination,
                "fee": "~0.000005 SOL"
            }
        except Exception as e:
            logger.error(f"Error withdrawing token: {e}")
            
            # Record failed withdrawal
            withdrawal_record = {
                "type": "token",
                "token_mint": token_mint,
                "amount": amount,
                "destination": destination,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "failed"
            }
            self.withdrawal_history.append(withdrawal_record)
            self._save_history()
            
            return {"success": False, "error": str(e)}
    
    def _get_priority_fee(self, operation_type: str) -> Optional[int]:
        """
        Get priority fee for a specific operation type.
        
        Args:
            operation_type: Type of operation (withdraw, buy, sell, etc.)
            
        Returns:
            Priority fee in micro-lamports, or None if not enabled
        """
        # Check if fee optimization is enabled
        if not get_config_value("fee_optimization_enabled", True):
            return None
        
        # Get fee multiplier for this operation type
        fee_multipliers = get_config_value("priority_fee_multipliers", {})
        multiplier = fee_multipliers.get(operation_type, fee_multipliers.get("default", 1.0))
        
        # Get base priority fee
        priority_fee_percentile = get_config_value("priority_fee_percentile", 75)
        min_priority_fee = get_config_value("min_priority_fee", 1000)
        
        # Get recent priority fee from Solana
        recent_priority_fee = solana_client.get_recent_priority_fee()
        
        # Calculate priority fee
        if recent_priority_fee is not None:
            # Get fee at specified percentile
            fee = recent_priority_fee.get(str(priority_fee_percentile), min_priority_fee)
            
            # Apply multiplier
            fee = int(fee * multiplier)
            
            # Ensure minimum fee
            fee = max(fee, min_priority_fee)
            
            return fee
        
        return min_priority_fee
    
    def get_withdrawal_history(self) -> list[Dict[str, Any]]:
        """
        Get withdrawal history.
        
        Returns:
            List of withdrawal records
        """
        return self.withdrawal_history
    
    def clear_withdrawal_history(self) -> None:
        """Clear withdrawal history."""
        self.withdrawal_history = []
        self._save_history()
        logger.info("Cleared withdrawal history")


# Create singleton instance
withdraw_manager = WithdrawManager()
