"""
External wallet monitoring module for the Solana Memecoin Trading Bot.
Tracks transactions and activities of specified external wallets.
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Set, Union
from datetime import datetime, timedelta
from pathlib import Path

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.trading.helius_api import helius_api
from src.notifications.notification_service import notification_service, NotificationPriority

# Get logger for this module
logger = get_logger(__name__)


class WalletMonitor:
    """Monitor for external wallets."""
    
    def __init__(self):
        """Initialize the wallet monitor."""
        self.enabled = get_config_value("wallet_monitoring_enabled", False)
        self.monitored_wallets: Dict[str, Dict[str, Any]] = {}
        self.wallet_transactions: Dict[str, List[Dict[str, Any]]] = {}
        self.max_transactions_per_wallet = int(get_config_value("max_transactions_per_wallet", "100"))
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.monitoring_interval = int(get_config_value("wallet_monitoring_interval_seconds", "300"))  # Default: 5 minutes
        
        # Load monitored wallets
        self._load_monitored_wallets()
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable wallet monitoring.
        
        Args:
            enabled: Whether wallet monitoring should be enabled
        """
        self.enabled = enabled
        update_config("wallet_monitoring_enabled", enabled)
        logger.info(f"Wallet monitoring {'enabled' if enabled else 'disabled'}")
        
        if enabled and not self.monitoring_thread:
            self.start_monitoring_thread()
        elif not enabled and self.monitoring_thread:
            self.stop_monitoring_thread()
    
    def _load_monitored_wallets(self) -> None:
        """Load monitored wallets from config."""
        self.monitored_wallets = get_config_value("monitored_wallets", {})
        self.wallet_transactions = get_config_value("wallet_transactions", {})
        logger.info(f"Loaded {len(self.monitored_wallets)} monitored wallets")
    
    def _save_monitored_wallets(self) -> None:
        """Save monitored wallets to config."""
        update_config("monitored_wallets", self.monitored_wallets)
        update_config("wallet_transactions", self.wallet_transactions)
    
    def start_monitoring_thread(self) -> None:
        """Start the monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Wallet monitoring thread already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Wallet monitoring thread started")
    
    def stop_monitoring_thread(self) -> None:
        """Stop the monitoring thread."""
        if not self.monitoring_thread:
            logger.warning("Wallet monitoring thread not running")
            return
        
        self.stop_monitoring.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.monitoring_thread = None
        logger.info("Wallet monitoring thread stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Check all monitored wallets
                self._check_wallets()
            except Exception as e:
                logger.error(f"Error in wallet monitoring loop: {e}")
            
            # Sleep for the monitoring interval
            self.stop_monitoring.wait(self.monitoring_interval)
    
    def _check_wallets(self) -> None:
        """Check all monitored wallets for new transactions."""
        if not self.monitored_wallets:
            return
        
        for wallet_address, wallet_info in self.monitored_wallets.items():
            try:
                # Skip if wallet is not active
                if not wallet_info.get("active", True):
                    continue
                
                # Get last checked time
                last_checked = wallet_info.get("last_checked")
                if last_checked:
                    last_checked = datetime.fromisoformat(last_checked)
                else:
                    # Default to 24 hours ago
                    last_checked = datetime.now() - timedelta(days=1)
                
                # Get transactions since last check
                transactions = self._get_wallet_transactions(wallet_address, last_checked)
                
                if transactions:
                    # Process new transactions
                    self._process_transactions(wallet_address, wallet_info, transactions)
                
                # Update last checked time
                wallet_info["last_checked"] = datetime.now().isoformat()
                self._save_monitored_wallets()
            except Exception as e:
                logger.error(f"Error checking wallet {wallet_address}: {e}")
    
    def _get_wallet_transactions(self, wallet_address: str, since: datetime) -> List[Dict[str, Any]]:
        """
        Get transactions for a wallet since a specific time.
        
        Args:
            wallet_address: Wallet address
            since: Datetime to get transactions since
            
        Returns:
            List of transactions
        """
        try:
            # Convert datetime to Unix timestamp (milliseconds)
            since_timestamp = int(since.timestamp() * 1000)
            
            # Get transactions from Helius API
            transactions = helius_api.get_wallet_transactions(wallet_address, since_timestamp)
            
            return transactions
        except Exception as e:
            logger.error(f"Error getting transactions for {wallet_address}: {e}")
            return []
    
    def _process_transactions(self, wallet_address: str, wallet_info: Dict[str, Any], 
                             transactions: List[Dict[str, Any]]) -> None:
        """
        Process new transactions for a wallet.
        
        Args:
            wallet_address: Wallet address
            wallet_info: Wallet information
            transactions: List of new transactions
        """
        # Initialize wallet transactions if needed
        if wallet_address not in self.wallet_transactions:
            self.wallet_transactions[wallet_address] = []
        
        # Get wallet label
        wallet_label = wallet_info.get("label", wallet_address[:8] + "..." + wallet_address[-8:])
        
        # Process each transaction
        for tx in transactions:
            # Extract transaction details
            tx_hash = tx.get("signature", "")
            timestamp = tx.get("timestamp", datetime.now().timestamp() * 1000)
            
            # Convert timestamp to datetime
            tx_time = datetime.fromtimestamp(timestamp / 1000)
            
            # Extract transaction type and details
            tx_type, tx_details = self._analyze_transaction(tx)
            
            # Skip if transaction type is unknown
            if tx_type == "unknown":
                continue
            
            # Create transaction record
            tx_record = {
                "hash": tx_hash,
                "type": tx_type,
                "details": tx_details,
                "timestamp": tx_time.isoformat(),
                "raw_data": tx
            }
            
            # Add to wallet transactions
            self.wallet_transactions[wallet_address].insert(0, tx_record)
            
            # Limit number of transactions
            if len(self.wallet_transactions[wallet_address]) > self.max_transactions_per_wallet:
                self.wallet_transactions[wallet_address] = self.wallet_transactions[wallet_address][:self.max_transactions_per_wallet]
            
            # Send notification if enabled
            if wallet_info.get("notifications_enabled", True):
                self._send_transaction_notification(wallet_label, tx_record)
        
        # Save wallet transactions
        self._save_monitored_wallets()
        
        logger.info(f"Processed {len(transactions)} new transactions for {wallet_label}")
    
    def _analyze_transaction(self, transaction: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Analyze a transaction to determine its type and extract details.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Tuple of (transaction_type, transaction_details)
        """
        # This is a simplified implementation
        # In a real implementation, we would analyze the transaction more thoroughly
        
        # Default values
        tx_type = "unknown"
        tx_details = {}
        
        # Check if transaction has parsed data
        parsed_data = transaction.get("parsedTransactionData", {})
        if not parsed_data:
            return tx_type, tx_details
        
        # Check if transaction has token transfers
        token_transfers = parsed_data.get("tokenTransfers", [])
        if token_transfers:
            # This is a token transfer transaction
            tx_type = "token_transfer"
            
            # Extract token transfer details
            transfers = []
            for transfer in token_transfers:
                token_mint = transfer.get("mint", "")
                from_address = transfer.get("fromUserAccount", "")
                to_address = transfer.get("toUserAccount", "")
                amount = transfer.get("tokenAmount", 0)
                decimals = transfer.get("tokenStandard", {}).get("decimals", 0)
                symbol = transfer.get("tokenStandard", {}).get("symbol", "")
                
                # Calculate actual amount
                actual_amount = amount / (10 ** decimals) if decimals > 0 else amount
                
                transfers.append({
                    "token_mint": token_mint,
                    "from": from_address,
                    "to": to_address,
                    "amount": actual_amount,
                    "symbol": symbol
                })
            
            tx_details["transfers"] = transfers
        
        # Check if transaction has SOL transfers
        sol_transfers = parsed_data.get("nativeTransfers", [])
        if sol_transfers:
            # This is a SOL transfer transaction
            tx_type = "sol_transfer" if tx_type == "unknown" else tx_type
            
            # Extract SOL transfer details
            transfers = []
            for transfer in sol_transfers:
                from_address = transfer.get("fromUserAccount", "")
                to_address = transfer.get("toUserAccount", "")
                amount = transfer.get("amount", 0)
                
                # Calculate actual amount in SOL
                actual_amount = amount / 1_000_000_000  # Convert lamports to SOL
                
                transfers.append({
                    "from": from_address,
                    "to": to_address,
                    "amount": actual_amount
                })
            
            tx_details["sol_transfers"] = transfers
        
        # Check if transaction has swap data
        swap_data = parsed_data.get("swap", {})
        if swap_data:
            # This is a swap transaction
            tx_type = "swap"
            
            # Extract swap details
            from_mint = swap_data.get("fromMint", "")
            to_mint = swap_data.get("toMint", "")
            from_amount = swap_data.get("fromAmount", 0)
            to_amount = swap_data.get("toAmount", 0)
            from_decimals = swap_data.get("fromDecimals", 0)
            to_decimals = swap_data.get("toDecimals", 0)
            
            # Calculate actual amounts
            actual_from_amount = from_amount / (10 ** from_decimals) if from_decimals > 0 else from_amount
            actual_to_amount = to_amount / (10 ** to_decimals) if to_decimals > 0 else to_amount
            
            tx_details["swap"] = {
                "from_mint": from_mint,
                "to_mint": to_mint,
                "from_amount": actual_from_amount,
                "to_amount": actual_to_amount,
                "from_symbol": swap_data.get("fromSymbol", ""),
                "to_symbol": swap_data.get("toSymbol", "")
            }
        
        return tx_type, tx_details
    
    def _send_transaction_notification(self, wallet_label: str, transaction: Dict[str, Any]) -> None:
        """
        Send notification for a new transaction.
        
        Args:
            wallet_label: Wallet label
            transaction: Transaction data
        """
        tx_type = transaction["type"]
        tx_details = transaction["details"]
        
        # Format message based on transaction type
        if tx_type == "token_transfer":
            transfers = tx_details.get("transfers", [])
            if transfers:
                transfer = transfers[0]  # Just use the first transfer for simplicity
                message = f"{wallet_label} transferred {transfer['amount']} {transfer['symbol']}"
        elif tx_type == "sol_transfer":
            transfers = tx_details.get("sol_transfers", [])
            if transfers:
                transfer = transfers[0]  # Just use the first transfer for simplicity
                message = f"{wallet_label} transferred {transfer['amount']} SOL"
        elif tx_type == "swap":
            swap = tx_details.get("swap", {})
            message = f"{wallet_label} swapped {swap.get('from_amount')} {swap.get('from_symbol')} for {swap.get('to_amount')} {swap.get('to_symbol')}"
        else:
            message = f"{wallet_label} made a transaction of type {tx_type}"
        
        # Send notification
        notification_service.send_wallet_alert(
            message=message,
            priority=NotificationPriority.NORMAL.value
        )
    
    def add_monitored_wallet(self, wallet_address: str, label: Optional[str] = None, 
                            notifications_enabled: bool = True) -> Dict[str, Any]:
        """
        Add a wallet to monitor.
        
        Args:
            wallet_address: Wallet address
            label: Optional label for the wallet
            notifications_enabled: Whether to enable notifications for this wallet
            
        Returns:
            The added wallet information
        """
        # Check if wallet is already monitored
        if wallet_address in self.monitored_wallets:
            logger.warning(f"Wallet {wallet_address} is already monitored")
            return self.monitored_wallets[wallet_address]
        
        # Create wallet info
        wallet_info = {
            "address": wallet_address,
            "label": label or wallet_address[:8] + "..." + wallet_address[-8:],
            "added_at": datetime.now().isoformat(),
            "last_checked": None,
            "active": True,
            "notifications_enabled": notifications_enabled
        }
        
        # Add to monitored wallets
        self.monitored_wallets[wallet_address] = wallet_info
        
        # Initialize wallet transactions
        self.wallet_transactions[wallet_address] = []
        
        # Save monitored wallets
        self._save_monitored_wallets()
        
        logger.info(f"Added monitored wallet: {wallet_info['label']} ({wallet_address})")
        return wallet_info
    
    def remove_monitored_wallet(self, wallet_address: str) -> bool:
        """
        Remove a monitored wallet.
        
        Args:
            wallet_address: Wallet address
            
        Returns:
            True if wallet was removed, False otherwise
        """
        if wallet_address not in self.monitored_wallets:
            logger.warning(f"Wallet {wallet_address} is not monitored")
            return False
        
        # Remove from monitored wallets
        del self.monitored_wallets[wallet_address]
        
        # Remove wallet transactions
        if wallet_address in self.wallet_transactions:
            del self.wallet_transactions[wallet_address]
        
        # Save monitored wallets
        self._save_monitored_wallets()
        
        logger.info(f"Removed monitored wallet: {wallet_address}")
        return True
    
    def update_monitored_wallet(self, wallet_address: str, label: Optional[str] = None, 
                               active: Optional[bool] = None, 
                               notifications_enabled: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """
        Update a monitored wallet.
        
        Args:
            wallet_address: Wallet address
            label: New label (optional)
            active: New active state (optional)
            notifications_enabled: New notifications state (optional)
            
        Returns:
            The updated wallet information, or None if wallet not found
        """
        if wallet_address not in self.monitored_wallets:
            logger.warning(f"Wallet {wallet_address} is not monitored")
            return None
        
        # Get wallet info
        wallet_info = self.monitored_wallets[wallet_address]
        
        # Update wallet info
        if label is not None:
            wallet_info["label"] = label
        
        if active is not None:
            wallet_info["active"] = active
        
        if notifications_enabled is not None:
            wallet_info["notifications_enabled"] = notifications_enabled
        
        # Save monitored wallets
        self._save_monitored_wallets()
        
        logger.info(f"Updated monitored wallet: {wallet_info['label']} ({wallet_address})")
        return wallet_info
    
    def get_monitored_wallets(self, active_only: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get all monitored wallets.
        
        Args:
            active_only: Whether to return only active wallets
            
        Returns:
            Dictionary of monitored wallets
        """
        if active_only:
            return {addr: info for addr, info in self.monitored_wallets.items() if info.get("active", True)}
        return self.monitored_wallets
    
    def get_wallet_transactions(self, wallet_address: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get transactions for a specific wallet.
        
        Args:
            wallet_address: Wallet address
            limit: Maximum number of transactions to return
            
        Returns:
            List of transactions
        """
        if wallet_address not in self.wallet_transactions:
            return []
        
        return self.wallet_transactions[wallet_address][:limit]
    
    def clear_wallet_transactions(self, wallet_address: str) -> bool:
        """
        Clear transactions for a specific wallet.
        
        Args:
            wallet_address: Wallet address
            
        Returns:
            True if transactions were cleared, False otherwise
        """
        if wallet_address not in self.wallet_transactions:
            logger.warning(f"Wallet {wallet_address} has no transactions")
            return False
        
        # Clear transactions
        self.wallet_transactions[wallet_address] = []
        
        # Save wallet transactions
        self._save_monitored_wallets()
        
        logger.info(f"Cleared transactions for wallet: {wallet_address}")
        return True


# Create singleton instance
wallet_monitor = WalletMonitor()
