"""
Copy trading module for the Solana Memecoin Trading Bot.
Handles copying trades from tracked wallets using Helius API.
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime

from src.trading.helius_api import helius_api
from src.trading.smart_wallet_discovery import smart_wallet_discovery
from src.trading.jupiter_api import jupiter_api
from src.wallet.wallet import wallet_manager
from src.trading.position_manager import position_manager
from config import get_config_value, update_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CopyTrading:
    """Enhanced manager for copy trading functionality."""

    def __init__(self):
        """Initialize the copy trading manager."""
        self.enabled = get_config_value("copy_trading_enabled", False)
        self.min_transaction_sol = float(get_config_value("copy_min_transaction_sol", "0.1"))
        self.max_transaction_sol = float(get_config_value("copy_max_transaction_sol", "1.0"))
        self.copy_percentage = float(get_config_value("copy_percentage", "50.0"))  # Copy 50% of the trade size
        self.blacklisted_tokens = set(get_config_value("copy_blacklisted_tokens", []))

        # Enhanced copy trading settings
        self.tracked_wallets = get_config_value("copy_tracked_wallets", {})
        self.wallet_performance = get_config_value("copy_wallet_performance", {})
        self.copy_delay_seconds = int(get_config_value("copy_delay_seconds", "5"))
        self.max_position_size_sol = float(get_config_value("copy_max_position_size_sol", "5.0"))
        self.stop_loss_percentage = float(get_config_value("copy_stop_loss_percentage", "10.0"))
        self.take_profit_percentage = float(get_config_value("copy_take_profit_percentage", "50.0"))

        # Filtering criteria
        self.min_wallet_success_rate = float(get_config_value("copy_min_success_rate", "60.0"))
        self.min_wallet_profit_ratio = float(get_config_value("copy_min_profit_ratio", "1.5"))
        self.max_wallet_drawdown = float(get_config_value("copy_max_drawdown", "30.0"))

        # Register callback with Helius API
        helius_api.register_transaction_callback(self.handle_transaction)

        # Known token mints
        self.sol_mint = "So11111111111111111111111111111111111111112"
        self.usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

        # Recently processed transactions to avoid duplicates
        self.processed_transactions = set()
        self.max_processed_transactions = 1000

        # Copy trading statistics
        self.copy_stats = {
            "total_copies": 0,
            "successful_copies": 0,
            "failed_copies": 0,
            "total_profit_loss": 0.0,
            "best_copy_profit": 0.0,
            "worst_copy_loss": 0.0
        }

        # Enhanced copy trading features
        self.correlation_matrix = {}
        self.max_correlation = get_config_value("copy_max_correlation", 0.7)
        self.portfolio_copy_limit = get_config_value("copy_portfolio_limit", 0.3)  # Max 30% of portfolio
        self.smart_discovery_enabled = get_config_value("smart_copy_discovery_enabled", False)

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable copy trading.

        Args:
            enabled: Whether copy trading should be enabled
        """
        self.enabled = enabled
        update_config("copy_trading_enabled", enabled)
        logger.info(f"Copy trading {'enabled' if enabled else 'disabled'}")

        # Start or stop Helius WebSocket
        if enabled:
            helius_api.start_websocket()
        else:
            helius_api.stop_websocket()

    def set_copy_parameters(self, min_sol: float, max_sol: float, percentage: float) -> None:
        """
        Set copy trading parameters.

        Args:
            min_sol: Minimum transaction size in SOL
            max_sol: Maximum transaction size in SOL
            percentage: Percentage of the trade size to copy
        """
        self.min_transaction_sol = min_sol
        self.max_transaction_sol = max_sol
        self.copy_percentage = percentage

        update_config("copy_min_transaction_sol", min_sol)
        update_config("copy_max_transaction_sol", max_sol)
        update_config("copy_percentage", percentage)

        logger.info(f"Copy parameters set: min={min_sol} SOL, max={max_sol} SOL, percentage={percentage}%")

    def add_tracked_wallet(self, wallet_address: str, multiplier: float = 1.0,
                          max_copy_amount: float = None, enabled: bool = True) -> None:
        """
        Add a wallet to track for copy trading.

        Args:
            wallet_address: The wallet address to track
            multiplier: Trade size multiplier (default 1.0)
            max_copy_amount: Maximum amount to copy per trade
            enabled: Whether tracking is enabled for this wallet
        """
        self.tracked_wallets[wallet_address] = {
            "multiplier": multiplier,
            "max_copy_amount": max_copy_amount or self.max_transaction_sol,
            "enabled": enabled,
            "added_at": datetime.now().isoformat(),
            "total_trades": 0,
            "successful_trades": 0,
            "total_profit_loss": 0.0
        }

        # Initialize performance tracking
        if wallet_address not in self.wallet_performance:
            self.wallet_performance[wallet_address] = {
                "trades": [],
                "success_rate": 0.0,
                "profit_ratio": 0.0,
                "max_drawdown": 0.0,
                "last_updated": datetime.now().isoformat()
            }

        # Update config
        update_config("copy_tracked_wallets", self.tracked_wallets)
        update_config("copy_wallet_performance", self.wallet_performance)

        logger.info(f"Added wallet {wallet_address} to copy trading with multiplier {multiplier}")

    def remove_tracked_wallet(self, wallet_address: str) -> bool:
        """
        Remove a wallet from tracking.

        Args:
            wallet_address: The wallet address to remove

        Returns:
            True if wallet was removed, False if not found
        """
        if wallet_address in self.tracked_wallets:
            del self.tracked_wallets[wallet_address]
            update_config("copy_tracked_wallets", self.tracked_wallets)
            logger.info(f"Removed wallet {wallet_address} from copy trading")
            return True
        return False

    def update_wallet_performance(self, wallet_address: str, trade_result: Dict[str, Any]) -> None:
        """
        Update performance metrics for a tracked wallet.

        Args:
            wallet_address: The wallet address
            trade_result: Trade result data
        """
        if wallet_address not in self.wallet_performance:
            self.wallet_performance[wallet_address] = {
                "trades": [],
                "success_rate": 0.0,
                "profit_ratio": 0.0,
                "max_drawdown": 0.0,
                "last_updated": datetime.now().isoformat()
            }

        perf = self.wallet_performance[wallet_address]
        perf["trades"].append(trade_result)

        # Keep only last 100 trades
        if len(perf["trades"]) > 100:
            perf["trades"] = perf["trades"][-100:]

        # Calculate metrics
        trades = perf["trades"]
        if trades:
            successful = [t for t in trades if t.get("profit_loss", 0) > 0]
            perf["success_rate"] = len(successful) / len(trades) * 100

            total_profit = sum(t.get("profit_loss", 0) for t in successful)
            total_loss = abs(sum(t.get("profit_loss", 0) for t in trades if t.get("profit_loss", 0) < 0))
            perf["profit_ratio"] = total_profit / total_loss if total_loss > 0 else float('inf')

            # Calculate max drawdown
            running_total = 0
            peak = 0
            max_drawdown = 0
            for trade in trades:
                running_total += trade.get("profit_loss", 0)
                if running_total > peak:
                    peak = running_total
                drawdown = (peak - running_total) / peak * 100 if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            perf["max_drawdown"] = max_drawdown

        perf["last_updated"] = datetime.now().isoformat()
        update_config("copy_wallet_performance", self.wallet_performance)

    def should_copy_wallet(self, wallet_address: str) -> bool:
        """
        Check if a wallet meets the criteria for copying.

        Args:
            wallet_address: The wallet address to check

        Returns:
            True if wallet should be copied, False otherwise
        """
        if wallet_address not in self.tracked_wallets:
            return False

        wallet_config = self.tracked_wallets[wallet_address]
        if not wallet_config.get("enabled", True):
            return False

        if wallet_address not in self.wallet_performance:
            return True  # New wallet, give it a chance

        perf = self.wallet_performance[wallet_address]

        # Check minimum success rate
        if perf["success_rate"] < self.min_wallet_success_rate:
            return False

        # Check minimum profit ratio
        if perf["profit_ratio"] < self.min_wallet_profit_ratio:
            return False

        # Check maximum drawdown
        if perf["max_drawdown"] > self.max_wallet_drawdown:
            return False

        return True

    def add_blacklisted_token(self, token_mint: str) -> None:
        """
        Add a token to the blacklist.

        Args:
            token_mint: The token mint address
        """
        self.blacklisted_tokens.add(token_mint)
        update_config("copy_blacklisted_tokens", list(self.blacklisted_tokens))
        logger.info(f"Added token {token_mint} to copy trading blacklist")

    def remove_blacklisted_token(self, token_mint: str) -> bool:
        """
        Remove a token from the blacklist.

        Args:
            token_mint: The token mint address

        Returns:
            True if the token was removed, False if it wasn't blacklisted
        """
        if token_mint in self.blacklisted_tokens:
            self.blacklisted_tokens.remove(token_mint)
            update_config("copy_blacklisted_tokens", list(self.blacklisted_tokens))
            logger.info(f"Removed token {token_mint} from copy trading blacklist")
            return True
        return False

    def get_blacklisted_tokens(self) -> List[str]:
        """
        Get all blacklisted tokens.

        Returns:
            List of blacklisted token mint addresses
        """
        return list(self.blacklisted_tokens)

    def handle_transaction(self, transaction_data: Dict[str, Any]) -> None:
        """
        Handle a transaction event from Helius API.

        Args:
            transaction_data: The transaction data
        """
        if not self.enabled:
            return

        try:
            # Extract transaction signature
            signature = transaction_data.get("signature")
            if not signature or signature in self.processed_transactions:
                return

            # Add to processed transactions
            self.processed_transactions.add(signature)
            if len(self.processed_transactions) > self.max_processed_transactions:
                # Remove oldest entries
                self.processed_transactions = set(list(self.processed_transactions)[-self.max_processed_transactions:])

            # Check if it's a swap transaction
            if not self._is_swap_transaction(transaction_data):
                return

            # Extract swap details
            swap_details = self._extract_swap_details(transaction_data)
            if not swap_details:
                return

            # Check if it meets our criteria for copying
            if not self._should_copy_trade(swap_details):
                return

            # Execute the copy trade
            self._execute_copy_trade(swap_details)
        except Exception as e:
            logger.error(f"Error handling transaction: {e}")

    def _is_swap_transaction(self, transaction_data: Dict[str, Any]) -> bool:
        """
        Check if a transaction is a swap.

        Args:
            transaction_data: The transaction data

        Returns:
            True if it's a swap transaction, False otherwise
        """
        # Check for Jupiter program ID
        jupiter_program_id = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"

        # Check if the transaction involves the Jupiter program
        account_keys = transaction_data.get("accountKeys", [])
        if jupiter_program_id not in account_keys:
            return False

        # Check for swap instructions
        instructions = transaction_data.get("instructions", [])
        for instruction in instructions:
            if instruction.get("programId") == jupiter_program_id:
                return True

        return False

    def _extract_swap_details(self, transaction_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract swap details from a transaction.

        Args:
            transaction_data: The transaction data

        Returns:
            Swap details if successful, None otherwise
        """
        try:
            # This is a simplified implementation
            # In a real implementation, we would parse the transaction data more thoroughly

            # Get the sender
            sender = transaction_data.get("feePayer")
            if not sender:
                return None

            # Look for token transfers
            token_transfers = []
            for event in transaction_data.get("tokenTransfers", []):
                token_transfers.append(event)

            # Look for native transfers (SOL)
            native_transfers = []
            for event in transaction_data.get("nativeTransfers", []):
                native_transfers.append(event)

            # Determine input and output tokens
            input_token = None
            input_amount = 0
            output_token = None
            output_amount = 0

            # Check if SOL was sent
            for transfer in native_transfers:
                if transfer.get("fromUserAccount") == sender:
                    input_token = self.sol_mint
                    input_amount = transfer.get("amount", 0) / 1_000_000_000  # Convert lamports to SOL
                    break

            # Check if a token was sent
            if not input_token:
                for transfer in token_transfers:
                    if transfer.get("fromUserAccount") == sender:
                        input_token = transfer.get("mint")
                        input_amount = transfer.get("tokenAmount", 0)
                        break

            # Check if SOL was received
            for transfer in native_transfers:
                if transfer.get("toUserAccount") == sender:
                    output_token = self.sol_mint
                    output_amount = transfer.get("amount", 0) / 1_000_000_000  # Convert lamports to SOL
                    break

            # Check if a token was received
            if not output_token:
                for transfer in token_transfers:
                    if transfer.get("toUserAccount") == sender:
                        output_token = transfer.get("mint")
                        output_amount = transfer.get("tokenAmount", 0)
                        break

            # Ensure we have both input and output
            if not input_token or not output_token:
                return None

            # Create swap details
            return {
                "sender": sender,
                "input_token": input_token,
                "input_amount": input_amount,
                "output_token": output_token,
                "output_amount": output_amount,
                "signature": transaction_data.get("signature"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error extracting swap details: {e}")
            return None

    def _should_copy_trade(self, swap_details: Dict[str, Any]) -> bool:
        """
        Determine if a trade should be copied.

        Args:
            swap_details: The swap details

        Returns:
            True if the trade should be copied, False otherwise
        """
        # Check if the token is blacklisted
        input_token = swap_details["input_token"]
        output_token = swap_details["output_token"]

        if input_token in self.blacklisted_tokens or output_token in self.blacklisted_tokens:
            logger.info(f"Token {input_token} or {output_token} is blacklisted, not copying trade")
            return False

        # Check if it's a buy (SOL or USDC to token)
        is_buy = (input_token == self.sol_mint or input_token == self.usdc_mint) and output_token != self.sol_mint and output_token != self.usdc_mint

        # Check if it's a sell (token to SOL or USDC)
        is_sell = (output_token == self.sol_mint or output_token == self.usdc_mint) and input_token != self.sol_mint and input_token != self.usdc_mint

        # Only copy buys and sells
        if not is_buy and not is_sell:
            logger.info(f"Not a buy or sell, not copying trade")
            return False

        # Check transaction size
        transaction_size_sol = 0
        if input_token == self.sol_mint:
            transaction_size_sol = swap_details["input_amount"]
        elif output_token == self.sol_mint:
            transaction_size_sol = swap_details["output_amount"]
        else:
            # Try to convert to SOL value
            try:
                if is_buy:
                    # Get the price of the output token in SOL
                    price = jupiter_api.get_token_price(output_token)
                    transaction_size_sol = swap_details["output_amount"] * price
                else:
                    # Get the price of the input token in SOL
                    price = jupiter_api.get_token_price(input_token)
                    transaction_size_sol = swap_details["input_amount"] * price
            except Exception as e:
                logger.error(f"Error converting transaction size to SOL: {e}")
                return False

        # Check if transaction size is within limits
        if transaction_size_sol < self.min_transaction_sol:
            logger.info(f"Transaction size ({transaction_size_sol} SOL) is below minimum ({self.min_transaction_sol} SOL), not copying trade")
            return False

        if transaction_size_sol > self.max_transaction_sol:
            logger.info(f"Transaction size ({transaction_size_sol} SOL) is above maximum ({self.max_transaction_sol} SOL), not copying trade")
            return False

        return True

    def _execute_copy_trade(self, swap_details: Dict[str, Any]) -> None:
        """
        Execute a copy trade.

        Args:
            swap_details: The swap details
        """
        try:
            # Get the current wallet
            wallet = wallet_manager.get_current_keypair()
            if not wallet:
                logger.error("No wallet connected, cannot execute copy trade")
                return

            # Check if it's a buy or sell
            input_token = swap_details["input_token"]
            output_token = swap_details["output_token"]

            is_buy = (input_token == self.sol_mint or input_token == self.usdc_mint) and output_token != self.sol_mint and output_token != self.usdc_mint

            # Calculate the amount to trade
            if is_buy:
                # It's a buy (SOL/USDC to token)
                if input_token == self.sol_mint:
                    # Calculate amount in SOL
                    original_amount_sol = swap_details["input_amount"]
                    copy_amount_sol = original_amount_sol * (self.copy_percentage / 100)

                    # Cap at max transaction size
                    copy_amount_sol = min(copy_amount_sol, self.max_transaction_sol)

                    logger.info(f"Copying buy: {copy_amount_sol} SOL for token {output_token}")

                    # Get priority fee
                    priority_fee = jupiter_api.get_priority_fee()

                    # Execute the buy
                    tx_signature = jupiter_api.execute_buy(
                        token_mint=output_token,
                        amount_sol=copy_amount_sol,
                        wallet=wallet,
                        priority_fee=priority_fee
                    )

                    logger.info(f"Copy trade executed: {tx_signature}")

                    # Get the token price
                    try:
                        price = jupiter_api.get_token_price(output_token)

                        # Calculate the expected amount of tokens
                        expected_tokens = copy_amount_sol / price

                        # Create a position for tracking
                        position_manager.create_position_from_buy(
                            token_mint=output_token,
                            token_name=f"Copy_{output_token[:6]}",
                            amount_token=expected_tokens,
                            price_sol=price,
                            decimals=9  # Assume 9 decimals for now
                        )

                        logger.info(f"Created position for copied token {output_token}")
                    except Exception as e:
                        logger.error(f"Error setting up position for copied token {output_token}: {e}")
            else:
                # It's a sell (token to SOL/USDC)
                # Check if we have a position for this token
                position = position_manager.get_position(input_token)
                if position:
                    logger.info(f"Copying sell: token {input_token} to {output_token}")

                    # Get priority fee
                    priority_fee = jupiter_api.get_priority_fee()

                    # Execute the sell (sell all)
                    tx_signature = position_manager.execute_sell(input_token, "copy_trade")

                    logger.info(f"Copy trade executed: {tx_signature}")
                else:
                    logger.info(f"No position found for token {input_token}, not copying sell")
        except Exception as e:
            logger.error(f"Error executing copy trade: {e}")

    async def auto_discover_wallets(self) -> Dict[str, Any]:
        """Automatically discover and add profitable wallets."""
        if not self.smart_discovery_enabled:
            return {"success": False, "message": "Smart discovery is disabled"}

        try:
            logger.info("Starting automatic wallet discovery...")

            # Discover wallets using smart discovery
            discovered_wallets = await smart_wallet_discovery.discover_wallets()

            added_wallets = []
            skipped_wallets = []

            for wallet_data in discovered_wallets:
                wallet_address = wallet_data['address']

                # Check if wallet is already tracked
                if wallet_address in self.tracked_wallets:
                    skipped_wallets.append({
                        'address': wallet_address,
                        'reason': 'Already tracked'
                    })
                    continue

                # Check if wallet meets copy criteria
                if self._meets_copy_criteria(wallet_data):
                    # Calculate optimal parameters
                    multiplier = self._calculate_optimal_multiplier(wallet_data)
                    max_copy_amount = self._calculate_max_copy_amount(wallet_data)

                    # Add wallet
                    self.add_tracked_wallet(
                        wallet_address,
                        multiplier=multiplier,
                        max_copy_amount=max_copy_amount,
                        enabled=True
                    )

                    added_wallets.append({
                        'address': wallet_address,
                        'score': wallet_data['total_score'],
                        'multiplier': multiplier,
                        'max_copy_amount': max_copy_amount
                    })
                else:
                    skipped_wallets.append({
                        'address': wallet_address,
                        'reason': 'Does not meet copy criteria'
                    })

            logger.info(f"Auto-discovery completed: {len(added_wallets)} added, {len(skipped_wallets)} skipped")

            return {
                "success": True,
                "added_wallets": added_wallets,
                "skipped_wallets": skipped_wallets,
                "total_discovered": len(discovered_wallets)
            }

        except Exception as e:
            logger.error(f"Error in auto wallet discovery: {e}")
            return {"success": False, "message": str(e)}

    def _meets_copy_criteria(self, wallet_data: Dict) -> bool:
        """Check if discovered wallet meets copy trading criteria."""
        try:
            # Check minimum score
            if wallet_data.get('total_score', 0) < get_config_value("min_wallet_score", 70):
                return False

            # Check confidence level
            if wallet_data.get('confidence_level', 0) < 0.5:
                return False

            # Check performance metrics
            performance_score = wallet_data.get('performance_score', 0)
            if performance_score < 60:  # Minimum 60% performance score
                return False

            # Check risk score (higher is better)
            risk_score = wallet_data.get('risk_score', 0)
            if risk_score < 50:  # Minimum 50% risk score
                return False

            return True

        except Exception as e:
            logger.warning(f"Error checking copy criteria: {e}")
            return False

    def _calculate_optimal_multiplier(self, wallet_data: Dict) -> float:
        """Calculate optimal copy multiplier based on wallet performance."""
        try:
            # Base multiplier
            base_multiplier = 1.0

            # Adjust based on performance score
            performance_score = wallet_data.get('performance_score', 50)
            performance_adjustment = (performance_score - 50) / 100  # -0.5 to +0.5

            # Adjust based on risk score (higher risk score = lower risk)
            risk_score = wallet_data.get('risk_score', 50)
            risk_adjustment = (risk_score - 50) / 200  # -0.25 to +0.25

            # Adjust based on confidence level
            confidence_level = wallet_data.get('confidence_level', 0.5)
            confidence_adjustment = (confidence_level - 0.5) * 0.5  # -0.25 to +0.25

            # Calculate final multiplier
            multiplier = base_multiplier + performance_adjustment + risk_adjustment + confidence_adjustment

            # Clamp between 0.1 and 2.0
            return max(0.1, min(2.0, multiplier))

        except Exception as e:
            logger.warning(f"Error calculating optimal multiplier: {e}")
            return 1.0

    def _calculate_max_copy_amount(self, wallet_data: Dict) -> float:
        """Calculate maximum copy amount based on wallet performance."""
        try:
            # Base amount
            base_amount = self.max_transaction_sol

            # Adjust based on total score
            total_score = wallet_data.get('total_score', 50)
            score_multiplier = total_score / 100  # 0.0 to 1.0+

            # Adjust based on confidence level
            confidence_level = wallet_data.get('confidence_level', 0.5)

            # Calculate final amount
            max_amount = base_amount * score_multiplier * confidence_level

            # Clamp between minimum and maximum limits
            min_amount = 0.05  # 0.05 SOL minimum
            max_amount_limit = 5.0  # 5 SOL maximum

            return max(min_amount, min(max_amount_limit, max_amount))

        except Exception as e:
            logger.warning(f"Error calculating max copy amount: {e}")
            return self.max_transaction_sol


# Create a singleton instance
copy_trading = CopyTrading()
