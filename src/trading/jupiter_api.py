"""
Jupiter API integration for the Solana Memecoin Trading Bot.
Handles price fetching and swap execution using Jupiter V6 API.
Includes enhanced transaction handling and MEV protection.
"""

import json
import logging
import time
import statistics
import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Union, Tuple

import requests
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solana.transaction import Transaction
from solders.keypair import Keypair
from solders.signature import Signature
from solders.pubkey import Pubkey
from solders.instruction import Instruction
from solders.message import Message
from solana.rpc.commitment import Commitment

from config import get_config_value
from src.solana.solana_interact import solana_client
from src.solana.jito_mev import jito_mev
from src.solana.gas_optimizer import gas_optimizer, TransactionType, TransactionPriority
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class JupiterAPI:
    """Client for interacting with Jupiter V6 API."""

    def __init__(self):
        """Initialize the Jupiter API client."""
        self.api_url = get_config_value("jupiter_api_url", "https://quote-api.jup.ag/v6")
        self.slippage_bps = int(get_config_value("slippage_bps", "50"))  # Default 0.5%
        self.rpc_url = get_config_value("rpc_url")
        self.solana_client = solana_client

        # Performance optimizations
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SolanaMemecoinBot/1.0'
        })

        # Connection pooling for better performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        # Enhanced caching system
        self._price_cache = {}
        self._quote_cache = {}
        self._token_info_cache = {}
        self._cache_ttl = 5  # 5 seconds cache TTL
        self._quote_cache_ttl = 2  # 2 seconds for quotes
        self._token_info_cache_ttl = 300  # 5 minutes for token info
        self._last_cache_clear = time.time()

        # Async session for concurrent operations
        self._async_session = None
        self._executor = ThreadPoolExecutor(max_workers=10)

        # Performance metrics
        self._request_count = 0
        self._cache_hits = 0
        self._total_request_time = 0.0

    def _clear_expired_cache(self) -> None:
        """Clear expired cache entries."""
        current_time = time.time()

        # Clear price cache
        expired_keys = [
            key for key, (_, timestamp) in self._price_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        for key in expired_keys:
            del self._price_cache[key]

        # Clear quote cache
        expired_keys = [
            key for key, (_, timestamp) in self._quote_cache.items()
            if current_time - timestamp > self._quote_cache_ttl
        ]
        for key in expired_keys:
            del self._quote_cache[key]

        # Clear token info cache
        expired_keys = [
            key for key, (_, timestamp) in self._token_info_cache.items()
            if current_time - timestamp > self._token_info_cache_ttl
        ]
        for key in expired_keys:
            del self._token_info_cache[key]

        self._last_cache_clear = current_time

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the Jupiter API client."""
        cache_hit_rate = (self._cache_hits / max(self._request_count, 1)) * 100
        avg_request_time = self._total_request_time / max(self._request_count, 1)

        return {
            "total_requests": self._request_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "average_request_time": f"{avg_request_time:.3f}s",
            "price_cache_size": len(self._price_cache),
            "quote_cache_size": len(self._quote_cache),
            "token_info_cache_size": len(self._token_info_cache)
        }

    def get_token_price(self, token_mint: str, quote_token: str = "So11111111111111111111111111111111111111112") -> float:
        """
        Get the price of a token in terms of another token (default: SOL) with caching.

        Args:
            token_mint: The mint address of the token to get the price for
            quote_token: The mint address of the quote token (default: SOL)

        Returns:
            The price of the token in terms of the quote token
        """
        # Clear expired cache entries periodically
        if time.time() - self._last_cache_clear > 30:  # Every 30 seconds
            self._clear_expired_cache()

        # Check cache first
        cache_key = f"{token_mint}:{quote_token}"
        current_time = time.time()

        if cache_key in self._price_cache:
            price, timestamp = self._price_cache[cache_key]
            if current_time - timestamp < self._cache_ttl:
                self._cache_hits += 1
                logger.debug(f"Cache hit for price {token_mint}: {price}")
                return price

        try:
            start_time = time.time()
            self._request_count += 1

            # Use Jupiter's price API
            url = f"{self.api_url}/price"
            params = {
                "ids": token_mint,
                "vsToken": quote_token
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if token_mint in data["data"]:
                price = float(data["data"][token_mint]["price"])

                # Cache the result
                self._price_cache[cache_key] = (price, current_time)

                # Update performance metrics
                self._total_request_time += time.time() - start_time

                logger.info(f"Price for {token_mint}: {price} {quote_token}")
                return price
            else:
                logger.error(f"No price data found for {token_mint}")
                raise ValueError(f"No price data found for {token_mint}")
        except Exception as e:
            logger.error(f"Error getting token price: {e}")
            raise

    def get_quote(self, input_mint: str, output_mint: str, amount: float,
                  slippage_bps: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a quote for swapping tokens with caching.

        Args:
            input_mint: The mint address of the input token
            output_mint: The mint address of the output token
            amount: The amount of input tokens to swap
            slippage_bps: The slippage tolerance in basis points (optional)

        Returns:
            The quote data from Jupiter
        """
        # Check cache first (quotes are cached for a shorter time)
        cache_key = f"{input_mint}:{output_mint}:{int(amount)}:{slippage_bps or self.slippage_bps}"
        current_time = time.time()

        if cache_key in self._quote_cache:
            quote_data, timestamp = self._quote_cache[cache_key]
            if current_time - timestamp < self._quote_cache_ttl:
                self._cache_hits += 1
                logger.debug(f"Cache hit for quote {input_mint} -> {output_mint}")
                return quote_data

        try:
            start_time = time.time()
            self._request_count += 1

            url = f"{self.api_url}/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(int(amount)),
                "slippageBps": slippage_bps or self.slippage_bps,
                "onlyDirectRoutes": "false",
                "asLegacyTransaction": "true"  # Use legacy transactions for simplicity
            }

            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()

            quote_data = response.json()

            # Cache the result
            self._quote_cache[cache_key] = (quote_data, current_time)

            # Update performance metrics
            self._total_request_time += time.time() - start_time

            logger.info(f"Quote received: {amount} {input_mint} -> {quote_data['outAmount']} {output_mint}")
            return quote_data
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            raise

    def execute_swap(self, quote_data: Dict[str, Any], wallet: Keypair,
                     priority_fee: Optional[int] = None) -> str:
        """
        Execute a swap based on a quote.

        Args:
            quote_data: The quote data from Jupiter
            wallet: The keypair to sign the transaction with
            priority_fee: Optional priority fee in micro-lamports

        Returns:
            The transaction signature
        """
        try:
            # Get the swap transaction
            url = f"{self.api_url}/swap"
            payload = {
                "quoteResponse": quote_data,
                "userPublicKey": str(wallet.pubkey()),
                "wrapAndUnwrapSol": True,
                "prioritizationFeeLamports": priority_fee or 0
            }

            response = requests.post(url, json=payload)
            response.raise_for_status()

            swap_data = response.json()

            # Get the transaction from the response
            tx_data = swap_data["swapTransaction"]

            # Sign and send the transaction
            tx_signature = self._sign_and_send_transaction(tx_data, wallet)

            logger.info(f"Swap executed: {tx_signature}")
            return tx_signature
        except Exception as e:
            logger.error(f"Error executing swap: {e}")
            raise

    def _sign_and_send_transaction(self, tx_data: str, wallet: Keypair) -> str:
        """
        Sign and send a transaction with enhanced handling and MEV protection.

        Args:
            tx_data: The transaction data as a base64 string
            wallet: The keypair to sign the transaction with

        Returns:
            The transaction signature
        """
        try:
            # Decode the transaction
            transaction = Transaction.deserialize(bytes.fromhex(tx_data))

            # Get priority fee using gas optimizer
            priority_fee = self.get_priority_fee(transaction_type="swap")

            # Get compute limit from gas optimizer
            compute_limit = gas_optimizer.get_compute_limit(TransactionType.SWAP)

            # Check if MEV protection is enabled
            use_mev_protection = get_config_value("mev_protection_enabled", False)

            if use_mev_protection and jito_mev.is_available():
                # Use Jito MEV protection
                logger.info("Using Jito MEV protection for transaction")
                signature = jito_mev.send_transaction(transaction, wallet, priority_fee, compute_limit)
            else:
                # Use enhanced transaction handling
                signature = solana_client.send_transaction(transaction, wallet, priority_fee, compute_limit)

            # Wait for confirmation
            self._confirm_transaction(signature)

            return signature
        except Exception as e:
            logger.error(f"Error signing and sending transaction: {e}")
            raise

    def _confirm_transaction(self, signature: str, max_retries: int = 30, retry_delay: float = 1.0) -> bool:
        """
        Wait for a transaction to be confirmed with enhanced reliability.

        Args:
            signature: The transaction signature
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds

        Returns:
            True if the transaction was confirmed, False otherwise
        """
        logger.info(f"Waiting for transaction confirmation: {signature}")

        for attempt in range(max_retries):
            try:
                # Use RPC manager with failover
                response = self.solana_client.rpc_manager.execute_with_failover(
                    "get_signature_statuses",
                    [signature],
                    {"searchTransactionHistory": attempt > 5}  # Search history after a few attempts
                )

                status = response["result"]["value"][0]

                if status is not None:
                    if status.get("err") is not None:
                        error_details = status.get("err")
                        logger.error(f"Transaction failed: {error_details}")

                        # Check for specific error types that might need special handling
                        if isinstance(error_details, dict) and "InstructionError" in str(error_details):
                            logger.error(f"Instruction error in transaction: {error_details}")

                        return False

                    confirmation_status = status.get("confirmationStatus")
                    if confirmation_status == "confirmed" or confirmation_status == "finalized":
                        logger.info(f"Transaction confirmed with status '{confirmation_status}': {signature}")

                        # For finalized transactions, we're done
                        if confirmation_status == "finalized":
                            return True

                        # For confirmed transactions, we might want to wait for finalization
                        # depending on the configuration
                        if not get_config_value("wait_for_finalization", False):
                            return True

                        # If we're waiting for finalization, continue but with longer delay
                        retry_delay = 2.0

            except Exception as e:
                logger.warning(f"Error checking transaction status (attempt {attempt+1}/{max_retries}): {e}")

            # Exponential backoff for retries
            adjusted_delay = retry_delay * (1.1 ** min(attempt, 10))
            time.sleep(adjusted_delay)

        logger.error(f"Transaction confirmation timed out after {max_retries} attempts: {signature}")
        return False

    def execute_buy(self, token_mint: str, amount_sol: float, wallet: Keypair,
                    priority_fee: Optional[int] = None) -> str:
        """
        Buy a token with SOL.

        Args:
            token_mint: The mint address of the token to buy
            amount_sol: The amount of SOL to spend
            wallet: The keypair to sign the transaction with
            priority_fee: Optional priority fee in micro-lamports

        Returns:
            The transaction signature
        """
        # SOL mint address
        sol_mint = "So11111111111111111111111111111111111111112"

        # Convert SOL to lamports
        amount_lamports = int(amount_sol * 1_000_000_000)

        # Get a quote
        quote_data = self.get_quote(sol_mint, token_mint, amount_lamports)

        # Get priority fee for buy transaction
        if priority_fee is None:
            priority_fee = self.get_priority_fee(transaction_type="buy")

        # Execute the swap
        return self.execute_swap(quote_data, wallet, priority_fee)

    def execute_sell(self, token_mint: str, amount_token: float, decimals: int, wallet: Keypair,
                     priority_fee: Optional[int] = None) -> str:
        """
        Sell a token for SOL.

        Args:
            token_mint: The mint address of the token to sell
            amount_token: The amount of tokens to sell
            decimals: The number of decimals for the token
            wallet: The keypair to sign the transaction with
            priority_fee: Optional priority fee in micro-lamports

        Returns:
            The transaction signature
        """
        # SOL mint address
        sol_mint = "So11111111111111111111111111111111111111112"

        # Convert token amount to smallest unit
        amount_raw = int(amount_token * (10 ** decimals))

        # Get a quote
        quote_data = self.get_quote(token_mint, sol_mint, amount_raw)

        # Get priority fee for sell transaction
        if priority_fee is None:
            priority_fee = self.get_priority_fee(transaction_type="sell")

        # Execute the swap
        return self.execute_swap(quote_data, wallet, priority_fee)

    def get_priority_fee(self, percentile: int = None, transaction_type: str = "default") -> int:
        """
        Get the current priority fee based on recent transactions with advanced optimization.

        Args:
            percentile: The percentile to use (default: from config)
            transaction_type: Type of transaction ("buy", "sell", "snipe", "default")

        Returns:
            The priority fee in micro-lamports
        """
        try:
            # Use the gas optimizer if enabled
            if get_config_value("fee_optimization_enabled", True):
                # Map transaction type to TransactionType enum
                tx_type_map = {
                    "buy": TransactionType.BUY,
                    "sell": TransactionType.SELL,
                    "snipe": TransactionType.SNIPE,
                    "swap": TransactionType.SWAP,
                    "limit_order": TransactionType.LIMIT_ORDER,
                    "withdraw": TransactionType.WITHDRAW,
                    "default": TransactionType.DEFAULT
                }

                # Map percentile to priority level
                priority_level = TransactionPriority.MEDIUM
                if percentile is not None:
                    if percentile <= 25:
                        priority_level = TransactionPriority.LOW
                    elif percentile <= 50:
                        priority_level = TransactionPriority.MEDIUM
                    elif percentile <= 75:
                        priority_level = TransactionPriority.HIGH
                    else:
                        priority_level = TransactionPriority.URGENT

                # Get tx_type from map or default
                tx_type = tx_type_map.get(transaction_type, TransactionType.DEFAULT)

                # Get optimized fee from gas optimizer
                fee = gas_optimizer.get_priority_fee(priority_level, tx_type)

                logger.info(f"Optimized priority fee for {transaction_type} ({priority_level.value}): {fee} micro-lamports")
                return fee

            # Fall back to legacy implementation if gas optimizer is disabled
            # Use config percentile if not specified
            if percentile is None:
                percentile = int(get_config_value("priority_fee_percentile", 75))

            # Get minimum fee from config
            min_fee = int(get_config_value("min_priority_fee", 1000))

            # Get transaction-specific fee multipliers
            fee_multipliers = get_config_value("priority_fee_multipliers", {
                "buy": 1.0,
                "sell": 1.0,
                "snipe": 2.0,  # Higher priority for sniping
                "swap": 1.2,    # Slightly higher for swaps
                "default": 1.0
            })

            # Get fee multiplier for this transaction type
            multiplier = fee_multipliers.get(transaction_type, fee_multipliers["default"])

            # Get network congestion multiplier
            network_congestion = self._get_network_congestion()
            congestion_multiplier = 1.0

            if network_congestion > 0.8:  # High congestion
                congestion_multiplier = 1.5
                logger.info(f"High network congestion detected ({network_congestion:.2f}), applying 1.5x multiplier")
            elif network_congestion > 0.5:  # Medium congestion
                congestion_multiplier = 1.2
                logger.info(f"Medium network congestion detected ({network_congestion:.2f}), applying 1.2x multiplier")

            # Get recent priority fees using the RPC manager
            fees_by_percentile = self.solana_client.get_recent_priority_fee(percentile)

            # If no fees found, use minimum fee
            if not fees_by_percentile:
                logger.warning("No recent prioritization fees found, using minimum fee")
                return max(int(min_fee * multiplier * congestion_multiplier), min_fee)

            # Get the fee at the specified percentile
            base_fee = fees_by_percentile.get(str(percentile), min_fee)

            # Apply multipliers and ensure minimum fee
            final_fee = max(int(base_fee * multiplier * congestion_multiplier), min_fee)

            # Apply time-of-day adjustment if enabled
            if get_config_value("time_based_fee_adjustment", False):
                time_multiplier = self._get_time_based_multiplier()
                if time_multiplier != 1.0:
                    final_fee = int(final_fee * time_multiplier)
                    logger.info(f"Applied time-based fee multiplier: {time_multiplier}x")

            # Apply transaction type specific adjustments
            if transaction_type == "snipe":
                # For sniping, we want to be extra competitive
                final_fee = max(final_fee, int(fees_by_percentile.get("90", final_fee) * 1.1))
                logger.info(f"Adjusted snipe fee to be more competitive: {final_fee}")
            elif transaction_type == "sell" and get_config_value("urgent_sell_fee_boost", False):
                # For urgent sells, boost the fee
                final_fee = int(final_fee * 1.3)
                logger.info(f"Applied urgent sell fee boost: {final_fee}")

            logger.info(f"Priority fee for {transaction_type}: {final_fee} micro-lamports (base: {base_fee}, " +
                       f"multiplier: {multiplier}, congestion: {congestion_multiplier:.2f})")
            return final_fee
        except Exception as e:
            logger.error(f"Error calculating priority fee: {e}")
            return int(get_config_value("min_priority_fee", 1000))  # Default to minimum fee

    def _get_network_congestion(self) -> float:
        """
        Calculate network congestion level (0.0 to 1.0).

        Returns:
            Congestion level from 0.0 (low) to 1.0 (high)
        """
        try:
            # Get recent performance samples
            response = self.solana_client.rpc_manager.execute_with_failover("get_recent_performance_samples")

            if "result" not in response or not response["result"]:
                return 0.5  # Default to medium congestion if no data

            # Get the most recent samples
            samples = response["result"][:5]  # Last 5 samples

            if not samples:
                return 0.5

            # Calculate average TPS and max TPS
            avg_tps = sum(sample["numTransactions"] / sample["samplePeriodSecs"] for sample in samples) / len(samples)

            # Solana's theoretical max TPS is around 50,000, but practical is lower
            # We'll use 20,000 as a reference point
            max_practical_tps = 20000

            # Calculate congestion as inverse of TPS ratio (lower TPS = higher congestion)
            congestion = 1.0 - min(avg_tps / max_practical_tps, 1.0)

            return congestion
        except Exception as e:
            logger.warning(f"Error calculating network congestion: {e}")
            return 0.5  # Default to medium congestion on error

    def _get_time_based_multiplier(self) -> float:
        """
        Get a fee multiplier based on time of day.

        Returns:
            Fee multiplier based on time of day
        """
        try:
            # Get current hour (UTC)
            current_hour = datetime.datetime.now(datetime.timezone.utc).hour

            # Define peak hours (typically US and Asian market hours)
            us_peak_hours = range(13, 21)  # 13:00-21:00 UTC (9am-5pm EST)
            asia_peak_hours = range(0, 8)  # 00:00-08:00 UTC (8am-4pm Asia)

            if current_hour in us_peak_hours:
                return 1.2  # 20% increase during US peak
            elif current_hour in asia_peak_hours:
                return 1.1  # 10% increase during Asia peak
            else:
                return 1.0  # No adjustment during off-peak
        except Exception as e:
            logger.warning(f"Error calculating time-based fee multiplier: {e}")
            return 1.0  # No adjustment on error

    async def get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async session for concurrent operations."""
        if self._async_session is None or self._async_session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
            self._async_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'SolanaMemecoinBot/1.0'
                }
            )
        return self._async_session

    async def get_token_price_async(self, token_mint: str, quote_token: str = "So11111111111111111111111111111111111111112") -> float:
        """
        Async version of get_token_price for concurrent operations.

        Args:
            token_mint: The mint address of the token to get the price for
            quote_token: The mint address of the quote token (default: SOL)

        Returns:
            The price of the token in terms of the quote token
        """
        # Check cache first
        cache_key = f"{token_mint}:{quote_token}"
        current_time = time.time()

        if cache_key in self._price_cache:
            price, timestamp = self._price_cache[cache_key]
            if current_time - timestamp < self._cache_ttl:
                self._cache_hits += 1
                return price

        try:
            start_time = time.time()
            self._request_count += 1

            session = await self.get_async_session()
            url = f"{self.api_url}/price"
            params = {
                "ids": token_mint,
                "vsToken": quote_token
            }

            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                if token_mint in data["data"]:
                    price = float(data["data"][token_mint]["price"])

                    # Cache the result
                    self._price_cache[cache_key] = (price, current_time)

                    # Update performance metrics
                    self._total_request_time += time.time() - start_time

                    return price
                else:
                    raise ValueError(f"No price data found for {token_mint}")
        except Exception as e:
            logger.error(f"Error getting token price async: {e}")
            raise

    async def get_multiple_token_prices_async(self, token_mints: List[str],
                                            quote_token: str = "So11111111111111111111111111111111111111112") -> Dict[str, float]:
        """
        Get prices for multiple tokens concurrently.

        Args:
            token_mints: List of token mint addresses
            quote_token: The mint address of the quote token (default: SOL)

        Returns:
            Dictionary mapping token mints to their prices
        """
        tasks = [
            self.get_token_price_async(token_mint, quote_token)
            for token_mint in token_mints
        ]

        try:
            prices = await asyncio.gather(*tasks, return_exceptions=True)
            result = {}

            for token_mint, price in zip(token_mints, prices):
                if isinstance(price, Exception):
                    logger.warning(f"Failed to get price for {token_mint}: {price}")
                    result[token_mint] = 0.0
                else:
                    result[token_mint] = price

            return result
        except Exception as e:
            logger.error(f"Error getting multiple token prices: {e}")
            raise

    def get_multiple_token_prices(self, token_mints: List[str],
                                quote_token: str = "So11111111111111111111111111111111111111112") -> Dict[str, float]:
        """
        Synchronous wrapper for getting multiple token prices concurrently.

        Args:
            token_mints: List of token mint addresses
            quote_token: The mint address of the quote token (default: SOL)

        Returns:
            Dictionary mapping token mints to their prices
        """
        try:
            # Run the async function in the executor
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.get_multiple_token_prices_async(token_mints, quote_token)
                )
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in sync wrapper for multiple token prices: {e}")
            # Fallback to sequential requests
            result = {}
            for token_mint in token_mints:
                try:
                    result[token_mint] = self.get_token_price(token_mint, quote_token)
                except Exception as token_error:
                    logger.warning(f"Failed to get price for {token_mint}: {token_error}")
                    result[token_mint] = 0.0
            return result

    def get_token_info(self, token_mint: str) -> Optional[Dict[str, Any]]:
        """
        Get token information with caching.

        Args:
            token_mint: The token mint address

        Returns:
            Token information or None if not found
        """
        # Check cache first
        current_time = time.time()

        if token_mint in self._token_info_cache:
            token_info, timestamp = self._token_info_cache[token_mint]
            if current_time - timestamp < self._token_info_cache_ttl:
                self._cache_hits += 1
                return token_info

        try:
            start_time = time.time()
            self._request_count += 1

            # Use Jupiter's token list API
            url = f"{self.api_url}/tokens"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            tokens = response.json()

            # Find the token in the list
            token_info = None
            for token in tokens:
                if token.get("address") == token_mint:
                    token_info = token
                    break

            # Cache the result (even if None)
            self._token_info_cache[token_mint] = (token_info, current_time)

            # Update performance metrics
            self._total_request_time += time.time() - start_time

            return token_info
        except Exception as e:
            logger.warning(f"Error getting token info for {token_mint}: {e}")
            return None

    async def close_async_session(self):
        """Close the async session."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Create a singleton instance
jupiter_api = JupiterAPI()
