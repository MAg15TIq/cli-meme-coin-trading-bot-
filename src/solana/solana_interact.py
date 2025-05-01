"""
Solana interaction module for the Solana Memecoin Trading Bot.
Handles connections to the Solana blockchain and basic operations.
Includes enhanced RPC reliability with multi-RPC failover and performance tracking.
"""

import json
import logging
import time
import threading
import statistics
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solders.pubkey import Pubkey as PublicKey
from solana.transaction import Transaction
from solders.keypair import Keypair
from solders.signature import Signature
from spl.token.client import Token
from spl.token.constants import TOKEN_PROGRAM_ID
from solana.rpc.commitment import Commitment
from solders.message import Message
from solders.transaction import VersionedTransaction, Transaction as SoldersTransaction
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class RPCEndpoint:
    """Class representing a Solana RPC endpoint with performance metrics."""

    def __init__(self, name: str, url: str):
        """
        Initialize an RPC endpoint.

        Args:
            name: Name of the endpoint
            url: URL of the endpoint
        """
        self.name = name
        self.url = url
        self.client = Client(url)
        self.available = True
        self.last_checked = datetime.now()
        self.response_times: List[float] = []
        self.success_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error = None
        self.last_error_time = None

    def get_avg_response_time(self) -> float:
        """Get the average response time in milliseconds."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times[-20:])  # Last 20 responses

    def record_success(self, response_time: float) -> None:
        """
        Record a successful RPC call.

        Args:
            response_time: Response time in milliseconds
        """
        self.available = True
        self.last_checked = datetime.now()
        self.response_times.append(response_time)
        self.success_count += 1
        self.consecutive_errors = 0

        # Keep only the last 100 response times
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]

    def record_error(self, error: Exception) -> None:
        """
        Record an RPC call error.

        Args:
            error: The exception that occurred
        """
        self.last_checked = datetime.now()
        self.error_count += 1
        self.consecutive_errors += 1
        self.last_error = str(error)
        self.last_error_time = datetime.now()

        # Mark as unavailable after 3 consecutive errors
        if self.consecutive_errors >= 3:
            self.available = False

    def get_health_score(self) -> float:
        """
        Calculate a health score for this endpoint (0-100).

        Returns:
            Health score from 0 (worst) to 100 (best)
        """
        if not self.available:
            return 0.0

        # Calculate success rate (70% of score)
        total_calls = self.success_count + self.error_count
        success_rate = self.success_count / max(total_calls, 1) * 70

        # Calculate response time score (30% of score)
        # Assume anything under 100ms is great, over 1000ms is poor
        avg_time = self.get_avg_response_time()
        if avg_time <= 100:
            time_score = 30
        elif avg_time >= 1000:
            time_score = 0
        else:
            time_score = 30 * (1 - (avg_time - 100) / 900)

        return success_rate + time_score


class RPCManager:
    """Manager for multiple RPC endpoints with failover and load balancing."""

    def __init__(self):
        """Initialize the RPC manager."""
        self.endpoints: Dict[str, RPCEndpoint] = {}
        self.primary_endpoint_name = "default"
        self.current_endpoint_name = "default"
        self.health_check_interval = int(get_config_value("rpc_health_check_interval", "300"))  # 5 minutes
        self.last_health_check = datetime.now()
        self.health_check_thread = None
        self.stop_health_check = threading.Event()

        # Load endpoints from config
        self._load_endpoints()

        # Start health check thread
        self._start_health_check_thread()

        logger.info(f"Initialized RPC Manager with {len(self.endpoints)} endpoints")

    def _load_endpoints(self) -> None:
        """Load RPC endpoints from config."""
        # Get endpoints from config
        rpc_endpoints = get_config_value("rpc_endpoints", {
            "default": get_config_value("rpc_url", "https://api.mainnet-beta.solana.com")
        })

        # Add each endpoint
        for name, url in rpc_endpoints.items():
            if url:  # Only add if URL is not empty
                self.endpoints[name] = RPCEndpoint(name, url)

        # Set primary endpoint
        self.primary_endpoint_name = get_config_value("primary_rpc_endpoint", "default")
        if self.primary_endpoint_name not in self.endpoints:
            self.primary_endpoint_name = next(iter(self.endpoints.keys())) if self.endpoints else "default"

        # Set current endpoint to primary
        self.current_endpoint_name = self.primary_endpoint_name

    def _start_health_check_thread(self) -> None:
        """Start the health check thread."""
        if self.health_check_thread and self.health_check_thread.is_alive():
            return

        # Clear stop event
        self.stop_health_check.clear()

        # Start thread
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        logger.info("Started RPC health check thread")

    def _health_check_loop(self) -> None:
        """Main health check loop."""
        while not self.stop_health_check.is_set():
            try:
                self._check_all_endpoints()
                self._select_best_endpoint()
            except Exception as e:
                logger.error(f"Error in RPC health check loop: {e}")

            # Sleep until next check
            self.stop_health_check.wait(self.health_check_interval)

    def _check_all_endpoints(self) -> None:
        """Check health of all endpoints."""
        logger.debug("Checking health of all RPC endpoints")

        for name, endpoint in self.endpoints.items():
            try:
                # Measure response time for getSlot
                start_time = time.time()
                response = endpoint.client.get_slot()
                end_time = time.time()

                # Check if response is valid
                if "result" in response:
                    response_time = (end_time - start_time) * 1000  # Convert to ms
                    endpoint.record_success(response_time)
                    logger.debug(f"RPC endpoint {name} is healthy: {response_time:.2f}ms")
                else:
                    endpoint.record_error(Exception("Invalid response"))
                    logger.warning(f"RPC endpoint {name} returned invalid response")
            except Exception as e:
                endpoint.record_error(e)
                logger.warning(f"RPC endpoint {name} is unhealthy: {str(e)}")

        self.last_health_check = datetime.now()

    def _select_best_endpoint(self) -> None:
        """Select the best endpoint based on health scores."""
        best_score = -1
        best_endpoint = None

        for name, endpoint in self.endpoints.items():
            if endpoint.available:
                score = endpoint.get_health_score()
                if score > best_score:
                    best_score = score
                    best_endpoint = name

        if best_endpoint and best_endpoint != self.current_endpoint_name:
            logger.info(f"Switching RPC endpoint from {self.current_endpoint_name} to {best_endpoint}")
            self.current_endpoint_name = best_endpoint

    def get_endpoint(self, preferred_endpoint: Optional[str] = None) -> RPCEndpoint:
        """
        Get the current best endpoint or a specific preferred endpoint if available.

        Args:
            preferred_endpoint: Optional preferred endpoint name

        Returns:
            The selected RPC endpoint
        """
        # If preferred endpoint is specified and available, use it
        if preferred_endpoint and preferred_endpoint in self.endpoints:
            endpoint = self.endpoints[preferred_endpoint]
            if endpoint.available:
                return endpoint

        # Otherwise use current best endpoint
        if self.current_endpoint_name in self.endpoints:
            return self.endpoints[self.current_endpoint_name]

        # Fallback to any available endpoint
        for name, endpoint in self.endpoints.items():
            if endpoint.available:
                return endpoint

        # If no endpoints are available, use primary anyway
        logger.error("No healthy RPC endpoints available, using primary endpoint")
        return self.endpoints.get(self.primary_endpoint_name, next(iter(self.endpoints.values())))

    def execute_with_failover(self, method_name: str, *args, **kwargs) -> Any:
        """
        Execute an RPC method with automatic failover.

        Args:
            method_name: The RPC method name to call
            *args: Arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            The result of the RPC call
        """
        # Get current endpoint
        endpoint = self.get_endpoint()

        # Try with current endpoint
        try:
            start_time = time.time()
            method = getattr(endpoint.client, method_name)
            response = method(*args, **kwargs)
            end_time = time.time()

            # Record success
            response_time = (end_time - start_time) * 1000  # Convert to ms
            endpoint.record_success(response_time)

            return response
        except Exception as e:
            # Record error
            endpoint.record_error(e)
            logger.warning(f"Error with RPC endpoint {endpoint.name}: {str(e)}")

            # Try failover with other endpoints
            for name, other_endpoint in self.endpoints.items():
                if name != endpoint.name and other_endpoint.available:
                    try:
                        logger.info(f"Trying failover to RPC endpoint {name}")
                        start_time = time.time()
                        method = getattr(other_endpoint.client, method_name)
                        response = method(*args, **kwargs)
                        end_time = time.time()

                        # Record success
                        response_time = (end_time - start_time) * 1000  # Convert to ms
                        other_endpoint.record_success(response_time)

                        # Switch current endpoint
                        self.current_endpoint_name = name
                        logger.info(f"Failover to RPC endpoint {name} successful")

                        return response
                    except Exception as e2:
                        other_endpoint.record_error(e2)
                        logger.warning(f"Failover to RPC endpoint {name} failed: {str(e2)}")

            # If all endpoints failed, raise the original exception
            logger.error(f"All RPC endpoints failed for method {method_name}")
            raise e


class SolanaClient:
    """Client for interacting with the Solana blockchain."""

    def __init__(self, rpc_url: Optional[str] = None):
        """
        Initialize the Solana client.

        Args:
            rpc_url: Optional RPC URL to override the one in config
        """
        self.rpc_url = rpc_url or get_config_value("rpc_url")
        self.client = Client(self.rpc_url)

        # Initialize RPC manager
        self.rpc_manager = RPCManager()

        logger.info(f"Initialized Solana client with RPC URL: {self.rpc_url}")

    def get_balance(self, public_key: Union[str, PublicKey]) -> float:
        """
        Get the SOL balance for a wallet.

        Args:
            public_key: The public key of the wallet

        Returns:
            The balance in SOL
        """
        # For testing, we can just use the string directly
        # In a real implementation, we would convert the string to a PublicKey

        try:
            # Use RPC manager with failover
            response = self.rpc_manager.execute_with_failover("get_balance", public_key)
            balance_lamports = response["result"]["value"]
            balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
            logger.debug(f"Balance for {public_key}: {balance_sol} SOL")
            return balance_sol
        except Exception as e:
            logger.error(f"Error getting balance for {public_key}: {e}")
            raise

    def get_token_accounts(self, owner_pubkey: Union[str, PublicKey]) -> List[Dict[str, Any]]:
        """
        Get all token accounts owned by a wallet.

        Args:
            owner_pubkey: The public key of the wallet owner

        Returns:
            List of token accounts with their details
        """
        # For testing, we can just use the string directly
        # In a real implementation, we would convert the string to a PublicKey

        try:
            # Use RPC manager with failover
            response = self.rpc_manager.execute_with_failover(
                "get_token_accounts_by_owner",
                owner_pubkey,
                {"programId": TOKEN_PROGRAM_ID}
            )

            token_accounts = []
            for account in response["result"]["value"]:
                account_data = account["account"]["data"]
                decoded_data = account_data["parsed"]["info"]

                if decoded_data["tokenAmount"]["uiAmount"] > 0:
                    token_accounts.append({
                        "mint": decoded_data["mint"],
                        "token_amount": decoded_data["tokenAmount"]["uiAmount"],
                        "decimals": decoded_data["tokenAmount"]["decimals"],
                        "account_address": account["pubkey"]
                    })

            return token_accounts
        except Exception as e:
            logger.error(f"Error getting token accounts for {owner_pubkey}: {e}")
            raise

    def get_token_balance(self, token_account: str) -> float:
        """
        Get the balance of a specific token account.

        Args:
            token_account: The token account address

        Returns:
            The token balance
        """
        try:
            # Use RPC manager with failover
            response = self.rpc_manager.execute_with_failover("get_token_account_balance", token_account)
            balance = response["result"]["value"]["uiAmount"]
            return balance
        except Exception as e:
            logger.error(f"Error getting token balance for {token_account}: {e}")
            raise

    def get_latest_blockhash(self) -> str:
        """
        Get the latest blockhash from the Solana blockchain.

        Returns:
            The latest blockhash as a string
        """
        try:
            # Use RPC manager with failover
            response = self.rpc_manager.execute_with_failover("get_latest_blockhash")
            return response["result"]["value"]["blockhash"]
        except Exception as e:
            logger.error(f"Error getting latest blockhash: {e}")
            raise

    def get_recent_priority_fee(self, percentile: int = 75) -> Dict[str, int]:
        """
        Get recent priority fees from the Solana blockchain.

        Args:
            percentile: The percentile to include in the result

        Returns:
            Dictionary of priority fees by percentile
        """
        try:
            # Use RPC manager with failover
            response = self.rpc_manager.execute_with_failover("get_recent_prioritization_fees")

            # Process the response
            fees = {}
            if "result" in response and response["result"]:
                # Extract all fees
                all_fees = [fee["prioritizationFee"] for fee in response["result"]]

                # Sort fees
                all_fees.sort()

                # Calculate percentiles
                if all_fees:
                    # Calculate requested percentile
                    index = int(len(all_fees) * percentile / 100)
                    fees[str(percentile)] = all_fees[min(index, len(all_fees) - 1)]

                    # Also include some standard percentiles
                    fees["50"] = all_fees[min(int(len(all_fees) * 0.5), len(all_fees) - 1)]  # Median
                    fees["90"] = all_fees[min(int(len(all_fees) * 0.9), len(all_fees) - 1)]  # 90th percentile
                    fees["max"] = all_fees[-1]  # Maximum
                    fees["min"] = all_fees[0]   # Minimum

            return fees
        except Exception as e:
            logger.error(f"Error getting recent priority fees: {e}")
            return {str(percentile): int(get_config_value("min_priority_fee", "1000"))}

    def send_transaction(self, transaction: Transaction, signer: Keypair,
                        priority_fee: Optional[int] = None, compute_limit: Optional[int] = None) -> str:
        """
        Send a transaction to the Solana blockchain with enhanced options.

        Args:
            transaction: The transaction to send
            signer: The keypair to sign the transaction with
            priority_fee: Optional priority fee in micro-lamports
            compute_limit: Optional compute unit limit

        Returns:
            The transaction signature
        """
        try:
            # Get recent blockhash
            blockhash = self.get_latest_blockhash()
            transaction.recent_blockhash = blockhash

            # Add compute budget instructions if needed
            if priority_fee is not None or compute_limit is not None:
                # Add compute budget instructions
                if compute_limit is not None:
                    # Add compute unit limit instruction
                    compute_limit_ix = set_compute_unit_limit(compute_limit)
                    transaction.instructions.insert(0, compute_limit_ix)

                if priority_fee is not None:
                    # Add priority fee instruction
                    priority_fee_ix = set_compute_unit_price(priority_fee)
                    transaction.instructions.insert(0 if compute_limit is None else 1, priority_fee_ix)

            # Sign transaction
            transaction.sign(signer)

            # Send transaction with RPC manager failover
            tx_opts = TxOpts(skip_preflight=False, preflight_commitment="confirmed")

            # Get the best RPC endpoint for transaction submission
            # For transactions, we prefer the endpoint with the lowest latency
            endpoint = self.rpc_manager.get_endpoint()

            try:
                # Try with the best endpoint first
                response = endpoint.client.send_transaction(transaction, signer, opts=tx_opts)

                # Record success
                endpoint.record_success(0)  # We don't have the actual response time here

                signature = response["result"]
                logger.info(f"Transaction sent via {endpoint.name}: {signature}")
                return signature
            except Exception as e:
                # Record error
                endpoint.record_error(e)
                logger.warning(f"Error sending transaction with RPC endpoint {endpoint.name}: {str(e)}")

                # Try failover with other endpoints
                for name, other_endpoint in self.rpc_manager.endpoints.items():
                    if name != endpoint.name and other_endpoint.available:
                        try:
                            logger.info(f"Trying failover to RPC endpoint {name} for transaction")
                            response = other_endpoint.client.send_transaction(transaction, signer, opts=tx_opts)

                            # Record success
                            other_endpoint.record_success(0)  # We don't have the actual response time here

                            # Switch current endpoint
                            self.rpc_manager.current_endpoint_name = name

                            signature = response["result"]
                            logger.info(f"Transaction sent via failover endpoint {name}: {signature}")
                            return signature
                        except Exception as e2:
                            other_endpoint.record_error(e2)
                            logger.warning(f"Failover to RPC endpoint {name} for transaction failed: {str(e2)}")

                # If all endpoints failed, raise the original exception
                logger.error("All RPC endpoints failed for transaction submission")
                raise e
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            raise

    def send_versioned_transaction(self, message: Message, signers: List[Keypair],
                                  priority_fee: Optional[int] = None,
                                  compute_limit: Optional[int] = None) -> str:
        """
        Send a versioned transaction to the Solana blockchain.

        Args:
            message: The transaction message
            signers: List of keypairs to sign the transaction with
            priority_fee: Optional priority fee in micro-lamports
            compute_limit: Optional compute unit limit

        Returns:
            The transaction signature
        """
        try:
            # Create versioned transaction
            tx = VersionedTransaction(message, [signer.sign_message(bytes(message)) for signer in signers])

            # Send transaction with RPC manager failover
            tx_opts = TxOpts(skip_preflight=False, preflight_commitment="confirmed")

            # Get the best RPC endpoint for transaction submission
            endpoint = self.rpc_manager.get_endpoint()

            try:
                # Try with the best endpoint first
                response = endpoint.client.send_transaction(tx, opts=tx_opts)

                # Record success
                endpoint.record_success(0)  # We don't have the actual response time here

                signature = response["result"]
                logger.info(f"Versioned transaction sent via {endpoint.name}: {signature}")
                return signature
            except Exception as e:
                # Record error
                endpoint.record_error(e)
                logger.warning(f"Error sending versioned transaction with RPC endpoint {endpoint.name}: {str(e)}")

                # Try failover with other endpoints
                for name, other_endpoint in self.rpc_manager.endpoints.items():
                    if name != endpoint.name and other_endpoint.available:
                        try:
                            logger.info(f"Trying failover to RPC endpoint {name} for versioned transaction")
                            response = other_endpoint.client.send_transaction(tx, opts=tx_opts)

                            # Record success
                            other_endpoint.record_success(0)  # We don't have the actual response time here

                            # Switch current endpoint
                            self.rpc_manager.current_endpoint_name = name

                            signature = response["result"]
                            logger.info(f"Versioned transaction sent via failover endpoint {name}: {signature}")
                            return signature
                        except Exception as e2:
                            other_endpoint.record_error(e2)
                            logger.warning(f"Failover to RPC endpoint {name} for versioned transaction failed: {str(e2)}")

                # If all endpoints failed, raise the original exception
                logger.error("All RPC endpoints failed for versioned transaction submission")
                raise e
        except Exception as e:
            logger.error(f"Error sending versioned transaction: {e}")
            raise


# Create a singleton instance
solana_client = SolanaClient()
