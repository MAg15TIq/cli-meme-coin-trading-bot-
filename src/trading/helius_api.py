"""
Helius API integration for the Solana Memecoin Trading Bot.
Handles wallet activity monitoring, token analytics, and webhook/WebSocket streams.
Includes enhanced token analytics and MEV protection.
"""

import json
import logging
import time
import threading
import asyncio
import websockets
from typing import Dict, Any, Optional, List, Union, Callable, Set, Tuple
from datetime import datetime, timedelta

import requests
from solders.pubkey import Pubkey

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class HeliusAPI:
    """Client for interacting with Helius API."""

    def __init__(self):
        """Initialize the Helius API client."""
        self.api_key = get_config_value("helius_api_key", "")
        self.api_url = f"https://api.helius.xyz/v0"
        self.webhook_url = get_config_value("helius_webhook_url", "")
        self.webhook_id = get_config_value("helius_webhook_id", "")
        self.websocket_url = f"wss://api.helius.xyz/v0/wallet-events?api-key={self.api_key}"

        # WebSocket connection
        self.ws_connection = None
        self.ws_thread = None
        self.ws_stop_event = threading.Event()

        # Tracked wallets and callbacks
        self.tracked_wallets: Set[str] = set()
        self.transaction_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Load tracked wallets from config
        self._load_tracked_wallets()

    def _load_tracked_wallets(self) -> None:
        """Load tracked wallets from config."""
        wallets = get_config_value("tracked_wallets", [])
        if wallets:
            self.tracked_wallets = set(wallets)
            logger.info(f"Loaded {len(self.tracked_wallets)} tracked wallets from config")

    def _save_tracked_wallets(self) -> None:
        """Save tracked wallets to config."""
        update_config("tracked_wallets", list(self.tracked_wallets))
        logger.info(f"Saved {len(self.tracked_wallets)} tracked wallets to config")

    def add_tracked_wallet(self, wallet_address: str) -> None:
        """
        Add a wallet to track.

        Args:
            wallet_address: The wallet address to track
        """
        self.tracked_wallets.add(wallet_address)
        self._save_tracked_wallets()
        logger.info(f"Added wallet to tracking: {wallet_address}")

        # Update webhook if configured
        if self.webhook_id and self.api_key:
            self._update_webhook_addresses()

    def remove_tracked_wallet(self, wallet_address: str) -> bool:
        """
        Remove a wallet from tracking.

        Args:
            wallet_address: The wallet address to stop tracking

        Returns:
            True if the wallet was removed, False if it wasn't tracked
        """
        if wallet_address in self.tracked_wallets:
            self.tracked_wallets.remove(wallet_address)
            self._save_tracked_wallets()
            logger.info(f"Removed wallet from tracking: {wallet_address}")

            # Update webhook if configured
            if self.webhook_id and self.api_key:
                self._update_webhook_addresses()

            return True
        return False

    def get_tracked_wallets(self) -> List[str]:
        """
        Get all tracked wallets.

        Returns:
            List of tracked wallet addresses
        """
        return list(self.tracked_wallets)

    def register_transaction_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback for transaction events.

        Args:
            callback: Function to call with transaction data
        """
        self.transaction_callbacks.append(callback)
        logger.info(f"Registered transaction callback: {callback.__name__}")

    def _update_webhook_addresses(self) -> bool:
        """
        Update the webhook with the current tracked wallets.

        Returns:
            True if successful, False otherwise
        """
        if not self.api_key or not self.webhook_id:
            logger.warning("Cannot update webhook: Missing API key or webhook ID")
            return False

        try:
            url = f"{self.api_url}/webhooks/{self.webhook_id}?api-key={self.api_key}"
            payload = {
                "webhookURL": self.webhook_url,
                "accountAddresses": list(self.tracked_wallets),
                "transactionTypes": ["ANY"],
                "webhookType": "enhanced"
            }

            response = requests.put(url, json=payload)
            response.raise_for_status()

            logger.info(f"Updated webhook with {len(self.tracked_wallets)} tracked wallets")
            return True
        except Exception as e:
            logger.error(f"Error updating webhook: {e}")
            return False

    def create_webhook(self, webhook_url: str) -> Optional[str]:
        """
        Create a new webhook for tracking wallet activities.

        Args:
            webhook_url: The URL to send webhook events to

        Returns:
            The webhook ID if successful, None otherwise
        """
        if not self.api_key:
            logger.warning("Cannot create webhook: Missing API key")
            return None

        try:
            url = f"{self.api_url}/webhooks?api-key={self.api_key}"
            payload = {
                "webhookURL": webhook_url,
                "accountAddresses": list(self.tracked_wallets),
                "transactionTypes": ["ANY"],
                "webhookType": "enhanced"
            }

            response = requests.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            webhook_id = data.get("webhookID")

            if webhook_id:
                # Save webhook info to config
                update_config("helius_webhook_url", webhook_url)
                update_config("helius_webhook_id", webhook_id)
                self.webhook_url = webhook_url
                self.webhook_id = webhook_id

                logger.info(f"Created webhook with ID: {webhook_id}")
                return webhook_id
            else:
                logger.error("No webhook ID in response")
                return None
        except Exception as e:
            logger.error(f"Error creating webhook: {e}")
            return None

    def delete_webhook(self, webhook_id: Optional[str] = None) -> bool:
        """
        Delete a webhook.

        Args:
            webhook_id: The webhook ID to delete (uses stored ID if None)

        Returns:
            True if successful, False otherwise
        """
        webhook_id = webhook_id or self.webhook_id

        if not self.api_key or not webhook_id:
            logger.warning("Cannot delete webhook: Missing API key or webhook ID")
            return False

        try:
            url = f"{self.api_url}/webhooks/{webhook_id}?api-key={self.api_key}"
            response = requests.delete(url)
            response.raise_for_status()

            # Clear webhook info from config if it's the stored webhook
            if webhook_id == self.webhook_id:
                update_config("helius_webhook_id", "")
                self.webhook_id = ""

            logger.info(f"Deleted webhook with ID: {webhook_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting webhook: {e}")
            return False

    def handle_webhook_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a webhook event.

        Args:
            event_data: The webhook event data
        """
        try:
            # Process the event
            logger.debug(f"Received webhook event: {json.dumps(event_data)[:200]}...")

            # Call registered callbacks
            for callback in self.transaction_callbacks:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"Error in transaction callback {callback.__name__}: {e}")
        except Exception as e:
            logger.error(f"Error handling webhook event: {e}")

    async def _websocket_listener(self) -> None:
        """WebSocket listener for wallet events."""
        if not self.api_key:
            logger.error("Cannot start WebSocket: Missing API key")
            return

        while not self.ws_stop_event.is_set():
            try:
                # Connect to WebSocket
                addresses_param = ",".join(self.tracked_wallets)
                ws_url = f"{self.websocket_url}&addresses={addresses_param}"

                logger.info(f"Connecting to Helius WebSocket with {len(self.tracked_wallets)} tracked wallets")

                async with websockets.connect(ws_url) as websocket:
                    self.ws_connection = websocket
                    logger.info("Connected to Helius WebSocket")

                    # Listen for messages
                    while not self.ws_stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=30)
                            event_data = json.loads(message)

                            # Process the event in a separate thread to avoid blocking the WebSocket
                            threading.Thread(
                                target=self.handle_webhook_event,
                                args=(event_data,),
                                daemon=True
                            ).start()
                        except asyncio.TimeoutError:
                            # Send a ping to keep the connection alive
                            try:
                                pong_waiter = await websocket.ping()
                                await asyncio.wait_for(pong_waiter, timeout=10)
                            except asyncio.TimeoutError:
                                logger.warning("WebSocket ping timeout, reconnecting...")
                                break
                        except Exception as e:
                            logger.error(f"Error receiving WebSocket message: {e}")
                            break

            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")

            # Don't reconnect immediately if stopping
            if self.ws_stop_event.is_set():
                break

            # Wait before reconnecting
            await asyncio.sleep(5)

    def start_websocket(self) -> bool:
        """
        Start the WebSocket listener for wallet events.

        Returns:
            True if started successfully, False otherwise
        """
        if not self.tracked_wallets:
            logger.warning("No wallets to track, not starting WebSocket")
            return False

        if self.ws_thread and self.ws_thread.is_alive():
            logger.info("WebSocket already running")
            return True

        # Clear stop event
        self.ws_stop_event.clear()

        # Start WebSocket in a separate thread
        def run_websocket_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._websocket_listener())

        self.ws_thread = threading.Thread(target=run_websocket_loop, daemon=True)
        self.ws_thread.start()

        logger.info("Started WebSocket listener thread")
        return True

    def stop_websocket(self) -> None:
        """Stop the WebSocket listener."""
        if not self.ws_thread or not self.ws_thread.is_alive():
            logger.info("WebSocket not running")
            return

        # Set stop event
        self.ws_stop_event.set()

        # Wait for thread to stop
        self.ws_thread.join(timeout=5)

        logger.info("Stopped WebSocket listener")

    def get_wallet_transactions(self, wallet_address: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent transactions for a wallet.

        Args:
            wallet_address: The wallet address to get transactions for
            limit: Maximum number of transactions to return

        Returns:
            List of transaction data
        """
        if not self.api_key:
            logger.warning("Cannot get transactions: Missing API key")
            return []

        try:
            url = f"{self.api_url}/addresses/{wallet_address}/transactions?api-key={self.api_key}&limit={limit}"
            response = requests.get(url)
            response.raise_for_status()

            transactions = response.json()
            logger.info(f"Got {len(transactions)} transactions for wallet {wallet_address}")
            return transactions
        except Exception as e:
            logger.error(f"Error getting transactions for wallet {wallet_address}: {e}")
            return []

    def get_wallet_balances(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get token balances for a wallet.

        Args:
            wallet_address: The wallet address to get balances for

        Returns:
            Dictionary of token balances
        """
        if not self.api_key:
            logger.warning("Cannot get balances: Missing API key")
            return {}

        try:
            url = f"{self.api_url}/addresses/{wallet_address}/balances?api-key={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()

            balances = response.json()
            logger.info(f"Got balances for wallet {wallet_address}")
            return balances
        except Exception as e:
            logger.error(f"Error getting balances for wallet {wallet_address}: {e}")
            return {}

    def get_token_metadata(self, token_address: str) -> Dict[str, Any]:
        """
        Get metadata for a token.

        Args:
            token_address: The token mint address

        Returns:
            Token metadata
        """
        if not self.api_key:
            logger.warning("Cannot get token metadata: Missing API key")
            return {}

        try:
            url = f"{self.api_url}/tokens/metadata?api-key={self.api_key}"
            payload = {"mintAccounts": [token_address]}

            response = requests.post(url, json=payload)
            response.raise_for_status()

            metadata = response.json()
            if metadata and len(metadata) > 0:
                logger.info(f"Got metadata for token {token_address}")
                return metadata[0]
            else:
                logger.warning(f"No metadata found for token {token_address}")
                return {}
        except Exception as e:
            logger.error(f"Error getting metadata for token {token_address}: {e}")
            return {}


    def get_token_holders(self, token_address: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get top holders for a token.

        Args:
            token_address: The token mint address
            limit: Maximum number of holders to return

        Returns:
            List of token holders with their balances
        """
        if not self.api_key:
            logger.warning("Cannot get token holders: Missing API key")
            return []

        try:
            url = f"{self.api_url}/tokens/{token_address}/holders?api-key={self.api_key}&limit={limit}"
            response = requests.get(url)
            response.raise_for_status()

            holders = response.json()
            logger.info(f"Got {len(holders)} holders for token {token_address}")
            return holders
        except Exception as e:
            logger.error(f"Error getting holders for token {token_address}: {e}")
            return []

    def get_token_supply(self, token_address: str) -> Dict[str, Any]:
        """
        Get supply information for a token.

        Args:
            token_address: The token mint address

        Returns:
            Token supply information
        """
        if not self.api_key:
            logger.warning("Cannot get token supply: Missing API key")
            return {}

        try:
            url = f"{self.api_url}/tokens/{token_address}/supply?api-key={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()

            supply_info = response.json()
            logger.info(f"Got supply info for token {token_address}")
            return supply_info
        except Exception as e:
            logger.error(f"Error getting supply for token {token_address}: {e}")
            return {}

    def get_token_transactions(self, token_address: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent transactions for a token.

        Args:
            token_address: The token mint address
            limit: Maximum number of transactions to return

        Returns:
            List of token transactions
        """
        if not self.api_key:
            logger.warning("Cannot get token transactions: Missing API key")
            return []

        try:
            url = f"{self.api_url}/tokens/{token_address}/transactions?api-key={self.api_key}&limit={limit}"
            response = requests.get(url)
            response.raise_for_status()

            transactions = response.json()
            logger.info(f"Got {len(transactions)} transactions for token {token_address}")
            return transactions
        except Exception as e:
            logger.error(f"Error getting transactions for token {token_address}: {e}")
            return []

    def get_token_liquidity(self, token_address: str) -> Dict[str, Any]:
        """
        Get liquidity information for a token.

        Args:
            token_address: The token mint address

        Returns:
            Token liquidity information
        """
        if not self.api_key:
            logger.warning("Cannot get token liquidity: Missing API key")
            return {}

        try:
            url = f"{self.api_url}/tokens/{token_address}/liquidity?api-key={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()

            liquidity_info = response.json()
            logger.info(f"Got liquidity info for token {token_address}")
            return liquidity_info
        except Exception as e:
            logger.error(f"Error getting liquidity for token {token_address}: {e}")
            return {}

    def analyze_token_contract(self, token_address: str) -> Dict[str, Any]:
        """
        Analyze a token's contract for potential issues.

        Args:
            token_address: The token mint address

        Returns:
            Analysis results
        """
        if not self.api_key:
            logger.warning("Cannot analyze token contract: Missing API key")
            return {}

        try:
            # Get token metadata
            metadata = self.get_token_metadata(token_address)

            # Get token holders
            holders = self.get_token_holders(token_address)

            # Get token supply
            supply_info = self.get_token_supply(token_address)

            # Analyze concentration
            concentration = self._analyze_holder_concentration(holders, supply_info)

            # Analyze contract
            contract_analysis = self._analyze_contract_features(token_address, metadata)

            # Combine results
            analysis = {
                "token_address": token_address,
                "name": metadata.get("name", "Unknown"),
                "symbol": metadata.get("symbol", "UNK"),
                "concentration": concentration,
                "contract": contract_analysis,
                "risk_score": self._calculate_risk_score(concentration, contract_analysis),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Analyzed token contract for {token_address}")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing token contract for {token_address}: {e}")
            return {}

    def _analyze_holder_concentration(self, holders: List[Dict[str, Any]],
                                     supply_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze holder concentration for a token.

        Args:
            holders: List of token holders
            supply_info: Token supply information

        Returns:
            Concentration analysis
        """
        try:
            # Calculate total supply
            total_supply = supply_info.get("total", 0)
            if total_supply == 0 and holders:
                # Estimate from holders
                total_supply = sum(holder.get("amount", 0) for holder in holders)

            if total_supply == 0:
                return {
                    "top_holder_percent": 0,
                    "top_5_percent": 0,
                    "top_10_percent": 0,
                    "holder_count": len(holders),
                    "is_concentrated": False
                }

            # Calculate concentration metrics
            top_holder_percent = (holders[0].get("amount", 0) / total_supply * 100) if holders else 0
            top_5_percent = (sum(holder.get("amount", 0) for holder in holders[:5]) / total_supply * 100) if len(holders) >= 5 else top_holder_percent
            top_10_percent = (sum(holder.get("amount", 0) for holder in holders[:10]) / total_supply * 100) if len(holders) >= 10 else top_5_percent

            # Determine if concentrated
            is_concentrated = top_holder_percent > 50 or top_5_percent > 80

            return {
                "top_holder_percent": top_holder_percent,
                "top_5_percent": top_5_percent,
                "top_10_percent": top_10_percent,
                "holder_count": len(holders),
                "is_concentrated": is_concentrated
            }
        except Exception as e:
            logger.error(f"Error analyzing holder concentration: {e}")
            return {
                "top_holder_percent": 0,
                "top_5_percent": 0,
                "top_10_percent": 0,
                "holder_count": 0,
                "is_concentrated": False,
                "error": str(e)
            }

    def _analyze_contract_features(self, token_address: str,
                                  metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze contract features for a token.

        Args:
            token_address: The token mint address
            metadata: Token metadata

        Returns:
            Contract analysis
        """
        try:
            # Check if token is verified
            is_verified = metadata.get("verified", False)

            # Check for mint authority
            has_mint_authority = metadata.get("mintAuthority") is not None

            # Check for freeze authority
            has_freeze_authority = metadata.get("freezeAuthority") is not None

            # Check for suspicious features
            suspicious_features = []

            if has_mint_authority:
                suspicious_features.append("mint_authority")

            if has_freeze_authority:
                suspicious_features.append("freeze_authority")

            # Determine if potentially malicious
            is_potentially_malicious = len(suspicious_features) > 0 and not is_verified

            return {
                "is_verified": is_verified,
                "has_mint_authority": has_mint_authority,
                "has_freeze_authority": has_freeze_authority,
                "suspicious_features": suspicious_features,
                "is_potentially_malicious": is_potentially_malicious
            }
        except Exception as e:
            logger.error(f"Error analyzing contract features: {e}")
            return {
                "is_verified": False,
                "has_mint_authority": False,
                "has_freeze_authority": False,
                "suspicious_features": [],
                "is_potentially_malicious": False,
                "error": str(e)
            }

    def _calculate_risk_score(self, concentration: Dict[str, Any],
                             contract: Dict[str, Any]) -> int:
        """
        Calculate a risk score for a token.

        Args:
            concentration: Concentration analysis
            contract: Contract analysis

        Returns:
            Risk score (0-100, higher is riskier)
        """
        try:
            risk_score = 0

            # Concentration risk (0-50 points)
            if concentration.get("is_concentrated", False):
                risk_score += 30

            if concentration.get("top_holder_percent", 0) > 80:
                risk_score += 20
            elif concentration.get("top_holder_percent", 0) > 50:
                risk_score += 10

            if concentration.get("top_5_percent", 0) > 90:
                risk_score += 20
            elif concentration.get("top_5_percent", 0) > 70:
                risk_score += 10

            if concentration.get("holder_count", 0) < 10:
                risk_score += 10

            # Contract risk (0-50 points)
            if contract.get("is_potentially_malicious", False):
                risk_score += 30

            if not contract.get("is_verified", False):
                risk_score += 20

            if contract.get("has_mint_authority", False):
                risk_score += 15

            if contract.get("has_freeze_authority", False):
                risk_score += 15

            # Cap at 100
            return min(risk_score, 100)
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50  # Default to medium risk on error

    def get_mev_bundle_price(self) -> Dict[str, Any]:
        """
        Get current MEV bundle pricing.

        Returns:
            MEV bundle pricing information
        """
        if not self.api_key:
            logger.warning("Cannot get MEV bundle price: Missing API key")
            return {}

        try:
            url = f"{self.api_url}/mev/bundle-price?api-key={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()

            price_info = response.json()
            logger.info(f"Got MEV bundle price info")
            return price_info
        except Exception as e:
            logger.error(f"Error getting MEV bundle price: {e}")
            return {}

    def submit_mev_bundle(self, transactions: List[str], tip_lamports: int) -> Dict[str, Any]:
        """
        Submit a bundle of transactions with MEV protection.

        Args:
            transactions: List of serialized transactions
            tip_lamports: Tip amount in lamports

        Returns:
            Bundle submission result
        """
        if not self.api_key:
            logger.warning("Cannot submit MEV bundle: Missing API key")
            return {}

        try:
            url = f"{self.api_url}/mev/bundle?api-key={self.api_key}"
            payload = {
                "transactions": transactions,
                "tip": tip_lamports
            }

            response = requests.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            logger.info(f"Submitted MEV bundle with {len(transactions)} transactions")
            return result
        except Exception as e:
            logger.error(f"Error submitting MEV bundle: {e}")
            return {}


# Create a singleton instance
helius_api = HeliusAPI()
