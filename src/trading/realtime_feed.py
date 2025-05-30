"""
Real-time WebSocket data feed for high-frequency trading.
Provides low-latency price updates and market data.
"""

import asyncio
import websockets
import json
import time
import threading
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from config import get_config_value
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PriceUpdate:
    """Represents a real-time price update."""
    token_mint: str
    price: float
    timestamp: float
    volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None


class RealtimeFeed:
    """Real-time WebSocket data feed for market data."""
    
    def __init__(self):
        """Initialize the real-time feed."""
        self.ws_url = get_config_value("rpc_ws_url", "wss://api.mainnet-beta.solana.com")
        self.jupiter_ws_url = "wss://price-api.jup.ag/v1/ws"
        
        # Connection management
        self.websocket = None
        self.jupiter_ws = None
        self.connected = False
        self.running = False
        
        # Subscriptions and callbacks
        self.price_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.general_callbacks: List[Callable] = []
        
        # Data storage
        self.latest_prices: Dict[str, PriceUpdate] = {}
        self.price_history: Dict[str, List[PriceUpdate]] = defaultdict(list)
        
        # Performance tracking
        self.message_count = 0
        self.last_message_time = 0
        self.latency_samples = []
        
        # Threading
        self.event_loop = None
        self.thread = None
        
        logger.info("Real-time feed initialized")
    
    def start(self):
        """Start the real-time feed."""
        if self.running:
            logger.warning("Real-time feed is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        logger.info("Real-time feed started")
    
    def stop(self):
        """Stop the real-time feed."""
        self.running = False
        if self.event_loop:
            asyncio.run_coroutine_threadsafe(self._stop_connections(), self.event_loop)
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Real-time feed stopped")
    
    def subscribe_to_price(self, token_mint: str, callback: Callable[[PriceUpdate], None]):
        """
        Subscribe to price updates for a specific token.
        
        Args:
            token_mint: Token mint address to monitor
            callback: Function to call when price updates
        """
        self.price_callbacks[token_mint].append(callback)
        logger.info(f"Subscribed to price updates for {token_mint}")
    
    def unsubscribe_from_price(self, token_mint: str, callback: Callable[[PriceUpdate], None]):
        """
        Unsubscribe from price updates.
        
        Args:
            token_mint: Token mint address
            callback: Callback function to remove
        """
        if token_mint in self.price_callbacks:
            try:
                self.price_callbacks[token_mint].remove(callback)
                logger.info(f"Unsubscribed from price updates for {token_mint}")
            except ValueError:
                logger.warning(f"Callback not found for {token_mint}")
    
    def subscribe_to_general_updates(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to general market updates.
        
        Args:
            callback: Function to call for general updates
        """
        self.general_callbacks.append(callback)
        logger.info("Subscribed to general market updates")
    
    def get_latest_price(self, token_mint: str) -> Optional[PriceUpdate]:
        """
        Get the latest price for a token.
        
        Args:
            token_mint: Token mint address
            
        Returns:
            Latest price update or None if not available
        """
        return self.latest_prices.get(token_mint)
    
    def get_price_history(self, token_mint: str, limit: int = 100) -> List[PriceUpdate]:
        """
        Get price history for a token.
        
        Args:
            token_mint: Token mint address
            limit: Maximum number of historical prices to return
            
        Returns:
            List of price updates
        """
        history = self.price_history.get(token_mint, [])
        return history[-limit:] if limit > 0 else history
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the feed."""
        avg_latency = sum(self.latency_samples) / len(self.latency_samples) if self.latency_samples else 0
        
        return {
            "connected": self.connected,
            "message_count": self.message_count,
            "average_latency_ms": avg_latency * 1000,
            "subscribed_tokens": len(self.price_callbacks),
            "cached_prices": len(self.latest_prices)
        }
    
    def _run_event_loop(self):
        """Run the asyncio event loop in a separate thread."""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        
        try:
            self.event_loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            logger.error(f"Error in real-time feed event loop: {e}")
        finally:
            self.event_loop.close()
    
    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages."""
        while self.running:
            try:
                # Connect to Jupiter price feed
                async with websockets.connect(self.jupiter_ws_url) as websocket:
                    self.jupiter_ws = websocket
                    self.connected = True
                    logger.info("Connected to Jupiter WebSocket feed")
                    
                    # Subscribe to price updates
                    subscribe_msg = {
                        "method": "subscribeToPrice",
                        "params": {
                            "tokens": list(self.price_callbacks.keys()) if self.price_callbacks else ["So11111111111111111111111111111111111111112"]
                        }
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    # Listen for messages
                    async for message in websocket:
                        if not self.running:
                            break
                        await self._handle_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, attempting to reconnect...")
                self.connected = False
                await asyncio.sleep(5)  # Wait before reconnecting
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connected = False
                await asyncio.sleep(5)
    
    async def _handle_message(self, message: str):
        """
        Handle incoming WebSocket message.
        
        Args:
            message: Raw WebSocket message
        """
        try:
            data = json.loads(message)
            message_time = time.time()
            
            # Calculate latency if timestamp is available
            if "timestamp" in data:
                latency = message_time - (data["timestamp"] / 1000)
                self.latency_samples.append(latency)
                if len(self.latency_samples) > 100:
                    self.latency_samples = self.latency_samples[-100:]
            
            self.message_count += 1
            self.last_message_time = message_time
            
            # Handle different message types
            if data.get("type") == "price":
                await self._handle_price_update(data)
            elif data.get("method") == "priceUpdate":
                await self._handle_jupiter_price_update(data)
            else:
                # General market update
                for callback in self.general_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in general callback: {e}")
                        
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_price_update(self, data: Dict[str, Any]):
        """Handle price update message."""
        try:
            token_mint = data.get("mint")
            price = float(data.get("price", 0))
            
            if not token_mint or price <= 0:
                return
            
            # Create price update
            price_update = PriceUpdate(
                token_mint=token_mint,
                price=price,
                timestamp=time.time(),
                volume_24h=data.get("volume24h"),
                price_change_24h=data.get("priceChange24h")
            )
            
            # Store latest price
            self.latest_prices[token_mint] = price_update
            
            # Add to history
            self.price_history[token_mint].append(price_update)
            if len(self.price_history[token_mint]) > 1000:
                self.price_history[token_mint] = self.price_history[token_mint][-1000:]
            
            # Call callbacks
            for callback in self.price_callbacks[token_mint]:
                try:
                    callback(price_update)
                except Exception as e:
                    logger.error(f"Error in price callback for {token_mint}: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling price update: {e}")
    
    async def _handle_jupiter_price_update(self, data: Dict[str, Any]):
        """Handle Jupiter-specific price update."""
        try:
            params = data.get("params", {})
            for token_data in params.get("data", []):
                token_mint = token_data.get("id")
                price = float(token_data.get("price", 0))
                
                if not token_mint or price <= 0:
                    continue
                
                price_update = PriceUpdate(
                    token_mint=token_mint,
                    price=price,
                    timestamp=time.time(),
                    volume_24h=token_data.get("volume24h"),
                    price_change_24h=token_data.get("priceChange24h")
                )
                
                # Store and notify
                self.latest_prices[token_mint] = price_update
                self.price_history[token_mint].append(price_update)
                
                # Limit history size
                if len(self.price_history[token_mint]) > 1000:
                    self.price_history[token_mint] = self.price_history[token_mint][-1000:]
                
                # Call callbacks
                for callback in self.price_callbacks[token_mint]:
                    try:
                        callback(price_update)
                    except Exception as e:
                        logger.error(f"Error in Jupiter price callback for {token_mint}: {e}")
                        
        except Exception as e:
            logger.error(f"Error handling Jupiter price update: {e}")
    
    async def _stop_connections(self):
        """Stop all WebSocket connections."""
        if self.jupiter_ws:
            await self.jupiter_ws.close()
        if self.websocket:
            await self.websocket.close()
        self.connected = False


# Create singleton instance
realtime_feed = RealtimeFeed()
