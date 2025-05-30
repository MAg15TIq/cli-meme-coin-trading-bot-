"""
Live Trading Engine - Phase 4A Implementation
Real-time market data integration and live order execution
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import websockets
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty

from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class MarketData:
    """Real-time market data structure"""
    symbol: str
    price: float
    volume_24h: float
    liquidity: float
    timestamp: datetime
    source: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class LiveOrder:
    """Live order structure"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: Optional[float] = None  # None for market orders
    order_type: str = 'market'  # 'market', 'limit', 'stop'
    status: str = 'pending'  # 'pending', 'filled', 'cancelled', 'failed'
    timestamp: datetime = None
    filled_amount: float = 0.0
    average_price: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class DataFeedManager:
    """Manages real-time data feeds from multiple sources"""

    def __init__(self):
        self.feeds = {}
        self.subscribers = {}
        self.running = False
        self.data_queue = Queue()
        self.validation_enabled = True

        # Data source configurations
        self.sources = {
            'jupiter': {
                'ws_url': 'wss://price.jup.ag/v4/price-stream',
                'rest_url': 'https://price.jup.ag/v4/price',
                'enabled': True
            },
            'raydium': {
                'ws_url': 'wss://api.raydium.io/v2/ws',
                'rest_url': 'https://api.raydium.io/v2/main/price',
                'enabled': True
            },
            'orca': {
                'ws_url': 'wss://api.orca.so/v1/ws',
                'rest_url': 'https://api.orca.so/v1/whirlpool/list',
                'enabled': True
            },
            'dexscreener': {
                'rest_url': 'https://api.dexscreener.com/latest/dex/tokens',
                'enabled': True
            }
        }

    async def start(self):
        """Start all data feeds"""
        self.running = True
        logger.info("Starting live data feeds...")

        # Start WebSocket connections
        tasks = []
        for source, config in self.sources.items():
            if config['enabled'] and 'ws_url' in config:
                task = asyncio.create_task(self._connect_websocket(source, config['ws_url']))
                tasks.append(task)

        # Start REST API polling for sources without WebSocket
        for source, config in self.sources.items():
            if config['enabled'] and 'ws_url' not in config:
                task = asyncio.create_task(self._poll_rest_api(source, config['rest_url']))
                tasks.append(task)

        # Start data processing
        processing_task = asyncio.create_task(self._process_data())
        tasks.append(processing_task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        """Stop all data feeds"""
        self.running = False
        logger.info("Stopping live data feeds...")

    async def _connect_websocket(self, source: str, ws_url: str):
        """Connect to WebSocket data feed"""
        while self.running:
            try:
                async with websockets.connect(ws_url) as websocket:
                    logger.info(f"Connected to {source} WebSocket")

                    # Send subscription message based on source
                    if source == 'jupiter':
                        await websocket.send(json.dumps({
                            "method": "subscribeToPrice",
                            "params": ["all"]
                        }))
                    elif source == 'raydium':
                        await websocket.send(json.dumps({
                            "method": "subscribe",
                            "params": ["price_updates"]
                        }))

                    async for message in websocket:
                        if not self.running:
                            break

                        try:
                            data = json.loads(message)
                            market_data = self._parse_websocket_data(source, data)
                            if market_data:
                                self.data_queue.put(market_data)
                        except Exception as e:
                            logger.error(f"Error parsing {source} data: {e}")

            except Exception as e:
                logger.error(f"WebSocket connection error for {source}: {e}")
                if self.running:
                    await asyncio.sleep(5)  # Reconnect delay

    async def _poll_rest_api(self, source: str, api_url: str):
        """Poll REST API for data"""
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(api_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            market_data_list = self._parse_rest_data(source, data)
                            for market_data in market_data_list:
                                self.data_queue.put(market_data)
                        else:
                            logger.warning(f"API error for {source}: {response.status}")

            except Exception as e:
                logger.error(f"REST API error for {source}: {e}")

            await asyncio.sleep(1)  # Poll every second

    def _parse_websocket_data(self, source: str, data: Dict) -> Optional[MarketData]:
        """Parse WebSocket data into MarketData format"""
        try:
            if source == 'jupiter':
                if 'data' in data and 'price' in data['data']:
                    return MarketData(
                        symbol=data['data'].get('id', 'UNKNOWN'),
                        price=float(data['data']['price']),
                        volume_24h=float(data['data'].get('volume24h', 0)),
                        liquidity=float(data['data'].get('liquidity', 0)),
                        timestamp=datetime.now(),
                        source=source
                    )
            elif source == 'raydium':
                if 'price' in data:
                    return MarketData(
                        symbol=data.get('symbol', 'UNKNOWN'),
                        price=float(data['price']),
                        volume_24h=float(data.get('volume', 0)),
                        liquidity=float(data.get('liquidity', 0)),
                        timestamp=datetime.now(),
                        source=source
                    )
        except Exception as e:
            logger.error(f"Error parsing {source} WebSocket data: {e}")

        return None

    def _parse_rest_data(self, source: str, data: Dict) -> List[MarketData]:
        """Parse REST API data into MarketData format"""
        market_data_list = []

        try:
            if source == 'dexscreener':
                if 'pairs' in data:
                    for pair in data['pairs']:
                        market_data = MarketData(
                            symbol=pair.get('baseToken', {}).get('symbol', 'UNKNOWN'),
                            price=float(pair.get('priceUsd', 0)),
                            volume_24h=float(pair.get('volume', {}).get('h24', 0)),
                            liquidity=float(pair.get('liquidity', {}).get('usd', 0)),
                            timestamp=datetime.now(),
                            source=source
                        )
                        market_data_list.append(market_data)
        except Exception as e:
            logger.error(f"Error parsing {source} REST data: {e}")

        return market_data_list

    async def _process_data(self):
        """Process incoming market data"""
        while self.running:
            try:
                # Process data from queue
                while not self.data_queue.empty():
                    try:
                        market_data = self.data_queue.get_nowait()

                        # Validate data if enabled
                        if self.validation_enabled:
                            if not self._validate_data(market_data):
                                continue

                        # Notify subscribers
                        await self._notify_subscribers(market_data)

                    except Empty:
                        break
                    except Exception as e:
                        logger.error(f"Error processing market data: {e}")

                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error in data processing loop: {e}")

    def _validate_data(self, market_data: MarketData) -> bool:
        """Validate market data quality"""
        try:
            # Basic validation checks
            if market_data.price <= 0:
                return False

            if market_data.volume_24h < 0:
                return False

            if market_data.liquidity < 0:
                return False

            # Check for reasonable price ranges (basic sanity check)
            if market_data.price > 1000000:  # Very high price
                logger.warning(f"Unusually high price for {market_data.symbol}: {market_data.price}")

            return True

        except Exception as e:
            logger.error(f"Error validating market data: {e}")
            return False

    async def _notify_subscribers(self, market_data: MarketData):
        """Notify all subscribers of new market data"""
        symbol = market_data.symbol
        if symbol in self.subscribers:
            for callback in self.subscribers[symbol]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(market_data)
                    else:
                        callback(market_data)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")

    def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to market data for a specific symbol"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
        logger.info(f"Subscribed to {symbol} market data")

    def unsubscribe(self, symbol: str, callback: Callable):
        """Unsubscribe from market data for a specific symbol"""
        if symbol in self.subscribers and callback in self.subscribers[symbol]:
            self.subscribers[symbol].remove(callback)
            if not self.subscribers[symbol]:
                del self.subscribers[symbol]
            logger.info(f"Unsubscribed from {symbol} market data")

class LiveTradingEngine:
    """Main live trading engine"""

    def __init__(self, config: Dict):
        self.config = config
        self.data_feed = DataFeedManager()
        self.orders = {}
        self.positions = {}
        self.running = False
        self.paper_trading = config.get('paper_trading_mode', True)
        self.max_daily_loss = config.get('max_daily_loss_limit', 1000.0)
        self.emergency_stop = config.get('emergency_stop_enabled', True)
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()

        # Performance tracking
        self.execution_times = []
        self.latency_target = config.get('execution_latency_target_ms', 100)

        logger.info(f"Live Trading Engine initialized (Paper Trading: {self.paper_trading})")

    async def start(self):
        """Start the live trading engine"""
        self.running = True
        logger.info("Starting Live Trading Engine...")

        # Start data feeds
        await self.data_feed.start()

        # Start monitoring tasks
        monitoring_task = asyncio.create_task(self._monitor_positions())
        risk_task = asyncio.create_task(self._monitor_risk())

        await asyncio.gather(monitoring_task, risk_task, return_exceptions=True)

    async def stop(self):
        """Stop the live trading engine"""
        self.running = False
        await self.data_feed.stop()
        logger.info("Live Trading Engine stopped")

    async def place_order(self, order: LiveOrder) -> str:
        """Place a live order"""
        start_time = time.time()

        try:
            # Check emergency stop
            if self.emergency_stop and self._check_emergency_conditions():
                raise Exception("Emergency stop activated")

            # Validate order
            if not self._validate_order(order):
                raise Exception("Order validation failed")

            # Execute order
            if self.paper_trading:
                result = await self._execute_paper_order(order)
            else:
                result = await self._execute_live_order(order)

            # Track execution time
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            self.execution_times.append(execution_time)

            if execution_time > self.latency_target:
                logger.warning(f"Order execution exceeded latency target: {execution_time:.2f}ms")

            self.orders[order.order_id] = order
            logger.info(f"Order placed: {order.order_id} ({execution_time:.2f}ms)")

            return order.order_id

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            order.status = 'failed'
            raise

    def _validate_order(self, order: LiveOrder) -> bool:
        """Validate order parameters"""
        try:
            # Basic validation
            if order.amount <= 0:
                logger.error("Order amount must be positive")
                return False

            if order.side not in ['buy', 'sell']:
                logger.error("Order side must be 'buy' or 'sell'")
                return False

            if order.order_type not in ['market', 'limit', 'stop']:
                logger.error("Invalid order type")
                return False

            # Check daily loss limit
            if self._check_daily_loss_limit(order):
                logger.error("Order would exceed daily loss limit")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False

    async def _execute_paper_order(self, order: LiveOrder) -> Dict:
        """Execute order in paper trading mode"""
        # Simulate order execution with current market price
        # This is a simplified implementation
        order.status = 'filled'
        order.filled_amount = order.amount
        order.average_price = 100.0  # Mock price - would get from market data

        return {"status": "success", "order_id": order.order_id}

    async def _execute_live_order(self, order: LiveOrder) -> Dict:
        """Execute order on live market"""
        try:
            from src.trading.jupiter_api import jupiter_api
            from src.wallet.wallet import wallet_manager

            # Get current wallet
            wallet = wallet_manager.get_current_keypair()
            if not wallet:
                raise Exception("No wallet connected for live trading")

            # Execute the order based on type
            if order.order_type.lower() == 'buy':
                # Execute buy order
                tx_signature = jupiter_api.execute_buy(
                    token_mint=order.token_mint,
                    amount_sol=order.amount,
                    wallet=wallet
                )

                # Get actual execution price
                current_price = jupiter_api.get_token_price(order.token_mint)

                order.status = 'filled'
                order.filled_amount = order.amount
                order.average_price = current_price
                order.transaction_signature = tx_signature

                logger.info(f"Live buy order executed: {order.amount} SOL for {order.token_mint}, tx: {tx_signature}")

            elif order.order_type.lower() == 'sell':
                # For sell orders, we need to get position info
                from src.trading.position_manager import position_manager
                position = position_manager.get_position(order.token_mint)

                if not position:
                    raise Exception(f"No position found for token {order.token_mint}")

                # Execute sell order
                tx_signature = jupiter_api.execute_sell(
                    token_mint=order.token_mint,
                    amount_token=order.amount,
                    decimals=position.decimals,
                    wallet=wallet
                )

                # Get actual execution price
                current_price = jupiter_api.get_token_price(order.token_mint)

                order.status = 'filled'
                order.filled_amount = order.amount
                order.average_price = current_price
                order.transaction_signature = tx_signature

                logger.info(f"Live sell order executed: {order.amount} tokens of {order.token_mint}, tx: {tx_signature}")

            else:
                raise Exception(f"Unsupported order type: {order.order_type}")

            # Update daily P&L tracking
            if hasattr(order, 'estimated_pnl'):
                self.daily_pnl += order.estimated_pnl

            return {
                "status": "success",
                "order_id": order.order_id,
                "transaction_signature": tx_signature,
                "execution_price": order.average_price
            }

        except Exception as e:
            logger.error(f"Error executing live order {order.order_id}: {e}")
            order.status = 'failed'
            order.error_message = str(e)

            return {
                "status": "failed",
                "order_id": order.order_id,
                "error": str(e)
            }

    def _check_emergency_conditions(self) -> bool:
        """Check if emergency stop conditions are met"""
        # Reset daily PnL if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset:
            self.daily_pnl = 0.0
            self.last_reset = current_date

        # Check daily loss limit
        if abs(self.daily_pnl) >= self.max_daily_loss:
            logger.critical(f"Daily loss limit reached: {self.daily_pnl}")
            return True

        return False

    def _check_daily_loss_limit(self, order: LiveOrder) -> bool:
        """Check if order would exceed daily loss limit"""
        # Simplified check - would need more sophisticated risk calculation
        estimated_risk = order.amount * 0.1  # Assume 10% risk
        return (abs(self.daily_pnl) + estimated_risk) >= self.max_daily_loss

    async def _monitor_positions(self):
        """Monitor open positions"""
        while self.running:
            try:
                # Monitor position risk, P&L, etc.
                # This would integrate with portfolio management
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")

    async def _monitor_risk(self):
        """Monitor overall risk metrics"""
        while self.running:
            try:
                # Monitor portfolio risk, exposure, etc.
                # This would integrate with risk management
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error monitoring risk: {e}")

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.execution_times:
            return {"avg_latency_ms": 0, "max_latency_ms": 0, "orders_count": 0}

        return {
            "avg_latency_ms": sum(self.execution_times) / len(self.execution_times),
            "max_latency_ms": max(self.execution_times),
            "min_latency_ms": min(self.execution_times),
            "orders_count": len(self.execution_times),
            "daily_pnl": self.daily_pnl,
            "paper_trading": self.paper_trading
        }

# Global instance
live_trading_engine = None

def get_live_trading_engine(config: Dict = None) -> LiveTradingEngine:
    """Get or create live trading engine instance"""
    global live_trading_engine
    if live_trading_engine is None and config:
        live_trading_engine = LiveTradingEngine(config)
    return live_trading_engine
