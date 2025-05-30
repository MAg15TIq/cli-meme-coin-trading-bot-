"""
Backtesting engine for the Solana Memecoin Trading Bot.
Allows testing trading strategies against historical data.
"""

import json
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class OrderType(Enum):
    """Order types for backtesting."""
    BUY = "buy"
    SELL = "sell"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Order status for backtesting."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class BacktestOrder:
    """Represents an order in the backtesting system."""
    id: str
    timestamp: datetime
    order_type: OrderType
    token_mint: str
    amount: float
    price: float
    status: OrderStatus = OrderStatus.PENDING
    filled_timestamp: Optional[datetime] = None
    filled_price: Optional[float] = None
    fees: float = 0.0


@dataclass
class BacktestPosition:
    """Represents a position in the backtesting system."""
    token_mint: str
    token_name: str
    amount: float
    entry_price: float
    entry_timestamp: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class BacktestMetrics:
    """Performance metrics for a backtest."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    start_balance: float = 0.0
    end_balance: float = 0.0
    roi: float = 0.0


class BacktestingEngine:
    """Backtesting engine for trading strategies."""

    def __init__(self):
        """Initialize the backtesting engine."""
        self.enabled = get_config_value("backtesting_enabled", False)
        self.data_path = get_config_value("backtesting_data_path", "backtesting_data")

        # Backtesting state
        self.current_time = None
        self.balance = 0.0
        self.initial_balance = 0.0
        self.positions = {}  # token_mint -> BacktestPosition
        self.orders = {}  # order_id -> BacktestOrder
        self.trade_history = []
        self.balance_history = []

        # Price data
        self.price_data = {}  # token_mint -> DataFrame with OHLCV data

        # Strategy function
        self.strategy_function = None

        # Configuration
        self.trading_fee = get_config_value("backtesting_trading_fee", 0.003)  # 0.3%
        self.slippage = get_config_value("backtesting_slippage", 0.001)  # 0.1%

        logger.info("Backtesting engine initialized")

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable backtesting.

        Args:
            enabled: Whether backtesting should be enabled
        """
        self.enabled = enabled
        update_config("backtesting_enabled", enabled)
        logger.info(f"Backtesting {'enabled' if enabled else 'disabled'}")

    def load_price_data(self, token_mint: str, data: pd.DataFrame) -> None:
        """
        Load price data for a token.

        Args:
            token_mint: Token mint address
            data: DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Ensure the data is sorted by timestamp
        data = data.sort_values('timestamp')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)

        self.price_data[token_mint] = data
        logger.info(f"Loaded {len(data)} price data points for {token_mint}")

    def load_price_data_from_file(self, token_mint: str, file_path: str) -> None:
        """
        Load price data from a CSV file.

        Args:
            token_mint: Token mint address
            file_path: Path to CSV file with OHLCV data
        """
        try:
            data = pd.read_csv(file_path)
            self.load_price_data(token_mint, data)
        except Exception as e:
            logger.error(f"Error loading price data from {file_path}: {e}")
            raise

    def set_strategy(self, strategy_function: Callable) -> None:
        """
        Set the trading strategy function.

        Args:
            strategy_function: Function that takes (engine, current_time, price_data) and executes trades
        """
        self.strategy_function = strategy_function
        logger.info("Trading strategy set")

    def reset_state(self, initial_balance: float = 1000.0) -> None:
        """
        Reset the backtesting state.

        Args:
            initial_balance: Starting balance for the backtest
        """
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.orders = {}
        self.trade_history = []
        self.balance_history = []
        self.current_time = None

        logger.info(f"Backtesting state reset with initial balance: {initial_balance}")

    def get_current_price(self, token_mint: str) -> Optional[float]:
        """
        Get the current price for a token at the current backtest time.

        Args:
            token_mint: Token mint address

        Returns:
            Current price or None if not available
        """
        if token_mint not in self.price_data or self.current_time is None:
            return None

        data = self.price_data[token_mint]

        # Find the closest price data point
        try:
            # Get the price at or before the current time
            price_row = data[data.index <= self.current_time].iloc[-1]
            return float(price_row['close'])
        except (IndexError, KeyError):
            return None

    def place_order(self, order_type: OrderType, token_mint: str, amount: float,
                   price: Optional[float] = None) -> str:
        """
        Place an order in the backtesting system.

        Args:
            order_type: Type of order
            token_mint: Token mint address
            amount: Amount to trade
            price: Price for limit orders (None for market orders)

        Returns:
            Order ID
        """
        if not self.current_time:
            raise ValueError("Current time not set")

        # Generate order ID
        order_id = f"{int(self.current_time.timestamp())}_{len(self.orders)}"

        # Use current market price if no price specified
        if price is None:
            price = self.get_current_price(token_mint)
            if price is None:
                raise ValueError(f"No price data available for {token_mint}")

        # Create order
        order = BacktestOrder(
            id=order_id,
            timestamp=self.current_time,
            order_type=order_type,
            token_mint=token_mint,
            amount=amount,
            price=price
        )

        self.orders[order_id] = order

        # Try to fill the order immediately for market orders
        if order_type in [OrderType.BUY, OrderType.SELL]:
            self._try_fill_order(order_id)

        return order_id

    def _try_fill_order(self, order_id: str) -> bool:
        """
        Try to fill an order.

        Args:
            order_id: Order ID

        Returns:
            True if order was filled
        """
        order = self.orders.get(order_id)
        if not order or order.status != OrderStatus.PENDING:
            return False

        current_price = self.get_current_price(order.token_mint)
        if current_price is None:
            return False

        # Check if order can be filled
        can_fill = False
        fill_price = current_price

        if order.order_type == OrderType.BUY:
            # For buy orders, fill at current price (market order)
            can_fill = True
            # Apply slippage
            fill_price = current_price * (1 + self.slippage)
        elif order.order_type == OrderType.SELL:
            # For sell orders, fill at current price (market order)
            can_fill = True
            # Apply slippage
            fill_price = current_price * (1 - self.slippage)

        if can_fill:
            return self._fill_order(order_id, fill_price)

        return False

    def _fill_order(self, order_id: str, fill_price: float) -> bool:
        """
        Fill an order.

        Args:
            order_id: Order ID
            fill_price: Price at which to fill the order

        Returns:
            True if order was successfully filled
        """
        order = self.orders.get(order_id)
        if not order:
            return False

        try:
            if order.order_type == OrderType.BUY:
                # Calculate cost including fees
                cost = order.amount * fill_price
                fees = cost * self.trading_fee
                total_cost = cost + fees

                # Check if we have enough balance
                if self.balance < total_cost:
                    logger.warning(f"Insufficient balance for buy order {order_id}")
                    order.status = OrderStatus.CANCELLED
                    return False

                # Execute buy
                self.balance -= total_cost

                # Create or update position
                if order.token_mint in self.positions:
                    # Average down the position
                    pos = self.positions[order.token_mint]
                    total_amount = pos.amount + order.amount
                    avg_price = ((pos.amount * pos.entry_price) + (order.amount * fill_price)) / total_amount
                    pos.amount = total_amount
                    pos.entry_price = avg_price
                else:
                    # Create new position
                    self.positions[order.token_mint] = BacktestPosition(
                        token_mint=order.token_mint,
                        token_name=order.token_mint[:8],
                        amount=order.amount,
                        entry_price=fill_price,
                        entry_timestamp=self.current_time,
                        current_price=fill_price
                    )

                order.fees = fees

            elif order.order_type == OrderType.SELL:
                # Check if we have the position
                if order.token_mint not in self.positions:
                    logger.warning(f"No position to sell for order {order_id}")
                    order.status = OrderStatus.CANCELLED
                    return False

                pos = self.positions[order.token_mint]
                if pos.amount < order.amount:
                    logger.warning(f"Insufficient position size for sell order {order_id}")
                    order.status = OrderStatus.CANCELLED
                    return False

                # Execute sell
                proceeds = order.amount * fill_price
                fees = proceeds * self.trading_fee
                net_proceeds = proceeds - fees

                self.balance += net_proceeds

                # Calculate realized P&L
                realized_pnl = (fill_price - pos.entry_price) * order.amount - fees
                pos.realized_pnl += realized_pnl

                # Update position
                pos.amount -= order.amount
                if pos.amount <= 0:
                    # Close position
                    del self.positions[order.token_mint]

                order.fees = fees

                # Record trade
                self.trade_history.append({
                    'timestamp': self.current_time,
                    'token_mint': order.token_mint,
                    'type': 'sell',
                    'amount': order.amount,
                    'price': fill_price,
                    'pnl': realized_pnl,
                    'fees': fees
                })

            # Mark order as filled
            order.status = OrderStatus.FILLED
            order.filled_timestamp = self.current_time
            order.filled_price = fill_price

            logger.debug(f"Order {order_id} filled at {fill_price}")
            return True

        except Exception as e:
            logger.error(f"Error filling order {order_id}: {e}")
            order.status = OrderStatus.CANCELLED
            return False

    def update_positions(self) -> None:
        """Update all positions with current prices."""
        for token_mint, position in self.positions.items():
            current_price = self.get_current_price(token_mint)
            if current_price:
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.amount

    def run_backtest(self, start_date: datetime, end_date: datetime,
                    time_step: timedelta = timedelta(hours=1)) -> BacktestMetrics:
        """
        Run a backtest over a specified time period.

        Args:
            start_date: Start date for the backtest
            end_date: End date for the backtest
            time_step: Time step for the backtest

        Returns:
            Backtest performance metrics
        """
        if not self.strategy_function:
            raise ValueError("No strategy function set")

        if not self.price_data:
            raise ValueError("No price data loaded")

        logger.info(f"Starting backtest from {start_date} to {end_date}")

        # Reset state
        self.reset_state()

        # Run backtest
        current_time = start_date
        while current_time <= end_date:
            self.current_time = current_time

            # Update positions with current prices
            self.update_positions()

            # Record balance
            total_value = self.balance + sum(
                pos.amount * pos.current_price
                for pos in self.positions.values()
                if pos.current_price > 0
            )
            self.balance_history.append({
                'timestamp': current_time,
                'balance': self.balance,
                'total_value': total_value
            })

            # Execute strategy
            try:
                self.strategy_function(self, current_time, self.price_data)
            except Exception as e:
                logger.error(f"Error executing strategy at {current_time}: {e}")

            # Process pending orders
            for order_id in list(self.orders.keys()):
                order = self.orders[order_id]
                if order.status == OrderStatus.PENDING:
                    self._try_fill_order(order_id)

            current_time += time_step

        # Calculate final metrics
        metrics = self._calculate_metrics()

        logger.info(f"Backtest completed. Total trades: {metrics.total_trades}, "
                   f"Win rate: {metrics.win_rate:.2%}, Net P&L: {metrics.net_pnl:.2f}")

        return metrics

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate performance metrics for the backtest."""
        metrics = BacktestMetrics()

        # Basic metrics
        metrics.start_balance = self.initial_balance
        metrics.end_balance = self.balance + sum(
            pos.amount * pos.current_price
            for pos in self.positions.values()
            if pos.current_price > 0
        )

        # Trade statistics
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['pnl'] < 0]

        metrics.total_trades = len(self.trade_history)
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        metrics.win_rate = metrics.winning_trades / max(metrics.total_trades, 1)

        # P&L calculations
        metrics.total_pnl = sum(t['pnl'] for t in self.trade_history)
        metrics.total_fees = sum(t['fees'] for t in self.trade_history)
        metrics.net_pnl = metrics.total_pnl - metrics.total_fees
        metrics.roi = (metrics.end_balance - metrics.start_balance) / metrics.start_balance

        # Win/Loss statistics
        if winning_trades:
            metrics.average_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
            metrics.largest_win = max(t['pnl'] for t in winning_trades)

        if losing_trades:
            metrics.average_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
            metrics.largest_loss = min(t['pnl'] for t in losing_trades)

        # Profit factor
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        metrics.profit_factor = total_wins / max(total_losses, 1)

        # Risk metrics
        if self.balance_history:
            values = [b['total_value'] for b in self.balance_history]
            returns = np.diff(values) / values[:-1]

            # Maximum drawdown
            peak = np.maximum.accumulate(values)
            drawdown = (peak - values) / peak
            metrics.max_drawdown = np.max(drawdown)

            # Sharpe ratio (assuming 0% risk-free rate)
            if len(returns) > 1 and np.std(returns) > 0:
                metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized

            # Sortino ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 1 and np.std(negative_returns) > 0:
                metrics.sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252)

        return metrics

    def get_backtest_summary(self, metrics: BacktestMetrics) -> Dict[str, Any]:
        """
        Get a summary of backtest results.

        Args:
            metrics: Backtest metrics

        Returns:
            Summary dictionary
        """
        return {
            'performance': {
                'total_return': f"{metrics.roi:.2%}",
                'net_pnl': f"{metrics.net_pnl:.2f}",
                'start_balance': f"{metrics.start_balance:.2f}",
                'end_balance': f"{metrics.end_balance:.2f}",
                'max_drawdown': f"{metrics.max_drawdown:.2%}"
            },
            'trading': {
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'win_rate': f"{metrics.win_rate:.2%}",
                'profit_factor': f"{metrics.profit_factor:.2f}"
            },
            'risk': {
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                'largest_win': f"{metrics.largest_win:.2f}",
                'largest_loss': f"{metrics.largest_loss:.2f}"
            },
            'costs': {
                'total_fees': f"{metrics.total_fees:.2f}",
                'average_fee_per_trade': f"{metrics.total_fees / max(metrics.total_trades, 1):.2f}"
            }
        }

    def export_results(self, metrics: BacktestMetrics, file_path: str) -> None:
        """
        Export backtest results to a file.

        Args:
            metrics: Backtest metrics
            file_path: Path to save the results
        """
        try:
            results = {
                'metrics': asdict(metrics),
                'summary': self.get_backtest_summary(metrics),
                'trade_history': self.trade_history,
                'balance_history': self.balance_history,
                'final_positions': {
                    mint: asdict(pos) for mint, pos in self.positions.items()
                }
            }

            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Backtest results exported to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting backtest results: {e}")
            raise


# Create a singleton instance
backtesting_engine = BacktestingEngine()
