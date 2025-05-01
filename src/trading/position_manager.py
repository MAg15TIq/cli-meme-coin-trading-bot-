"""
Position management module for the Solana Memecoin Trading Bot.
Handles tracking open positions, monitoring prices, and executing stop-loss/take-profit orders.
Includes enhanced risk management and exit strategies.
"""

import json
import logging
import threading
import time
import math
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum

from src.trading.jupiter_api import jupiter_api
from src.wallet.wallet import wallet_manager
from src.trading.technical_analysis import technical_analyzer
from src.trading.risk_manager import risk_manager
from src.solana.gas_optimizer import gas_optimizer, TransactionType
from config import get_config_value, update_config
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk level for a position."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ExitStrategy(Enum):
    """Exit strategy types."""
    FIXED = "fixed"           # Fixed take-profit and stop-loss
    TRAILING = "trailing"     # Trailing stop-loss
    TIERED = "tiered"         # Tiered take-profits
    TIME_BASED = "time_based" # Time-based exit
    INDICATOR = "indicator"   # Technical indicator based


class TakeProfitLevel:
    """Represents a take-profit level for a position."""

    def __init__(self, price: float, percentage: float, sell_percentage: float):
        """
        Initialize a take-profit level.

        Args:
            price: The price at which to take profit (in SOL)
            percentage: The percentage increase from entry price
            sell_percentage: The percentage of the position to sell
        """
        self.price = price
        self.percentage = percentage
        self.sell_percentage = sell_percentage
        self.triggered = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "price": self.price,
            "percentage": self.percentage,
            "sell_percentage": self.sell_percentage,
            "triggered": self.triggered
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TakeProfitLevel':
        """Create from dictionary."""
        tp = cls(
            price=data["price"],
            percentage=data["percentage"],
            sell_percentage=data["sell_percentage"]
        )
        tp.triggered = data.get("triggered", False)
        return tp


class TrailingStop:
    """Represents a trailing stop for a position."""

    def __init__(self, initial_price: float, trail_percent: float):
        """
        Initialize a trailing stop.

        Args:
            initial_price: The initial price to start trailing from
            trail_percent: The percentage below the highest price to set the stop
        """
        self.highest_price = initial_price
        self.trail_percent = trail_percent
        self.stop_price = initial_price * (1 - trail_percent / 100)
        self.activated = False
        self.activation_threshold = initial_price * 1.05  # 5% above entry by default

    def update(self, current_price: float) -> None:
        """
        Update the trailing stop based on the current price.

        Args:
            current_price: The current price
        """
        # Check if trailing stop should be activated
        if not self.activated and current_price >= self.activation_threshold:
            self.activated = True
            logger.info(f"Trailing stop activated at {current_price}")

        # Only update if activated
        if self.activated:
            # If price has moved higher, update the trailing stop
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.stop_price = current_price * (1 - self.trail_percent / 100)
                logger.debug(f"Updated trailing stop to {self.stop_price} ({self.trail_percent}% below {self.highest_price})")

    def is_triggered(self, current_price: float) -> bool:
        """
        Check if the trailing stop is triggered.

        Args:
            current_price: The current price

        Returns:
            True if triggered, False otherwise
        """
        return self.activated and current_price <= self.stop_price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "highest_price": self.highest_price,
            "trail_percent": self.trail_percent,
            "stop_price": self.stop_price,
            "activated": self.activated,
            "activation_threshold": self.activation_threshold
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrailingStop':
        """Create from dictionary."""
        ts = cls(
            initial_price=data["highest_price"],
            trail_percent=data["trail_percent"]
        )
        ts.stop_price = data["stop_price"]
        ts.activated = data.get("activated", False)
        ts.activation_threshold = data.get("activation_threshold", ts.highest_price * 1.05)
        return ts


class TimeBasedExit:
    """Represents a time-based exit strategy."""

    def __init__(self, expiry_time: datetime, min_profit_percent: float = 0.0):
        """
        Initialize a time-based exit.

        Args:
            expiry_time: The time at which to exit the position
            min_profit_percent: Minimum profit percentage required to exit (0 means exit regardless of profit)
        """
        self.expiry_time = expiry_time
        self.min_profit_percent = min_profit_percent

    def is_triggered(self, current_time: datetime, current_pnl: float) -> bool:
        """
        Check if the time-based exit is triggered.

        Args:
            current_time: The current time
            current_pnl: The current profit/loss percentage

        Returns:
            True if triggered, False otherwise
        """
        if current_time >= self.expiry_time:
            # If minimum profit is required, check if we've reached it
            if self.min_profit_percent > 0:
                return current_pnl >= self.min_profit_percent
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "expiry_time": self.expiry_time.isoformat(),
            "min_profit_percent": self.min_profit_percent
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeBasedExit':
        """Create from dictionary."""
        return cls(
            expiry_time=datetime.fromisoformat(data["expiry_time"]),
            min_profit_percent=data.get("min_profit_percent", 0.0)
        )


class IndicatorBasedExit:
    """Represents an indicator-based exit strategy."""

    def __init__(self, indicator_type: str, params: Dict[str, Any]):
        """
        Initialize an indicator-based exit.

        Args:
            indicator_type: The type of indicator (e.g., "rsi", "macd")
            params: Parameters for the indicator
        """
        self.indicator_type = indicator_type
        self.params = params
        self.last_check_time = datetime.now()
        self.check_interval = timedelta(minutes=params.get("check_interval_minutes", 5))

    def is_triggered(self, token_mint: str, current_time: datetime) -> bool:
        """
        Check if the indicator-based exit is triggered.

        Args:
            token_mint: The token mint address
            current_time: The current time

        Returns:
            True if triggered, False otherwise
        """
        # Only check at the specified interval
        if current_time - self.last_check_time < self.check_interval:
            return False

        self.last_check_time = current_time

        # Check the indicator
        try:
            if self.indicator_type == "rsi":
                # Get RSI value
                rsi_value = technical_analyzer.get_rsi(token_mint)

                # Check if RSI is above the overbought threshold
                if rsi_value is not None and rsi_value > self.params.get("overbought", 70):
                    logger.info(f"RSI exit triggered: {rsi_value} > {self.params.get('overbought', 70)}")
                    return True

            elif self.indicator_type == "macd":
                # Get MACD values
                macd_result = technical_analyzer.get_macd(token_mint)

                if macd_result is not None:
                    macd_line, signal_line = macd_result

                    # Check for bearish crossover
                    if macd_line < signal_line and self.params.get("exit_on_bearish_cross", True):
                        logger.info(f"MACD bearish crossover exit triggered: {macd_line} < {signal_line}")
                        return True

            # Add more indicators as needed

        except Exception as e:
            logger.error(f"Error checking indicator-based exit: {e}")

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "indicator_type": self.indicator_type,
            "params": self.params,
            "last_check_time": self.last_check_time.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndicatorBasedExit':
        """Create from dictionary."""
        exit_strategy = cls(
            indicator_type=data["indicator_type"],
            params=data["params"]
        )
        exit_strategy.last_check_time = datetime.fromisoformat(data.get("last_check_time", datetime.now().isoformat()))
        return exit_strategy


class Position:
    """Represents an open trading position with enhanced risk management."""

    def __init__(self, token_mint: str, token_name: str, amount: float, entry_price: float,
                 entry_time: datetime, stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None, decimals: int = 9):
        """
        Initialize a position.

        Args:
            token_mint: The mint address of the token
            token_name: The name of the token (for display)
            amount: The amount of tokens held
            entry_price: The price at which the position was entered (in SOL)
            entry_time: The time at which the position was entered
            stop_loss: Optional stop-loss price (in SOL)
            take_profit: Optional take-profit price (in SOL) - for backward compatibility
            decimals: The number of decimals for the token
        """
        self.token_mint = token_mint
        self.token_name = token_name
        self.amount = amount
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.decimals = decimals
        self.current_price = entry_price
        self.last_updated = entry_time

        # Multi-tiered take-profits
        self.take_profit_levels: List[TakeProfitLevel] = []

        # For backward compatibility
        if take_profit is not None:
            self.add_take_profit_level(take_profit, 100.0, 100.0)
            self.take_profit = take_profit  # Keep the original take_profit property
        else:
            self.take_profit = None

        # Enhanced risk management
        self.risk_level = RiskLevel.MEDIUM  # Default risk level
        self.position_size_percent = 0.0    # Percentage of portfolio
        self.max_loss_sol = 0.0             # Maximum loss in SOL

        # Enhanced exit strategies
        self.exit_strategies: Dict[str, Any] = {}
        self.trailing_stop: Optional[TrailingStop] = None
        self.time_based_exit: Optional[TimeBasedExit] = None
        self.indicator_based_exit: Optional[IndicatorBasedExit] = None

    def update_price(self, price: float) -> None:
        """
        Update the current price of the position.

        Args:
            price: The new price
        """
        self.current_price = price
        self.last_updated = datetime.now()

    def get_pnl(self) -> float:
        """
        Get the profit/loss of the position.

        Returns:
            The profit/loss as a percentage
        """
        return (self.current_price - self.entry_price) / self.entry_price * 100

    def get_value(self) -> float:
        """
        Get the current value of the position in SOL.

        Returns:
            The value in SOL
        """
        return self.amount * self.current_price

    def add_take_profit_level(self, price: float, percentage: float, sell_percentage: float) -> None:
        """
        Add a take-profit level.

        Args:
            price: The price at which to take profit (in SOL)
            percentage: The percentage increase from entry price
            sell_percentage: The percentage of the position to sell
        """
        tp_level = TakeProfitLevel(price, percentage, sell_percentage)
        self.take_profit_levels.append(tp_level)
        self.take_profit_levels.sort(key=lambda tp: tp.price)  # Sort by price
        logger.info(f"Added take-profit level at {price} SOL ({percentage}%) for {self.token_name}")

    def add_take_profit_percentage(self, percentage: float, sell_percentage: float) -> None:
        """
        Add a take-profit level based on percentage increase.

        Args:
            percentage: The percentage increase from entry price
            sell_percentage: The percentage of the position to sell
        """
        price = self.entry_price * (1 + percentage / 100)
        self.add_take_profit_level(price, percentage, sell_percentage)

    def clear_take_profit_levels(self) -> None:
        """
        Clear all take-profit levels.
        """
        self.take_profit_levels = []
        self.take_profit = None
        logger.info(f"Cleared take-profit levels for {self.token_name}")

    def get_take_profit_levels(self) -> List[TakeProfitLevel]:
        """
        Get all take-profit levels.

        Returns:
            List of take-profit levels
        """
        return self.take_profit_levels

    def set_trailing_stop(self, trail_percent: float, activation_threshold_percent: float = 5.0) -> None:
        """
        Set a trailing stop for the position.

        Args:
            trail_percent: The percentage below the highest price to set the stop
            activation_threshold_percent: The percentage above entry price to activate the trailing stop
        """
        activation_threshold = self.entry_price * (1 + activation_threshold_percent / 100)
        self.trailing_stop = TrailingStop(self.entry_price, trail_percent)
        self.trailing_stop.activation_threshold = activation_threshold
        logger.info(f"Set trailing stop for {self.token_name} at {trail_percent}% below highest price, " +
                   f"activates at {activation_threshold_percent}% above entry")

    def set_time_based_exit(self, hours: float, min_profit_percent: float = 0.0) -> None:
        """
        Set a time-based exit for the position.

        Args:
            hours: The number of hours after entry to exit
            min_profit_percent: Minimum profit percentage required to exit
        """
        expiry_time = self.entry_time + timedelta(hours=hours)
        self.time_based_exit = TimeBasedExit(expiry_time, min_profit_percent)
        logger.info(f"Set time-based exit for {self.token_name} at {expiry_time.isoformat()}, " +
                   f"minimum profit: {min_profit_percent}%")

    def set_indicator_based_exit(self, indicator_type: str, params: Dict[str, Any]) -> None:
        """
        Set an indicator-based exit for the position.

        Args:
            indicator_type: The type of indicator (e.g., "rsi", "macd")
            params: Parameters for the indicator
        """
        self.indicator_based_exit = IndicatorBasedExit(indicator_type, params)
        logger.info(f"Set {indicator_type} indicator-based exit for {self.token_name}")

    def set_risk_level(self, risk_level: RiskLevel) -> None:
        """
        Set the risk level for the position.

        Args:
            risk_level: The risk level
        """
        self.risk_level = risk_level
        logger.info(f"Set risk level for {self.token_name} to {risk_level.value}")

    def set_position_size(self, portfolio_percent: float, max_loss_sol: float) -> None:
        """
        Set position size parameters.

        Args:
            portfolio_percent: The percentage of the portfolio allocated to this position
            max_loss_sol: The maximum loss in SOL
        """
        self.position_size_percent = portfolio_percent
        self.max_loss_sol = max_loss_sol
        logger.info(f"Set position size for {self.token_name} to {portfolio_percent}% of portfolio, " +
                   f"max loss: {max_loss_sol} SOL")

    def check_exit_strategies(self, current_time: datetime = None) -> bool:
        """
        Check all exit strategies for the position.

        Args:
            current_time: The current time (defaults to now)

        Returns:
            True if any exit strategy is triggered, False otherwise
        """
        if current_time is None:
            current_time = datetime.now()

        # Check trailing stop
        if self.trailing_stop is not None:
            self.trailing_stop.update(self.current_price)
            if self.trailing_stop.is_triggered(self.current_price):
                logger.info(f"Trailing stop triggered for {self.token_name} at {self.current_price} SOL")
                return True

        # Check time-based exit
        if self.time_based_exit is not None:
            if self.time_based_exit.is_triggered(current_time, self.get_pnl()):
                logger.info(f"Time-based exit triggered for {self.token_name} at {current_time.isoformat()}")
                return True

        # Check indicator-based exit
        if self.indicator_based_exit is not None:
            if self.indicator_based_exit.is_triggered(self.token_mint, current_time):
                logger.info(f"Indicator-based exit triggered for {self.token_name}")
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the position to a dictionary.

        Returns:
            The position as a dictionary
        """
        result = {
            "token_mint": self.token_mint,
            "token_name": self.token_name,
            "amount": self.amount,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,  # For backward compatibility
            "take_profit_levels": [tp.to_dict() for tp in self.take_profit_levels],
            "decimals": self.decimals,
            "current_price": self.current_price,
            "last_updated": self.last_updated.isoformat(),
            "risk_level": self.risk_level.value,
            "position_size_percent": self.position_size_percent,
            "max_loss_sol": self.max_loss_sol
        }

        # Add enhanced exit strategies if they exist
        if self.trailing_stop is not None:
            result["trailing_stop"] = self.trailing_stop.to_dict()

        if self.time_based_exit is not None:
            result["time_based_exit"] = self.time_based_exit.to_dict()

        if self.indicator_based_exit is not None:
            result["indicator_based_exit"] = self.indicator_based_exit.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """
        Create a position from a dictionary.

        Args:
            data: The position data

        Returns:
            The position
        """
        position = cls(
            token_mint=data["token_mint"],
            token_name=data["token_name"],
            amount=data["amount"],
            entry_price=data["entry_price"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),  # For backward compatibility
            decimals=data.get("decimals", 9)
        )

        # Load take-profit levels if available
        if "take_profit_levels" in data:
            position.take_profit_levels = [
                TakeProfitLevel.from_dict(tp_data)
                for tp_data in data["take_profit_levels"]
            ]

        # Load risk management settings
        if "risk_level" in data:
            try:
                position.risk_level = RiskLevel(data["risk_level"])
            except ValueError:
                position.risk_level = RiskLevel.MEDIUM

        position.position_size_percent = data.get("position_size_percent", 0.0)
        position.max_loss_sol = data.get("max_loss_sol", 0.0)

        # Load enhanced exit strategies
        if "trailing_stop" in data:
            position.trailing_stop = TrailingStop.from_dict(data["trailing_stop"])

        if "time_based_exit" in data:
            position.time_based_exit = TimeBasedExit.from_dict(data["time_based_exit"])

        if "indicator_based_exit" in data:
            position.indicator_based_exit = IndicatorBasedExit.from_dict(data["indicator_based_exit"])

        return position


class PositionManager:
    """Manager for tracking and monitoring positions."""

    def __init__(self):
        """Initialize the position manager."""
        self.positions: Dict[str, Position] = {}  # token_mint -> Position
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.monitoring_interval = int(get_config_value("monitoring_interval_seconds", "60"))
        self.positions_file = Path.home() / ".solana-trading-bot" / "positions.json"
        self.load_positions()

    def add_position(self, position: Position) -> None:
        """
        Add a position to track.

        Args:
            position: The position to add
        """
        self.positions[position.token_mint] = position
        logger.info(f"Added position: {position.token_name} ({position.token_mint})")
        self.save_positions()

        # Start monitoring if not already running
        self.ensure_monitoring_running()

    def remove_position(self, token_mint: str) -> Optional[Position]:
        """
        Remove a position from tracking.

        Args:
            token_mint: The mint address of the token

        Returns:
            The removed position, or None if not found
        """
        position = self.positions.pop(token_mint, None)
        if position:
            logger.info(f"Removed position: {position.token_name} ({position.token_mint})")
            self.save_positions()
        return position

    def get_position(self, token_mint: str) -> Optional[Position]:
        """
        Get a position by token mint.

        Args:
            token_mint: The mint address of the token

        Returns:
            The position, or None if not found
        """
        return self.positions.get(token_mint)

    def get_all_positions(self) -> List[Position]:
        """
        Get all tracked positions.

        Returns:
            List of positions
        """
        return list(self.positions.values())

    def update_position_price(self, token_mint: str, price: Optional[float] = None) -> None:
        """
        Update the price of a position.

        Args:
            token_mint: The mint address of the token
            price: The new price (if None, fetched from Jupiter)
        """
        position = self.get_position(token_mint)
        if not position:
            logger.warning(f"Position not found: {token_mint}")
            return

        try:
            if price is None:
                # Fetch the current price from Jupiter
                price = jupiter_api.get_token_price(token_mint)

            position.update_price(price)
            logger.debug(f"Updated price for {position.token_name}: {price} SOL")
            self.save_positions()
        except Exception as e:
            logger.error(f"Error updating price for {token_mint}: {e}")

    def update_all_prices(self) -> None:
        """Update prices for all positions."""
        for token_mint in list(self.positions.keys()):
            self.update_position_price(token_mint)

    def check_stop_loss_take_profit(self, token_mint: str) -> bool:
        """
        Check if a position has hit its stop-loss, take-profit level, or other exit strategies.

        Args:
            token_mint: The mint address of the token

        Returns:
            True if an order was triggered, False otherwise
        """
        position = self.get_position(token_mint)
        if not position:
            logger.warning(f"Position not found: {token_mint}")
            return False

        # Check enhanced exit strategies
        if position.check_exit_strategies():
            logger.info(f"Enhanced exit strategy triggered for {position.token_name}")
            self.execute_sell(token_mint, "enhanced_exit")
            return True

        # Check traditional stop-loss
        if position.stop_loss is not None and position.current_price <= position.stop_loss:
            logger.info(f"Stop-loss triggered for {position.token_name} at {position.current_price} SOL")
            self.execute_sell(token_mint, "stop_loss")
            return True

        # Check multi-tiered take-profits
        triggered = False
        for tp_level in sorted(position.take_profit_levels, key=lambda tp: tp.price, reverse=True):
            if not tp_level.triggered and position.current_price >= tp_level.price:
                logger.info(f"Take-profit triggered for {position.token_name} at {position.current_price} SOL (level: {tp_level.percentage}%)")

                # Calculate amount to sell
                sell_amount = position.amount * (tp_level.sell_percentage / 100)

                # Mark as triggered
                tp_level.triggered = True

                # Execute partial sell
                if sell_amount >= position.amount:
                    # Sell all
                    self.execute_sell(token_mint, "take_profit")
                    triggered = True
                    break
                else:
                    # Partial sell
                    self._execute_partial_sell(token_mint, sell_amount, "take_profit")
                    triggered = True

        # For backward compatibility
        if not triggered and position.take_profit is not None and position.current_price >= position.take_profit:
            logger.info(f"Legacy take-profit triggered for {position.token_name} at {position.current_price} SOL")
            self.execute_sell(token_mint, "take_profit")
            return True

        return triggered

    def _execute_partial_sell(self, token_mint: str, amount: float, trigger_type: str) -> Optional[str]:
        """
        Execute a partial sell for a position.

        Args:
            token_mint: The mint address of the token
            amount: The amount to sell
            trigger_type: The type of trigger ("stop_loss", "take_profit", "manual")

        Returns:
            The transaction signature, or None if failed
        """
        position = self.get_position(token_mint)
        if not position:
            logger.warning(f"Position not found: {token_mint}")
            return None

        # Get the current wallet
        wallet = wallet_manager.get_current_keypair()
        if not wallet:
            logger.error("No wallet connected")
            return None

        try:
            # Get priority fee
            priority_fee = jupiter_api.get_priority_fee()

            # Execute the sell
            tx_signature = jupiter_api.execute_sell(
                token_mint=position.token_mint,
                amount_token=amount,
                decimals=position.decimals,
                wallet=wallet,
                priority_fee=priority_fee
            )

            logger.info(f"Partially sold {amount} {position.token_name} for {position.current_price * amount} SOL")

            # Update the position
            position.amount -= amount
            self.save_positions()

            return tx_signature
        except Exception as e:
            logger.error(f"Error partially selling position {token_mint}: {e}")
            return None

    def check_all_stop_loss_take_profit(self) -> None:
        """Check stop-loss and take-profit for all positions."""
        for token_mint in list(self.positions.keys()):
            self.check_stop_loss_take_profit(token_mint)

    def execute_sell(self, token_mint: str, trigger_type: str) -> Optional[str]:
        """
        Execute a sell order for a position.

        Args:
            token_mint: The mint address of the token
            trigger_type: The type of trigger ("stop_loss", "take_profit", "manual")

        Returns:
            The transaction signature, or None if failed
        """
        position = self.get_position(token_mint)
        if not position:
            logger.warning(f"Position not found: {token_mint}")
            return None

        # Get the current wallet
        wallet = wallet_manager.get_current_keypair()
        if not wallet:
            logger.error("No wallet connected")
            return None

        try:
            # Get priority fee
            priority_fee = jupiter_api.get_priority_fee()

            # Execute the sell
            tx_signature = jupiter_api.execute_sell(
                token_mint=position.token_mint,
                amount_token=position.amount,
                decimals=position.decimals,
                wallet=wallet,
                priority_fee=priority_fee
            )

            logger.info(f"Sold {position.amount} {position.token_name} for {position.current_price * position.amount} SOL")

            # Remove the position
            self.remove_position(token_mint)

            return tx_signature
        except Exception as e:
            logger.error(f"Error selling position {token_mint}: {e}")
            return None

    def start_monitoring(self) -> None:
        """Start the position monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.info("Monitoring thread already running")
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Started position monitoring thread")

    def stop_monitoring_thread(self) -> None:
        """Stop the position monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)
            logger.info("Stopped position monitoring thread")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                if self.positions:
                    logger.debug("Updating position prices...")
                    self.update_all_prices()

                    logger.debug("Checking stop-loss/take-profit conditions...")
                    self.check_all_stop_loss_take_profit()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            # Sleep for the monitoring interval
            self.stop_monitoring.wait(self.monitoring_interval)

    def ensure_monitoring_running(self) -> None:
        """Ensure the monitoring thread is running if there are positions."""
        if self.positions and (not self.monitoring_thread or not self.monitoring_thread.is_alive()):
            self.start_monitoring()

    def save_positions(self) -> None:
        """Save positions to file."""
        try:
            # Create directory if it doesn't exist
            self.positions_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert positions to dictionaries
            positions_data = {
                token_mint: position.to_dict()
                for token_mint, position in self.positions.items()
            }

            # Save to file
            with open(self.positions_file, 'w') as f:
                json.dump(positions_data, f, indent=4)

            logger.debug(f"Saved {len(positions_data)} positions to {self.positions_file}")
        except Exception as e:
            logger.error(f"Error saving positions: {e}")

    def load_positions(self) -> None:
        """Load positions from file."""
        try:
            if not self.positions_file.exists():
                logger.info(f"Positions file not found: {self.positions_file}")
                return

            # Load from file
            with open(self.positions_file, 'r') as f:
                positions_data = json.load(f)

            # Convert dictionaries to positions
            self.positions = {
                token_mint: Position.from_dict(position_data)
                for token_mint, position_data in positions_data.items()
            }

            logger.info(f"Loaded {len(self.positions)} positions from {self.positions_file}")

            # Start monitoring if there are positions
            self.ensure_monitoring_running()
        except Exception as e:
            logger.error(f"Error loading positions: {e}")

    def create_position_from_buy(self, token_mint: str, token_name: str, amount_token: float,
                                price_sol: float, decimals: int = 9, risk_level: RiskLevel = None) -> Position:
        """
        Create a position from a buy order with enhanced risk management.

        Args:
            token_mint: The mint address of the token
            token_name: The name of the token
            amount_token: The amount of tokens bought
            price_sol: The price per token in SOL
            decimals: The number of decimals for the token
            risk_level: Optional risk level for the position

        Returns:
            The created position
        """
        # Get risk assessment from risk manager
        if risk_level is None:
            risk_level_str = risk_manager.get_token_risk_level(token_mint).value
            try:
                risk_level = RiskLevel(risk_level_str)
            except ValueError:
                risk_level = RiskLevel.MEDIUM
                logger.warning(f"Invalid risk level from risk manager: {risk_level_str}, defaulting to MEDIUM")

        # Get stop-loss percentage based on risk level
        if risk_level == RiskLevel.LOW:
            sl_percentage = float(get_config_value("conservative_stop_loss_percent", "5.0"))
        elif risk_level == RiskLevel.MEDIUM:
            sl_percentage = float(get_config_value("moderate_stop_loss_percent", "10.0"))
        elif risk_level == RiskLevel.HIGH:
            sl_percentage = float(get_config_value("aggressive_stop_loss_percent", "15.0"))
        else:  # VERY_HIGH
            sl_percentage = float(get_config_value("aggressive_stop_loss_percent", "15.0")) * 1.5

        # Calculate stop-loss price
        stop_loss = price_sol * (1 - sl_percentage / 100)

        # Create the position
        position = Position(
            token_mint=token_mint,
            token_name=token_name,
            amount=amount_token,
            entry_price=price_sol,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            decimals=decimals
        )

        # Set risk level
        position.risk_level = risk_level

        # Calculate position size metrics
        position_value_sol = amount_token * price_sol

        # Get portfolio metrics from risk manager
        portfolio_metrics = risk_manager.update_portfolio_metrics()
        portfolio_value_sol = portfolio_metrics.get("portfolio_value_sol", self._get_portfolio_value())

        # Calculate position size percentage
        if portfolio_value_sol > 0:
            position.position_size_percent = (position_value_sol / portfolio_value_sol) * 100
        else:
            position.position_size_percent = 100.0  # First position

        # Calculate max loss based on risk level
        if risk_level == RiskLevel.LOW:
            max_loss_percent = sl_percentage
        elif risk_level == RiskLevel.MEDIUM:
            max_loss_percent = sl_percentage * 1.2
        elif risk_level == RiskLevel.HIGH:
            max_loss_percent = sl_percentage * 1.5
        else:  # VERY_HIGH
            max_loss_percent = sl_percentage * 2.0

        position.max_loss_sol = position_value_sol * (max_loss_percent / 100)

        # Add exit strategies based on risk level and config
        self._apply_exit_strategies(position)

        # Add multi-tiered take-profits based on risk level
        if risk_level == RiskLevel.LOW:
            # Conservative take-profit strategy
            position.add_take_profit_percentage(15.0, 30.0)  # Sell 30% at 15% profit
            position.add_take_profit_percentage(30.0, 40.0)  # Sell 40% at 30% profit
            position.add_take_profit_percentage(50.0, 30.0)  # Sell 30% at 50% profit
        elif risk_level == RiskLevel.MEDIUM:
            # Moderate take-profit strategy
            position.add_take_profit_percentage(20.0, 30.0)  # Sell 30% at 20% profit
            position.add_take_profit_percentage(50.0, 40.0)  # Sell 40% at 50% profit
            position.add_take_profit_percentage(100.0, 30.0)  # Sell 30% at 100% profit
        elif risk_level == RiskLevel.HIGH:
            # Aggressive take-profit strategy
            position.add_take_profit_percentage(30.0, 30.0)  # Sell 30% at 30% profit
            position.add_take_profit_percentage(75.0, 30.0)  # Sell 30% at 75% profit
            position.add_take_profit_percentage(150.0, 40.0)  # Sell 40% at 150% profit
        else:  # VERY_HIGH
            # Very aggressive take-profit strategy
            position.add_take_profit_percentage(50.0, 30.0)  # Sell 30% at 50% profit
            position.add_take_profit_percentage(100.0, 30.0)  # Sell 30% at 100% profit
            position.add_take_profit_percentage(200.0, 40.0)  # Sell 40% at 200% profit

        # Add the position
        self.add_position(position)

        return position

    def _get_portfolio_value(self) -> float:
        """
        Calculate the total value of the portfolio in SOL.

        Returns:
            The total value in SOL
        """
        total_value = 0.0

        # Add value of all positions
        for position in self.positions.values():
            total_value += position.get_value()

        # Add SOL balance if available
        try:
            wallet = wallet_manager.get_current_keypair()
            if wallet:
                from src.solana.solana_interact import solana_client
                sol_balance = solana_client.get_balance(wallet.pubkey())
                total_value += sol_balance
        except Exception as e:
            logger.warning(f"Error getting SOL balance: {e}")

        return total_value

    def _apply_exit_strategies(self, position: Position) -> None:
        """
        Apply exit strategies based on risk level and configuration.

        Args:
            position: The position to apply strategies to
        """
        # Get exit strategy settings from config
        exit_strategies = get_config_value("exit_strategies", {})

        # Apply trailing stop if enabled
        if exit_strategies.get("trailing_stop", {}).get("enabled", False):
            # Get trail percentage based on risk level
            trail_percentages = exit_strategies.get("trailing_stop", {}).get("trail_percentages", {
                "low": 10.0,
                "medium": 15.0,
                "high": 20.0,
                "very_high": 25.0
            })

            trail_percent = trail_percentages.get(position.risk_level.value, 15.0)
            activation_percent = exit_strategies.get("trailing_stop", {}).get("activation_percent", 5.0)

            position.set_trailing_stop(trail_percent, activation_percent)
            logger.info(f"Applied trailing stop to {position.token_name} with {trail_percent}% trail")

        # Apply time-based exit if enabled
        if exit_strategies.get("time_based", {}).get("enabled", False):
            # Get hold time based on risk level
            hold_hours = exit_strategies.get("time_based", {}).get("hold_hours", {
                "low": 48.0,
                "medium": 24.0,
                "high": 12.0,
                "very_high": 6.0
            })

            hours = hold_hours.get(position.risk_level.value, 24.0)
            min_profit = exit_strategies.get("time_based", {}).get("min_profit_percent", 0.0)

            position.set_time_based_exit(hours, min_profit)
            logger.info(f"Applied time-based exit to {position.token_name} with {hours} hours hold time")

        # Apply indicator-based exit if enabled
        if exit_strategies.get("indicator_based", {}).get("enabled", False):
            indicator_type = exit_strategies.get("indicator_based", {}).get("indicator", "rsi")
            params = exit_strategies.get("indicator_based", {}).get("params", {})

            position.set_indicator_based_exit(indicator_type, params)
            logger.info(f"Applied {indicator_type} indicator-based exit to {position.token_name}")


# Create a singleton instance
position_manager = PositionManager()
