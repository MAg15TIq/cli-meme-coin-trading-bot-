"""
Smart Order Management for the Solana Memecoin Trading Bot.
Implements intelligent stop-loss and take-profit strategies with dynamic adjustments.
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config import get_config_value
from src.utils.logging_utils import get_logger
from src.trading.advanced_orders import advanced_order_manager, AdvancedOrderType
from src.trading.jupiter_api import jupiter_api

# Get logger for this module
logger = get_logger(__name__)


class StopLossStrategy(Enum):
    """Stop-loss strategy types."""
    FIXED = "fixed"
    VOLATILITY_BASED = "volatility_based"
    TRAILING = "trailing"
    TIME_DECAY = "time_decay"
    ATR_BASED = "atr_based"


class TakeProfitStrategy(Enum):
    """Take-profit strategy types."""
    FIXED = "fixed"
    FIBONACCI = "fibonacci"
    VOLUME_WEIGHTED = "volume_weighted"
    MOMENTUM_BASED = "momentum_based"
    TIERED = "tiered"


@dataclass
class SmartStopLoss:
    """Smart stop-loss order configuration."""
    token_mint: str
    token_name: str
    position_size: float
    entry_price: float
    strategy: StopLossStrategy
    stop_price: float
    stop_percentage: float
    trailing_distance: Optional[float] = None
    volatility_multiplier: Optional[float] = None
    time_decay_rate: Optional[float] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SmartTakeProfit:
    """Smart take-profit order configuration."""
    token_mint: str
    token_name: str
    position_size: float
    entry_price: float
    strategy: TakeProfitStrategy
    target_levels: List[Dict[str, float]]  # [{"price": float, "quantity": float}]
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class VolatilityCalculator:
    """Calculate token volatility for dynamic adjustments."""

    def __init__(self):
        self.price_cache = {}
        self.cache_ttl = 300  # 5 minutes

    def get_recent_volatility(self, token_mint: str, days: int = 7) -> float:
        """
        Calculate recent volatility for a token.

        Args:
            token_mint: Token mint address
            days: Number of days to analyze

        Returns:
            Annualized volatility
        """
        try:
            # Get price history (simplified - would need real price data)
            prices = self._get_price_history(token_mint, days)

            if len(prices) < 2:
                return 0.3  # Default volatility

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]

            # Calculate volatility (annualized)
            volatility = np.std(returns) * np.sqrt(365)

            return max(min(volatility, 2.0), 0.05)  # Cap between 5% and 200%

        except Exception as e:
            logger.error(f"Error calculating volatility for {token_mint}: {e}")
            return 0.3  # Default volatility

    def _get_price_history(self, token_mint: str, days: int) -> List[float]:
        """Get historical price data for a token."""
        # This would integrate with a real price data provider
        # For now, return simulated data
        base_price = 1.0
        prices = []

        for i in range(days * 24):  # Hourly data
            # Simulate price movement
            change = np.random.normal(0, 0.02)  # 2% hourly volatility
            base_price *= (1 + change)
            prices.append(base_price)

        return prices


class TrendAnalyzer:
    """Analyze price trends for momentum-based strategies."""

    def __init__(self):
        self.trend_cache = {}

    def get_trend_strength(self, token_mint: str, timeframe_hours: int = 24) -> float:
        """
        Calculate trend strength for a token.

        Args:
            token_mint: Token mint address
            timeframe_hours: Timeframe for trend analysis

        Returns:
            Trend strength (-1 to 1, negative = downtrend, positive = uptrend)
        """
        try:
            # Get price data
            prices = self._get_recent_prices(token_mint, timeframe_hours)

            if len(prices) < 10:
                return 0.0

            # Calculate moving averages
            short_ma = np.mean(prices[-5:])  # Last 5 periods
            long_ma = np.mean(prices[-20:])  # Last 20 periods

            # Calculate trend strength
            trend_strength = (short_ma - long_ma) / long_ma

            return max(min(trend_strength, 1.0), -1.0)

        except Exception as e:
            logger.error(f"Error calculating trend strength for {token_mint}: {e}")
            return 0.0

    def _get_recent_prices(self, token_mint: str, hours: int) -> List[float]:
        """Get recent price data for trend analysis."""
        # This would integrate with real price data
        # For now, return simulated data
        return [1.0 + np.random.normal(0, 0.01) for _ in range(hours)]


class SmartStopLossManager:
    """Intelligent stop-loss management with dynamic adjustments."""

    def __init__(self):
        """Initialize the smart stop-loss manager."""
        self.volatility_calculator = VolatilityCalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.active_stop_losses = {}

        logger.info("Smart stop-loss manager initialized")

    def create_smart_stop_loss(self, token_mint: str, token_name: str, position_size: float,
                              entry_price: float, strategy: StopLossStrategy = StopLossStrategy.VOLATILITY_BASED,
                              custom_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a smart stop-loss order with dynamic adjustments.

        Args:
            token_mint: Token mint address
            token_name: Token name/symbol
            position_size: Position size
            entry_price: Entry price
            strategy: Stop-loss strategy to use
            custom_params: Custom parameters for the strategy

        Returns:
            Stop-loss order ID
        """
        try:
            if strategy == StopLossStrategy.VOLATILITY_BASED:
                stop_loss = self._create_volatility_stop_loss(
                    token_mint, token_name, position_size, entry_price, custom_params
                )
            elif strategy == StopLossStrategy.TRAILING:
                stop_loss = self._create_trailing_stop_loss(
                    token_mint, token_name, position_size, entry_price, custom_params
                )
            elif strategy == StopLossStrategy.TIME_DECAY:
                stop_loss = self._create_time_decay_stop_loss(
                    token_mint, token_name, position_size, entry_price, custom_params
                )
            elif strategy == StopLossStrategy.ATR_BASED:
                stop_loss = self._create_atr_stop_loss(
                    token_mint, token_name, position_size, entry_price, custom_params
                )
            else:
                stop_loss = self._create_fixed_stop_loss(
                    token_mint, token_name, position_size, entry_price, custom_params
                )

            # Create the actual order using advanced order manager
            order_id = advanced_order_manager.create_conditional_order(
                token_mint=token_mint,
                token_name=token_name,
                side="sell",
                total_amount=position_size,
                conditions=[{
                    "type": "price_below",
                    "value": stop_loss.stop_price
                }],
                target_price=stop_loss.stop_price
            )

            # Store the smart stop-loss configuration
            self.active_stop_losses[order_id] = stop_loss

            logger.info(f"Created {strategy.value} stop-loss for {token_name}: "
                       f"stop at {stop_loss.stop_price:.6f} ({stop_loss.stop_percentage:.2f}%)")

            return order_id

        except Exception as e:
            logger.error(f"Error creating smart stop-loss: {e}")
            raise

    def _create_volatility_stop_loss(self, token_mint: str, token_name: str,
                                   position_size: float, entry_price: float,
                                   custom_params: Optional[Dict[str, Any]]) -> SmartStopLoss:
        """Create volatility-adjusted stop-loss."""
        # Calculate recent volatility
        volatility = self.volatility_calculator.get_recent_volatility(token_mint, days=7)

        # Base stop-loss percentage
        base_stop_loss = custom_params.get('base_stop_loss', 0.05) if custom_params else 0.05

        # Adjust stop-loss based on volatility
        volatility_multiplier = min(max(volatility / 0.3, 0.5), 3.0)  # Cap between 0.5x and 3x
        adjusted_stop_loss = base_stop_loss * volatility_multiplier

        stop_price = entry_price * (1 - adjusted_stop_loss)

        return SmartStopLoss(
            token_mint=token_mint,
            token_name=token_name,
            position_size=position_size,
            entry_price=entry_price,
            strategy=StopLossStrategy.VOLATILITY_BASED,
            stop_price=stop_price,
            stop_percentage=adjusted_stop_loss * 100,
            volatility_multiplier=volatility_multiplier
        )

    def _create_trailing_stop_loss(self, token_mint: str, token_name: str,
                                 position_size: float, entry_price: float,
                                 custom_params: Optional[Dict[str, Any]]) -> SmartStopLoss:
        """Create trailing stop-loss."""
        trailing_distance = custom_params.get('trailing_distance', 0.05) if custom_params else 0.05

        # Get current price to set initial stop
        current_price = entry_price  # Would get real current price
        stop_price = current_price * (1 - trailing_distance)

        return SmartStopLoss(
            token_mint=token_mint,
            token_name=token_name,
            position_size=position_size,
            entry_price=entry_price,
            strategy=StopLossStrategy.TRAILING,
            stop_price=stop_price,
            stop_percentage=trailing_distance * 100,
            trailing_distance=trailing_distance
        )

    def _create_time_decay_stop_loss(self, token_mint: str, token_name: str,
                                   position_size: float, entry_price: float,
                                   custom_params: Optional[Dict[str, Any]]) -> SmartStopLoss:
        """Create time-decay stop-loss that tightens over time."""
        initial_stop = custom_params.get('initial_stop', 0.10) if custom_params else 0.10
        final_stop = custom_params.get('final_stop', 0.03) if custom_params else 0.03
        decay_days = custom_params.get('decay_days', 7) if custom_params else 7

        # Start with initial stop-loss
        stop_price = entry_price * (1 - initial_stop)

        return SmartStopLoss(
            token_mint=token_mint,
            token_name=token_name,
            position_size=position_size,
            entry_price=entry_price,
            strategy=StopLossStrategy.TIME_DECAY,
            stop_price=stop_price,
            stop_percentage=initial_stop * 100,
            time_decay_rate=(initial_stop - final_stop) / decay_days
        )

    def _create_atr_stop_loss(self, token_mint: str, token_name: str,
                            position_size: float, entry_price: float,
                            custom_params: Optional[Dict[str, Any]]) -> SmartStopLoss:
        """Create ATR-based stop-loss."""
        atr_multiplier = custom_params.get('atr_multiplier', 2.0) if custom_params else 2.0

        # Calculate ATR (simplified)
        volatility = self.volatility_calculator.get_recent_volatility(token_mint)
        atr = entry_price * volatility / np.sqrt(365)  # Daily ATR approximation

        stop_distance = atr * atr_multiplier
        stop_price = entry_price - stop_distance
        stop_percentage = (stop_distance / entry_price) * 100

        return SmartStopLoss(
            token_mint=token_mint,
            token_name=token_name,
            position_size=position_size,
            entry_price=entry_price,
            strategy=StopLossStrategy.ATR_BASED,
            stop_price=stop_price,
            stop_percentage=stop_percentage
        )

    def _create_fixed_stop_loss(self, token_mint: str, token_name: str,
                              position_size: float, entry_price: float,
                              custom_params: Optional[Dict[str, Any]]) -> SmartStopLoss:
        """Create fixed percentage stop-loss."""
        stop_percentage = custom_params.get('stop_percentage', 5.0) if custom_params else 5.0
        stop_price = entry_price * (1 - stop_percentage / 100)

        return SmartStopLoss(
            token_mint=token_mint,
            token_name=token_name,
            position_size=position_size,
            entry_price=entry_price,
            strategy=StopLossStrategy.FIXED,
            stop_price=stop_price,
            stop_percentage=stop_percentage
        )


class SmartTakeProfitManager:
    """Intelligent take-profit management with multiple strategies."""

    def __init__(self):
        """Initialize the smart take-profit manager."""
        self.fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618]
        self.trend_analyzer = TrendAnalyzer()
        self.active_take_profits = {}

        logger.info("Smart take-profit manager initialized")

    def create_tiered_take_profit(self, token_mint: str, token_name: str, position_size: float,
                                entry_price: float, strategy: TakeProfitStrategy = TakeProfitStrategy.FIBONACCI,
                                custom_params: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Create multiple take-profit levels with different strategies.

        Args:
            token_mint: Token mint address
            token_name: Token name/symbol
            position_size: Position size
            entry_price: Entry price
            strategy: Take-profit strategy to use
            custom_params: Custom parameters for the strategy

        Returns:
            List of take-profit order IDs
        """
        try:
            if strategy == TakeProfitStrategy.FIBONACCI:
                take_profit = self._create_fibonacci_take_profit(
                    token_mint, token_name, position_size, entry_price, custom_params
                )
            elif strategy == TakeProfitStrategy.VOLUME_WEIGHTED:
                take_profit = self._create_volume_weighted_take_profit(
                    token_mint, token_name, position_size, entry_price, custom_params
                )
            elif strategy == TakeProfitStrategy.MOMENTUM_BASED:
                take_profit = self._create_momentum_take_profit(
                    token_mint, token_name, position_size, entry_price, custom_params
                )
            elif strategy == TakeProfitStrategy.TIERED:
                take_profit = self._create_tiered_take_profit(
                    token_mint, token_name, position_size, entry_price, custom_params
                )
            else:
                take_profit = self._create_fixed_take_profit(
                    token_mint, token_name, position_size, entry_price, custom_params
                )

            # Create the actual orders
            order_ids = []
            for level in take_profit.target_levels:
                order_id = advanced_order_manager.create_conditional_order(
                    token_mint=token_mint,
                    token_name=token_name,
                    side="sell",
                    total_amount=level["quantity"],
                    conditions=[{
                        "type": "price_above",
                        "value": level["price"]
                    }],
                    target_price=level["price"]
                )
                order_ids.append(order_id)

            # Store the smart take-profit configuration
            for order_id in order_ids:
                self.active_take_profits[order_id] = take_profit

            logger.info(f"Created {strategy.value} take-profit for {token_name}: "
                       f"{len(take_profit.target_levels)} levels")

            return order_ids

        except Exception as e:
            logger.error(f"Error creating smart take-profit: {e}")
            raise

    def _create_fibonacci_take_profit(self, token_mint: str, token_name: str,
                                    position_size: float, entry_price: float,
                                    custom_params: Optional[Dict[str, Any]]) -> SmartTakeProfit:
        """Create Fibonacci-based take-profit levels."""
        max_profit_target = custom_params.get('max_profit_target', 2.0) if custom_params else 2.0

        target_levels = []
        remaining_position = position_size

        for i, fib_level in enumerate(self.fibonacci_levels[:5]):  # First 5 levels
            # Calculate target price
            profit_multiplier = fib_level * max_profit_target
            target_price = entry_price * (1 + profit_multiplier)

            # Calculate quantity to sell (increasing amounts at higher levels)
            sell_percentage = 0.15 + (i * 0.05)  # 15%, 20%, 25%, 30%, 35%
            quantity = min(remaining_position * sell_percentage, remaining_position)

            if quantity > 0:
                target_levels.append({
                    "price": target_price,
                    "quantity": quantity,
                    "level": fib_level,
                    "profit_target": profit_multiplier
                })
                remaining_position -= quantity

        return SmartTakeProfit(
            token_mint=token_mint,
            token_name=token_name,
            position_size=position_size,
            entry_price=entry_price,
            strategy=TakeProfitStrategy.FIBONACCI,
            target_levels=target_levels
        )

    def _create_volume_weighted_take_profit(self, token_mint: str, token_name: str,
                                          position_size: float, entry_price: float,
                                          custom_params: Optional[Dict[str, Any]]) -> SmartTakeProfit:
        """Create volume-weighted take-profit levels."""
        # This would analyze volume patterns to set optimal exit levels
        # For now, create simplified levels
        target_levels = [
            {"price": entry_price * 1.25, "quantity": position_size * 0.3},
            {"price": entry_price * 1.50, "quantity": position_size * 0.4},
            {"price": entry_price * 2.00, "quantity": position_size * 0.3}
        ]

        return SmartTakeProfit(
            token_mint=token_mint,
            token_name=token_name,
            position_size=position_size,
            entry_price=entry_price,
            strategy=TakeProfitStrategy.VOLUME_WEIGHTED,
            target_levels=target_levels
        )

    def _create_momentum_take_profit(self, token_mint: str, token_name: str,
                                   position_size: float, entry_price: float,
                                   custom_params: Optional[Dict[str, Any]]) -> SmartTakeProfit:
        """Create momentum-based take-profit levels."""
        trend_strength = self.trend_analyzer.get_trend_strength(token_mint)

        # Adjust targets based on momentum
        if trend_strength > 0.5:  # Strong uptrend
            multipliers = [1.3, 1.8, 2.5]
            quantities = [0.25, 0.35, 0.4]
        elif trend_strength > 0:  # Weak uptrend
            multipliers = [1.2, 1.5, 2.0]
            quantities = [0.3, 0.4, 0.3]
        else:  # No trend or downtrend
            multipliers = [1.15, 1.3, 1.5]
            quantities = [0.4, 0.4, 0.2]

        target_levels = []
        for i, (mult, qty) in enumerate(zip(multipliers, quantities)):
            target_levels.append({
                "price": entry_price * mult,
                "quantity": position_size * qty,
                "momentum_adjusted": True
            })

        return SmartTakeProfit(
            token_mint=token_mint,
            token_name=token_name,
            position_size=position_size,
            entry_price=entry_price,
            strategy=TakeProfitStrategy.MOMENTUM_BASED,
            target_levels=target_levels
        )

    def _create_tiered_take_profit(self, token_mint: str, token_name: str,
                                 position_size: float, entry_price: float,
                                 custom_params: Optional[Dict[str, Any]]) -> SmartTakeProfit:
        """Create simple tiered take-profit levels."""
        num_tiers = custom_params.get('num_tiers', 4) if custom_params else 4
        max_profit = custom_params.get('max_profit', 3.0) if custom_params else 3.0

        target_levels = []
        tier_size = position_size / num_tiers

        for i in range(num_tiers):
            profit_mult = 1 + ((i + 1) / num_tiers) * max_profit
            target_price = entry_price * profit_mult

            target_levels.append({
                "price": target_price,
                "quantity": tier_size,
                "tier": i + 1
            })

        return SmartTakeProfit(
            token_mint=token_mint,
            token_name=token_name,
            position_size=position_size,
            entry_price=entry_price,
            strategy=TakeProfitStrategy.TIERED,
            target_levels=target_levels
        )

    def _create_fixed_take_profit(self, token_mint: str, token_name: str,
                                position_size: float, entry_price: float,
                                custom_params: Optional[Dict[str, Any]]) -> SmartTakeProfit:
        """Create fixed percentage take-profit."""
        profit_percentage = custom_params.get('profit_percentage', 50.0) if custom_params else 50.0
        target_price = entry_price * (1 + profit_percentage / 100)

        target_levels = [{
            "price": target_price,
            "quantity": position_size,
            "profit_percentage": profit_percentage
        }]

        return SmartTakeProfit(
            token_mint=token_mint,
            token_name=token_name,
            position_size=position_size,
            entry_price=entry_price,
            strategy=TakeProfitStrategy.FIXED,
            target_levels=target_levels
        )


# Create singleton instances
smart_stop_loss_manager = SmartStopLossManager()
smart_take_profit_manager = SmartTakeProfitManager()
