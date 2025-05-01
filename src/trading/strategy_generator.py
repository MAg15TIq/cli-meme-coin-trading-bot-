"""
AI-powered trading strategy generator for the Solana Memecoin Trading Bot.
Generates and evaluates trading strategies based on market conditions.
"""

import json
import logging
import time
import threading
import random
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta

from config import get_config_value, update_config
from src.trading.jupiter_api import jupiter_api
from src.trading.position_manager import position_manager
from src.trading.sentiment_analysis import sentiment_analyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Strategy:
    """Represents a trading strategy."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any],
                 entry_conditions: List[Dict[str, Any]], exit_conditions: List[Dict[str, Any]],
                 risk_level: str, token_filters: Dict[str, Any], created_at: datetime):
        """
        Initialize a strategy.

        Args:
            name: Strategy name
            description: Strategy description
            parameters: Strategy parameters
            entry_conditions: Conditions for entering a position
            exit_conditions: Conditions for exiting a position
            risk_level: Risk level (low, medium, high)
            token_filters: Filters for selecting tokens
            created_at: Creation timestamp
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.entry_conditions = entry_conditions
        self.exit_conditions = exit_conditions
        self.risk_level = risk_level
        self.token_filters = token_filters
        self.created_at = created_at
        self.performance: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the strategy to a dictionary.

        Returns:
            The strategy as a dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "risk_level": self.risk_level,
            "token_filters": self.token_filters,
            "created_at": self.created_at.isoformat(),
            "performance": self.performance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Strategy':
        """
        Create a strategy from a dictionary.

        Args:
            data: The strategy data

        Returns:
            The strategy
        """
        strategy = cls(
            name=data["name"],
            description=data["description"],
            parameters=data["parameters"],
            entry_conditions=data["entry_conditions"],
            exit_conditions=data["exit_conditions"],
            risk_level=data["risk_level"],
            token_filters=data["token_filters"],
            created_at=datetime.fromisoformat(data["created_at"])
        )
        strategy.performance = data.get("performance", {})
        return strategy


class StrategyGenerator:
    """Generator for AI-powered trading strategies."""

    def __init__(self):
        """Initialize the strategy generator."""
        self.strategies: Dict[str, Strategy] = {}
        self.active_strategy: Optional[str] = None

        # Load strategies
        self._load_strategies()

    def _load_strategies(self) -> None:
        """Load strategies from config."""
        strategies_data = get_config_value("trading_strategies", {})
        active_strategy = get_config_value("active_trading_strategy", None)

        for strategy_id, strategy_data in strategies_data.items():
            try:
                self.strategies[strategy_id] = Strategy.from_dict(strategy_data)
            except Exception as e:
                logger.error(f"Error loading strategy {strategy_id}: {e}")

        self.active_strategy = active_strategy
        logger.info(f"Loaded {len(self.strategies)} strategies")

    def _save_strategies(self) -> None:
        """Save strategies to config."""
        strategies_data = {
            strategy_id: strategy.to_dict()
            for strategy_id, strategy in self.strategies.items()
        }

        update_config("trading_strategies", strategies_data)
        update_config("active_trading_strategy", self.active_strategy)

        logger.info(f"Saved {len(self.strategies)} strategies")

    def generate_strategy(self, risk_level: str = "medium") -> Strategy:
        """
        Generate a new trading strategy.

        Args:
            risk_level: Risk level (low, medium, high)

        Returns:
            The generated strategy
        """
        # This is a simplified implementation
        # In a real implementation, we would use more sophisticated methods

        # Generate a unique ID
        strategy_id = f"strategy_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Generate strategy name
        adjectives = ["Aggressive", "Balanced", "Conservative", "Dynamic", "Efficient"]
        nouns = ["Momentum", "Trend", "Volatility", "Sentiment", "Volume"]
        strategy_name = f"{random.choice(adjectives)} {random.choice(nouns)} Strategy"

        # Generate description
        descriptions = {
            "low": "A conservative strategy focusing on established tokens with strong fundamentals. Prioritizes capital preservation with modest returns.",
            "medium": "A balanced strategy targeting moderate growth with reasonable risk. Combines trend following with fundamental analysis.",
            "high": "An aggressive strategy seeking high returns through early-stage tokens and momentum trading. Higher risk with potential for significant gains."
        }
        description = descriptions.get(risk_level, descriptions["medium"])

        # Generate parameters based on risk level
        parameters = self._generate_parameters(risk_level)

        # Generate entry conditions
        entry_conditions = self._generate_entry_conditions(risk_level)

        # Generate exit conditions
        exit_conditions = self._generate_exit_conditions(risk_level)

        # Generate token filters
        token_filters = self._generate_token_filters(risk_level)

        # Create strategy
        strategy = Strategy(
            name=strategy_name,
            description=description,
            parameters=parameters,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_level=risk_level,
            token_filters=token_filters,
            created_at=datetime.now()
        )

        # Add to strategies
        self.strategies[strategy_id] = strategy

        # Save strategies
        self._save_strategies()

        logger.info(f"Generated new strategy: {strategy_name} (ID: {strategy_id})")
        return strategy

    def _generate_parameters(self, risk_level: str) -> Dict[str, Any]:
        """
        Generate strategy parameters based on risk level.

        Args:
            risk_level: Risk level (low, medium, high)

        Returns:
            Strategy parameters
        """
        if risk_level == "low":
            return {
                "position_size_percentage": random.uniform(1.0, 5.0),
                "max_positions": random.randint(3, 5),
                "min_liquidity_sol": random.uniform(50.0, 100.0),
                "max_slippage_bps": random.randint(30, 50),
                "min_market_cap_usd": random.uniform(1000000, 5000000),
                "min_holder_count": random.randint(500, 1000)
            }
        elif risk_level == "medium":
            return {
                "position_size_percentage": random.uniform(5.0, 10.0),
                "max_positions": random.randint(5, 8),
                "min_liquidity_sol": random.uniform(10.0, 50.0),
                "max_slippage_bps": random.randint(50, 100),
                "min_market_cap_usd": random.uniform(100000, 1000000),
                "min_holder_count": random.randint(100, 500)
            }
        else:  # high
            return {
                "position_size_percentage": random.uniform(10.0, 20.0),
                "max_positions": random.randint(8, 12),
                "min_liquidity_sol": random.uniform(1.0, 10.0),
                "max_slippage_bps": random.randint(100, 200),
                "min_market_cap_usd": random.uniform(10000, 100000),
                "min_holder_count": random.randint(10, 100)
            }

    def _generate_entry_conditions(self, risk_level: str) -> List[Dict[str, Any]]:
        """
        Generate entry conditions based on risk level.

        Args:
            risk_level: Risk level (low, medium, high)

        Returns:
            List of entry conditions
        """
        conditions = []

        # Price conditions
        if risk_level == "low":
            conditions.append({
                "type": "price_above_ma",
                "parameters": {
                    "ma_period": random.randint(20, 50),
                    "min_percentage": random.uniform(1.0, 5.0)
                },
                "description": f"Price is above the {random.randint(20, 50)}-period moving average by at least {random.uniform(1.0, 5.0):.1f}%"
            })
        elif risk_level == "medium":
            conditions.append({
                "type": "price_momentum",
                "parameters": {
                    "lookback_period": random.randint(5, 20),
                    "min_percentage": random.uniform(5.0, 15.0)
                },
                "description": f"Price has increased by at least {random.uniform(5.0, 15.0):.1f}% in the last {random.randint(5, 20)} periods"
            })
        else:  # high
            conditions.append({
                "type": "price_breakout",
                "parameters": {
                    "lookback_period": random.randint(3, 10),
                    "min_percentage": random.uniform(10.0, 30.0)
                },
                "description": f"Price has broken out by at least {random.uniform(10.0, 30.0):.1f}% from the {random.randint(3, 10)}-period high"
            })

        # Volume conditions
        conditions.append({
            "type": "volume_increase",
            "parameters": {
                "lookback_period": random.randint(3, 10),
                "min_percentage": random.uniform(50.0, 200.0)
            },
            "description": f"Volume has increased by at least {random.uniform(50.0, 200.0):.1f}% compared to the {random.randint(3, 10)}-period average"
        })

        # Sentiment conditions (for medium and high risk)
        if risk_level in ["medium", "high"]:
            conditions.append({
                "type": "positive_sentiment",
                "parameters": {
                    "min_score": random.uniform(0.2, 0.5),
                    "min_mentions": random.randint(10, 100)
                },
                "description": f"Social sentiment is positive (score > {random.uniform(0.2, 0.5):.1f}) with at least {random.randint(10, 100)} mentions"
            })

        return conditions

    def _generate_exit_conditions(self, risk_level: str) -> List[Dict[str, Any]]:
        """
        Generate exit conditions based on risk level.

        Args:
            risk_level: Risk level (low, medium, high)

        Returns:
            List of exit conditions
        """
        conditions = []

        # Stop loss
        if risk_level == "low":
            stop_loss = random.uniform(3.0, 8.0)
        elif risk_level == "medium":
            stop_loss = random.uniform(8.0, 15.0)
        else:  # high
            stop_loss = random.uniform(15.0, 25.0)

        conditions.append({
            "type": "stop_loss",
            "parameters": {
                "percentage": stop_loss
            },
            "description": f"Exit if price drops by {stop_loss:.1f}% from entry"
        })

        # Take profit (multi-tiered for medium and high risk)
        if risk_level == "low":
            conditions.append({
                "type": "take_profit",
                "parameters": {
                    "percentage": random.uniform(10.0, 20.0),
                    "position_percentage": 100.0
                },
                "description": f"Exit entire position if price increases by {random.uniform(10.0, 20.0):.1f}% from entry"
            })
        else:
            # First tier
            first_tp = random.uniform(15.0, 30.0)
            first_tp_pct = random.uniform(30.0, 50.0)
            conditions.append({
                "type": "take_profit",
                "parameters": {
                    "percentage": first_tp,
                    "position_percentage": first_tp_pct
                },
                "description": f"Exit {first_tp_pct:.1f}% of position if price increases by {first_tp:.1f}% from entry"
            })

            # Second tier
            second_tp = first_tp + random.uniform(20.0, 50.0)
            second_tp_pct = random.uniform(30.0, 50.0)
            conditions.append({
                "type": "take_profit",
                "parameters": {
                    "percentage": second_tp,
                    "position_percentage": second_tp_pct
                },
                "description": f"Exit {second_tp_pct:.1f}% of position if price increases by {second_tp:.1f}% from entry"
            })

            # Final tier (remaining position)
            final_tp = second_tp + random.uniform(50.0, 100.0)
            conditions.append({
                "type": "take_profit",
                "parameters": {
                    "percentage": final_tp,
                    "position_percentage": 100.0 - first_tp_pct - second_tp_pct
                },
                "description": f"Exit remaining position if price increases by {final_tp:.1f}% from entry"
            })

        # Time-based exit (for medium and high risk)
        if risk_level in ["medium", "high"]:
            max_hold_days = random.randint(3, 14)
            conditions.append({
                "type": "time_exit",
                "parameters": {
                    "max_days": max_hold_days
                },
                "description": f"Exit if position has been held for {max_hold_days} days"
            })

        # Sentiment-based exit (for high risk)
        if risk_level == "high":
            conditions.append({
                "type": "sentiment_exit",
                "parameters": {
                    "min_negative_score": random.uniform(-0.5, -0.2)
                },
                "description": f"Exit if sentiment turns negative (score < {random.uniform(-0.5, -0.2):.1f})"
            })

        return conditions

    def _generate_token_filters(self, risk_level: str) -> Dict[str, Any]:
        """
        Generate token filters based on risk level.

        Args:
            risk_level: Risk level (low, medium, high)

        Returns:
            Token filters
        """
        if risk_level == "low":
            return {
                "min_market_cap_usd": random.uniform(1000000, 5000000),
                "min_liquidity_sol": random.uniform(50.0, 100.0),
                "min_holder_count": random.randint(500, 1000),
                "min_age_days": random.randint(30, 90),
                "require_website": True,
                "require_social_media": True
            }
        elif risk_level == "medium":
            return {
                "min_market_cap_usd": random.uniform(100000, 1000000),
                "min_liquidity_sol": random.uniform(10.0, 50.0),
                "min_holder_count": random.randint(100, 500),
                "min_age_days": random.randint(7, 30),
                "require_website": random.choice([True, False]),
                "require_social_media": True
            }
        else:  # high
            return {
                "min_market_cap_usd": random.uniform(10000, 100000),
                "min_liquidity_sol": random.uniform(1.0, 10.0),
                "min_holder_count": random.randint(10, 100),
                "min_age_days": random.randint(1, 7),
                "require_website": False,
                "require_social_media": random.choice([True, False])
            }

    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """
        Get a strategy by ID.

        Args:
            strategy_id: The strategy ID

        Returns:
            The strategy, or None if not found
        """
        return self.strategies.get(strategy_id)

    def list_strategies(self) -> List[Dict[str, Any]]:
        """
        List all strategies.

        Returns:
            List of strategy information dictionaries
        """
        return [
            {
                "id": strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "risk_level": strategy.risk_level,
                "created_at": strategy.created_at.isoformat(),
                "active": strategy_id == self.active_strategy,
                "performance": strategy.performance
            }
            for strategy_id, strategy in self.strategies.items()
        ]

    def set_active_strategy(self, strategy_id: Optional[str]) -> bool:
        """
        Set the active strategy.

        Args:
            strategy_id: The strategy ID, or None to deactivate

        Returns:
            True if successful, False otherwise
        """
        if strategy_id is not None and strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return False

        self.active_strategy = strategy_id
        self._save_strategies()

        if strategy_id:
            logger.info(f"Set active strategy to {strategy_id}")
        else:
            logger.info("Deactivated strategy")

        return True

    def delete_strategy(self, strategy_id: str) -> bool:
        """
        Delete a strategy.

        Args:
            strategy_id: The strategy ID

        Returns:
            True if successful, False otherwise
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return False

        # Remove from active strategy if it's the active one
        if strategy_id == self.active_strategy:
            self.active_strategy = None

        # Remove from strategies
        del self.strategies[strategy_id]

        # Save strategies
        self._save_strategies()

        logger.info(f"Deleted strategy {strategy_id}")
        return True

    def update_strategy_performance(self, strategy_id: str, performance: Dict[str, Any]) -> bool:
        """
        Update strategy performance metrics.

        Args:
            strategy_id: The strategy ID
            performance: Performance metrics

        Returns:
            True if successful, False otherwise
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return False

        # Update performance
        self.strategies[strategy_id].performance = performance

        # Save strategies
        self._save_strategies()

        logger.info(f"Updated performance for strategy {strategy_id}")
        return True

    def import_strategy(self, strategy_data: Dict[str, Any]) -> Optional[str]:
        """
        Import a strategy from external data.

        Args:
            strategy_data: The strategy data to import

        Returns:
            The imported strategy ID if successful, None otherwise
        """
        try:
            # Validate strategy data
            required_fields = ["name", "description", "parameters", "entry_conditions",
                              "exit_conditions", "risk_level", "token_filters"]

            for field in required_fields:
                if field not in strategy_data:
                    logger.warning(f"Invalid strategy data: missing {field}")
                    return None

            # Generate a unique ID
            strategy_id = f"imported_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Create strategy object
            if "created_at" in strategy_data:
                created_at = datetime.fromisoformat(strategy_data["created_at"])
            else:
                created_at = datetime.now()

            strategy = Strategy(
                name=strategy_data["name"],
                description=strategy_data["description"],
                parameters=strategy_data["parameters"],
                entry_conditions=strategy_data["entry_conditions"],
                exit_conditions=strategy_data["exit_conditions"],
                risk_level=strategy_data["risk_level"],
                token_filters=strategy_data["token_filters"],
                created_at=created_at
            )

            # Add performance data if available
            if "performance" in strategy_data:
                strategy.performance = strategy_data["performance"]

            # Add to strategies
            self.strategies[strategy_id] = strategy

            # Save strategies
            self._save_strategies()

            logger.info(f"Imported strategy: {strategy.name} (ID: {strategy_id})")
            return strategy_id
        except Exception as e:
            logger.error(f"Error importing strategy: {e}")
            return None

    def get_strategy_data(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the complete data for a strategy.

        Args:
            strategy_id: The strategy ID

        Returns:
            The strategy data, or None if not found
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return None

        return self.strategies[strategy_id].to_dict()

    def backtest_strategy(self, strategy_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Backtest a strategy over a historical period.

        Args:
            strategy_id: The strategy ID
            start_date: Start date for backtesting
            end_date: End date for backtesting

        Returns:
            Backtest results
        """
        # This is a simplified implementation
        # In a real implementation, we would use historical data and simulate trades

        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return {"error": "Strategy not found"}

        strategy = self.strategies[strategy_id]

        # Simulate random performance based on risk level
        if strategy.risk_level == "low":
            roi = random.uniform(5.0, 20.0)
            max_drawdown = random.uniform(2.0, 10.0)
            win_rate = random.uniform(60.0, 80.0)
        elif strategy.risk_level == "medium":
            roi = random.uniform(20.0, 50.0)
            max_drawdown = random.uniform(10.0, 25.0)
            win_rate = random.uniform(50.0, 70.0)
        else:  # high
            roi = random.uniform(50.0, 150.0)
            max_drawdown = random.uniform(25.0, 50.0)
            win_rate = random.uniform(40.0, 60.0)

        # Create backtest results
        results = {
            "strategy_id": strategy_id,
            "strategy_name": strategy.name,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "roi_percentage": roi,
            "max_drawdown_percentage": max_drawdown,
            "win_rate_percentage": win_rate,
            "total_trades": random.randint(10, 50),
            "risk_adjusted_return": roi / max_drawdown,
            "monthly_returns": self._generate_monthly_returns(start_date, end_date, roi),
            "backtest_date": datetime.now().isoformat()
        }

        # Update strategy performance
        self.update_strategy_performance(strategy_id, results)

        logger.info(f"Backtested strategy {strategy_id}: ROI {roi:.1f}%, Max Drawdown {max_drawdown:.1f}%")
        return results

    def _generate_monthly_returns(self, start_date: datetime, end_date: datetime, total_roi: float) -> Dict[str, float]:
        """
        Generate simulated monthly returns.

        Args:
            start_date: Start date
            end_date: End date
            total_roi: Total ROI percentage

        Returns:
            Dictionary of monthly returns
        """
        # Calculate number of months
        months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
        months = max(1, months)

        # Generate random monthly returns that sum to total_roi
        monthly_returns = {}

        current_date = start_date.replace(day=1)
        remaining_roi = total_roi

        for _ in range(months):
            # For the last month, use the remaining ROI
            if _ == months - 1:
                month_roi = remaining_roi
            else:
                # Generate a random portion of the remaining ROI
                portion = random.uniform(0.0, 1.0)
                month_roi = remaining_roi * portion
                remaining_roi -= month_roi

            # Add some randomness (positive or negative)
            month_roi += random.uniform(-5.0, 5.0)

            # Add to monthly returns
            month_key = current_date.strftime("%Y-%m")
            monthly_returns[month_key] = month_roi

            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)

        return monthly_returns

    def apply_strategy_to_position(self, strategy_id: str, token_mint: str) -> Dict[str, Any]:
        """
        Apply a strategy to a specific token position.

        Args:
            strategy_id: The strategy ID
            token_mint: The token mint address

        Returns:
            Dictionary with strategy application results
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return {"error": "Strategy not found"}

        strategy = self.strategies[strategy_id]

        # Get position
        position = position_manager.get_position(token_mint)
        if not position:
            logger.warning(f"Position not found for token {token_mint}")
            return {"error": "Position not found"}

        # Apply exit conditions to set stop-loss and take-profit levels
        stop_loss = None
        take_profit_levels = []

        for condition in strategy.exit_conditions:
            if condition["type"] == "stop_loss":
                stop_loss_pct = condition["parameters"]["percentage"]
                stop_loss = position.entry_price * (1 - stop_loss_pct / 100)

            elif condition["type"] == "take_profit":
                tp_pct = condition["parameters"]["percentage"]
                position_pct = condition["parameters"]["position_percentage"]

                take_profit_price = position.entry_price * (1 + tp_pct / 100)
                take_profit_levels.append({
                    "price": take_profit_price,
                    "percentage": tp_pct,
                    "position_percentage": position_pct
                })

        # Update position
        if stop_loss is not None:
            position.stop_loss = stop_loss

        # Clear existing take-profit levels
        position.clear_take_profit_levels()

        # Add new take-profit levels
        for tp in take_profit_levels:
            position.add_take_profit_level(
                price=tp["price"],
                percentage=tp["percentage"],
                sell_percentage=tp["position_percentage"]
            )

        # Save positions
        position_manager.save_positions()

        logger.info(f"Applied strategy {strategy_id} to position {token_mint}")

        return {
            "strategy_id": strategy_id,
            "strategy_name": strategy.name,
            "token_mint": token_mint,
            "token_name": position.token_name,
            "stop_loss": stop_loss,
            "take_profit_levels": take_profit_levels,
            "applied_at": datetime.now().isoformat()
        }


# Create a singleton instance
strategy_generator = StrategyGenerator()
