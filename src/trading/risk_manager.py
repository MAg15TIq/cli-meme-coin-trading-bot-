"""
Risk management module for the Solana Memecoin Trading Bot.
Provides comprehensive risk assessment, position sizing, and portfolio management.
Includes refinement capabilities based on real-world usage.
"""

import json
import logging
import threading
import time
import statistics
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.trading.token_analytics import token_analytics
from src.trading.position_manager import RiskLevel
from src.wallet.wallet import wallet_manager

# Get logger for this module
logger = get_logger(__name__)


class RiskProfile:
    """Represents a risk profile with specific parameters."""

    def __init__(self, name: str, max_allocation_percent: float, max_position_percent: float,
                 stop_loss_percent: float, max_drawdown_percent: float):
        """
        Initialize a risk profile.

        Args:
            name: The name of the profile (conservative, moderate, aggressive)
            max_allocation_percent: Maximum percentage of portfolio to allocate to risky assets
            max_position_percent: Maximum percentage of portfolio for a single position
            stop_loss_percent: Default stop-loss percentage for positions
            max_drawdown_percent: Maximum acceptable drawdown before reducing exposure
        """
        self.name = name
        self.max_allocation_percent = max_allocation_percent
        self.max_position_percent = max_position_percent
        self.stop_loss_percent = stop_loss_percent
        self.max_drawdown_percent = max_drawdown_percent

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "max_allocation_percent": self.max_allocation_percent,
            "max_position_percent": self.max_position_percent,
            "stop_loss_percent": self.stop_loss_percent,
            "max_drawdown_percent": self.max_drawdown_percent
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskProfile':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            max_allocation_percent=data["max_allocation_percent"],
            max_position_percent=data["max_position_percent"],
            stop_loss_percent=data["stop_loss_percent"],
            max_drawdown_percent=data["max_drawdown_percent"]
        )


class RiskManager:
    """Manager for risk assessment and management."""

    def __init__(self):
        """Initialize the risk manager."""
        self.enabled = get_config_value("risk_management_enabled", True)

        # Load risk profiles
        self.risk_profiles = {
            "conservative": RiskProfile(
                name="conservative",
                max_allocation_percent=float(get_config_value("conservative_max_allocation_percent", "20.0")),
                max_position_percent=float(get_config_value("conservative_max_position_percent", "2.0")),
                stop_loss_percent=float(get_config_value("conservative_stop_loss_percent", "5.0")),
                max_drawdown_percent=float(get_config_value("conservative_max_drawdown_percent", "10.0"))
            ),
            "moderate": RiskProfile(
                name="moderate",
                max_allocation_percent=float(get_config_value("moderate_max_allocation_percent", "40.0")),
                max_position_percent=float(get_config_value("moderate_max_position_percent", "5.0")),
                stop_loss_percent=float(get_config_value("moderate_stop_loss_percent", "10.0")),
                max_drawdown_percent=float(get_config_value("moderate_max_drawdown_percent", "20.0"))
            ),
            "aggressive": RiskProfile(
                name="aggressive",
                max_allocation_percent=float(get_config_value("aggressive_max_allocation_percent", "60.0")),
                max_position_percent=float(get_config_value("aggressive_max_position_percent", "10.0")),
                stop_loss_percent=float(get_config_value("aggressive_stop_loss_percent", "15.0")),
                max_drawdown_percent=float(get_config_value("aggressive_max_drawdown_percent", "30.0"))
            )
        }

        # Current user profile
        self.current_profile_name = get_config_value("risk_profile", "moderate")

        # Risk allocation limits by token risk level
        self.risk_allocation_limits = {
            RiskLevel.LOW.value: float(get_config_value("max_low_risk_allocation_percent", "60.0")),
            RiskLevel.MEDIUM.value: float(get_config_value("max_medium_risk_allocation_percent", "30.0")),
            RiskLevel.HIGH.value: float(get_config_value("max_high_risk_allocation_percent", "10.0")),
            RiskLevel.VERY_HIGH.value: float(get_config_value("max_very_high_risk_allocation_percent", "5.0"))
        }

        # Portfolio metrics
        self.portfolio_value_sol = 0.0
        self.portfolio_allocation = {
            RiskLevel.LOW.value: 0.0,
            RiskLevel.MEDIUM.value: 0.0,
            RiskLevel.HIGH.value: 0.0,
            RiskLevel.VERY_HIGH.value: 0.0
        }
        self.portfolio_drawdown = 0.0
        self.max_portfolio_value_sol = 0.0

        # Risk metrics cache
        self.token_risk_cache = {}
        self.token_risk_cache_ttl = int(get_config_value("token_risk_cache_ttl", "3600"))  # 1 hour

        logger.info(f"Initialized risk manager with {self.current_profile_name} profile")

    def get_current_profile(self) -> RiskProfile:
        """
        Get the current risk profile.

        Returns:
            The current risk profile
        """
        return self.risk_profiles.get(self.current_profile_name, self.risk_profiles["moderate"])

    def set_risk_profile(self, profile_name: str) -> bool:
        """
        Set the current risk profile.

        Args:
            profile_name: The name of the profile to set

        Returns:
            True if successful, False otherwise
        """
        if profile_name in self.risk_profiles:
            self.current_profile_name = profile_name
            update_config("risk_profile", profile_name)
            logger.info(f"Set risk profile to {profile_name}")
            return True
        else:
            logger.error(f"Unknown risk profile: {profile_name}")
            return False

    def get_token_risk_level(self, token_mint: str) -> RiskLevel:
        """
        Get the risk level for a token.

        Args:
            token_mint: The token mint address

        Returns:
            The risk level
        """
        # Check cache first
        if token_mint in self.token_risk_cache:
            cache_entry = self.token_risk_cache[token_mint]
            if datetime.now().timestamp() - cache_entry["timestamp"] < self.token_risk_cache_ttl:
                return RiskLevel(cache_entry["risk_level"])

        # Get risk level from token analytics
        risk_level_str = token_analytics.get_risk_level(token_mint)

        try:
            risk_level = RiskLevel(risk_level_str)
        except ValueError:
            logger.warning(f"Invalid risk level from token analytics: {risk_level_str}, defaulting to MEDIUM")
            risk_level = RiskLevel.MEDIUM

        # Update cache
        self.token_risk_cache[token_mint] = {
            "risk_level": risk_level.value,
            "timestamp": datetime.now().timestamp()
        }

        return risk_level

    def calculate_position_size(self, token_mint: str, max_amount_sol: float = None) -> Dict[str, Any]:
        """
        Calculate the recommended position size for a token.

        Args:
            token_mint: The token mint address
            max_amount_sol: Optional maximum amount in SOL

        Returns:
            Position size recommendation
        """
        if not self.enabled:
            logger.warning("Risk management is disabled")
            return {
                "position_size_sol": max_amount_sol or 0.1,
                "risk_level": RiskLevel.MEDIUM.value,
                "stop_loss_percentage": 10.0
            }

        try:
            # Update portfolio value
            self.update_portfolio_metrics()

            # Get token risk level
            risk_level = self.get_token_risk_level(token_mint)

            # Get current profile
            profile = self.get_current_profile()

            # Calculate position size based on portfolio value and risk level
            max_position_percent = profile.max_position_percent

            # Adjust based on token risk level
            if risk_level == RiskLevel.LOW:
                position_percent = max_position_percent
            elif risk_level == RiskLevel.MEDIUM:
                position_percent = max_position_percent * 0.75
            elif risk_level == RiskLevel.HIGH:
                position_percent = max_position_percent * 0.5
            else:  # VERY_HIGH
                position_percent = max_position_percent * 0.25

            # Calculate position size in SOL
            position_size_sol = self.portfolio_value_sol * (position_percent / 100)

            # Check if we're exceeding risk allocation limits
            current_allocation = self.portfolio_allocation.get(risk_level.value, 0.0)
            max_allocation = self.risk_allocation_limits.get(risk_level.value, 100.0)

            # Calculate remaining allocation
            remaining_allocation_sol = (self.portfolio_value_sol * (max_allocation / 100)) - current_allocation

            # Adjust position size if needed
            if position_size_sol > remaining_allocation_sol:
                position_size_sol = remaining_allocation_sol
                logger.info(f"Position size adjusted due to risk allocation limits: {position_size_sol} SOL")

            # Apply max amount if specified
            if max_amount_sol is not None and position_size_sol > max_amount_sol:
                position_size_sol = max_amount_sol

            # Ensure position size is positive
            position_size_sol = max(position_size_sol, 0.01)

            # Calculate stop-loss percentage based on risk level
            if risk_level == RiskLevel.LOW:
                stop_loss_percentage = profile.stop_loss_percent
            elif risk_level == RiskLevel.MEDIUM:
                stop_loss_percentage = profile.stop_loss_percent * 1.2
            elif risk_level == RiskLevel.HIGH:
                stop_loss_percentage = profile.stop_loss_percent * 1.5
            else:  # VERY_HIGH
                stop_loss_percentage = profile.stop_loss_percent * 2.0

            return {
                "position_size_sol": position_size_sol,
                "position_size_percentage": position_percent,
                "risk_level": risk_level.value,
                "stop_loss_percentage": stop_loss_percentage,
                "portfolio_value_sol": self.portfolio_value_sol,
                "current_risk_allocation": {
                    "low": self.portfolio_allocation[RiskLevel.LOW.value],
                    "medium": self.portfolio_allocation[RiskLevel.MEDIUM.value],
                    "high": self.portfolio_allocation[RiskLevel.HIGH.value],
                    "very_high": self.portfolio_allocation[RiskLevel.VERY_HIGH.value]
                }
            }
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                "position_size_sol": max_amount_sol or 0.1,
                "risk_level": RiskLevel.MEDIUM.value,
                "stop_loss_percentage": 10.0,
                "error": str(e)
            }

    def update_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Update portfolio metrics.

        Returns:
            Current portfolio metrics
        """
        try:
            # Get wallet balance
            wallet_balance = wallet_manager.get_sol_balance()

            # Get position values by risk level
            from src.trading.position_manager import position_manager
            positions = position_manager.get_all_positions()

            # Reset allocations
            self.portfolio_allocation = {
                RiskLevel.LOW.value: 0.0,
                RiskLevel.MEDIUM.value: 0.0,
                RiskLevel.HIGH.value: 0.0,
                RiskLevel.VERY_HIGH.value: 0.0
            }

            # Calculate total portfolio value and allocations
            total_position_value = 0.0

            for position in positions.values():
                position_value = position.get_value()
                total_position_value += position_value

                # Update allocation by risk level
                risk_level = position.risk_level.value
                self.portfolio_allocation[risk_level] += position_value

            # Total portfolio value
            self.portfolio_value_sol = wallet_balance + total_position_value

            # Update max portfolio value
            if self.portfolio_value_sol > self.max_portfolio_value_sol:
                self.max_portfolio_value_sol = self.portfolio_value_sol

            # Calculate drawdown
            if self.max_portfolio_value_sol > 0:
                self.portfolio_drawdown = (self.max_portfolio_value_sol - self.portfolio_value_sol) / self.max_portfolio_value_sol * 100
            else:
                self.portfolio_drawdown = 0.0

            return {
                "portfolio_value_sol": self.portfolio_value_sol,
                "wallet_balance_sol": wallet_balance,
                "position_value_sol": total_position_value,
                "portfolio_drawdown": self.portfolio_drawdown,
                "max_portfolio_value_sol": self.max_portfolio_value_sol,
                "risk_allocation": self.portfolio_allocation
            }
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
            return {
                "error": str(e)
            }

    def check_portfolio_risk(self) -> Dict[str, Any]:
        """
        Check portfolio risk and provide recommendations.

        Returns:
            Risk assessment and recommendations
        """
        if not self.enabled:
            return {"enabled": False}

        try:
            # Update portfolio metrics
            self.update_portfolio_metrics()

            # Get current profile
            profile = self.get_current_profile()

            # Check drawdown
            drawdown_exceeded = self.portfolio_drawdown > profile.max_drawdown_percent

            # Check risk allocation limits
            allocation_exceeded = {}
            for risk_level, allocation in self.portfolio_allocation.items():
                max_allocation = self.risk_allocation_limits.get(risk_level, 100.0)
                allocation_percent = (allocation / self.portfolio_value_sol * 100) if self.portfolio_value_sol > 0 else 0
                allocation_exceeded[risk_level] = allocation_percent > max_allocation

            # Generate recommendations
            recommendations = []

            if drawdown_exceeded:
                recommendations.append({
                    "type": "drawdown",
                    "message": f"Portfolio drawdown ({self.portfolio_drawdown:.2f}%) exceeds maximum ({profile.max_drawdown_percent:.2f}%)",
                    "action": "Consider reducing exposure or setting tighter stop-losses"
                })

            for risk_level, exceeded in allocation_exceeded.items():
                if exceeded:
                    allocation_percent = (self.portfolio_allocation[risk_level] / self.portfolio_value_sol * 100) if self.portfolio_value_sol > 0 else 0
                    max_allocation = self.risk_allocation_limits.get(risk_level, 100.0)
                    recommendations.append({
                        "type": "allocation",
                        "risk_level": risk_level,
                        "message": f"{risk_level.capitalize()} risk allocation ({allocation_percent:.2f}%) exceeds maximum ({max_allocation:.2f}%)",
                        "action": "Consider reducing positions in this risk category"
                    })

            return {
                "enabled": True,
                "portfolio_value_sol": self.portfolio_value_sol,
                "portfolio_drawdown": self.portfolio_drawdown,
                "max_drawdown_percent": profile.max_drawdown_percent,
                "drawdown_exceeded": drawdown_exceeded,
                "risk_allocation": {
                    risk_level: {
                        "value_sol": allocation,
                        "percentage": (allocation / self.portfolio_value_sol * 100) if self.portfolio_value_sol > 0 else 0,
                        "max_percentage": self.risk_allocation_limits.get(risk_level, 100.0),
                        "exceeded": allocation_exceeded[risk_level]
                    } for risk_level, allocation in self.portfolio_allocation.items()
                },
                "recommendations": recommendations
            }
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return {
                "enabled": True,
                "error": str(e)
            }

    def should_reduce_position(self, token_mint: str) -> Tuple[bool, str, float]:
        """
        Check if a position should be reduced based on risk parameters.

        Args:
            token_mint: The token mint address

        Returns:
            Tuple of (should_reduce, reason, percentage_to_reduce)
        """
        if not self.enabled:
            return (False, "", 0.0)

        try:
            # Update portfolio metrics
            self.update_portfolio_metrics()

            # Get current profile
            profile = self.get_current_profile()

            # Check drawdown
            if self.portfolio_drawdown > profile.max_drawdown_percent:
                return (True, f"Portfolio drawdown ({self.portfolio_drawdown:.2f}%) exceeds maximum ({profile.max_drawdown_percent:.2f}%)", 50.0)

            # Get token risk level
            risk_level = self.get_token_risk_level(token_mint)

            # Check risk allocation
            allocation = self.portfolio_allocation.get(risk_level.value, 0.0)
            max_allocation = self.risk_allocation_limits.get(risk_level.value, 100.0)
            allocation_percent = (allocation / self.portfolio_value_sol * 100) if self.portfolio_value_sol > 0 else 0

            if allocation_percent > max_allocation:
                excess_percent = allocation_percent - max_allocation
                reduction_percent = min(excess_percent / allocation_percent * 100, 100.0)
                return (True, f"{risk_level.value.capitalize()} risk allocation ({allocation_percent:.2f}%) exceeds maximum ({max_allocation:.2f}%)", reduction_percent)

            return (False, "", 0.0)
        except Exception as e:
            logger.error(f"Error checking if position should be reduced: {e}")
            return (False, str(e), 0.0)

    def refine_risk_parameters(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine risk parameters based on real-world performance data.

        Args:
            performance_data: Dictionary containing performance metrics
                - trades: List of trade data with profit_loss_percent and risk_level
                - portfolio_drawdown_max: Maximum portfolio drawdown
                - win_rate: Overall win rate

        Returns:
            Dictionary with refinement results
        """
        if not self.enabled:
            logger.warning("Risk management is disabled, cannot refine parameters")
            return {"success": False, "reason": "Risk management is disabled"}

        try:
            logger.info("Refining risk parameters based on performance data")

            # Extract performance metrics
            trades = performance_data.get("trades", [])
            portfolio_drawdown_max = performance_data.get("portfolio_drawdown_max", 0.0)
            win_rate = performance_data.get("win_rate", 0.5)

            if not trades:
                logger.warning("No trade data provided for risk parameter refinement")
                return {"success": False, "reason": "No trade data provided"}

            # Group trades by risk level
            trades_by_risk_level = {}
            for trade in trades:
                risk_level = trade.get("risk_level")
                if risk_level not in trades_by_risk_level:
                    trades_by_risk_level[risk_level] = []
                trades_by_risk_level[risk_level].append(trade)

            # Calculate performance metrics by risk level
            risk_level_metrics = {}
            for risk_level, risk_trades in trades_by_risk_level.items():
                if not risk_trades:
                    continue

                # Calculate profit/loss statistics
                pnl_values = [trade.get("profit_loss_percent", 0.0) for trade in risk_trades]
                avg_pnl = statistics.mean(pnl_values) if pnl_values else 0.0

                # Calculate win rate
                wins = sum(1 for pnl in pnl_values if pnl > 0)
                risk_win_rate = wins / len(pnl_values) if pnl_values else 0.0

                risk_level_metrics[risk_level] = {
                    "avg_pnl": avg_pnl,
                    "win_rate": risk_win_rate,
                    "trade_count": len(risk_trades)
                }

            # Store original values for reporting changes
            original_allocation_limits = self.risk_allocation_limits.copy()
            original_profiles = {name: profile.to_dict() for name, profile in self.risk_profiles.items()}

            # Adjust risk allocation limits based on performance
            adjustments = {}
            for risk_level, metrics in risk_level_metrics.items():
                if risk_level not in self.risk_allocation_limits:
                    continue

                # Calculate adjustment factor based on performance
                # Positive PnL and high win rate increase allocation, negative PnL decreases it
                adjustment_factor = 1.0

                if metrics["avg_pnl"] > 10.0 and metrics["win_rate"] > 0.6:
                    # Excellent performance: increase allocation
                    adjustment_factor = 1.1
                elif metrics["avg_pnl"] > 5.0 and metrics["win_rate"] > 0.5:
                    # Good performance: slightly increase allocation
                    adjustment_factor = 1.05
                elif metrics["avg_pnl"] < -10.0 or metrics["win_rate"] < 0.3:
                    # Poor performance: significantly decrease allocation
                    adjustment_factor = 0.8
                elif metrics["avg_pnl"] < -5.0 or metrics["win_rate"] < 0.4:
                    # Below average performance: decrease allocation
                    adjustment_factor = 0.9

                # Apply adjustment
                current_limit = self.risk_allocation_limits[risk_level]
                new_limit = current_limit * adjustment_factor

                # Ensure limits stay within reasonable bounds
                if risk_level == RiskLevel.LOW.value:
                    new_limit = max(min(new_limit, 80.0), 40.0)
                elif risk_level == RiskLevel.MEDIUM.value:
                    new_limit = max(min(new_limit, 50.0), 20.0)
                elif risk_level == RiskLevel.HIGH.value:
                    new_limit = max(min(new_limit, 20.0), 5.0)
                elif risk_level == RiskLevel.VERY_HIGH.value:
                    new_limit = max(min(new_limit, 10.0), 2.0)

                # Update allocation limit
                self.risk_allocation_limits[risk_level] = new_limit

                # Record adjustment
                adjustments[risk_level] = {
                    "original": current_limit,
                    "new": new_limit,
                    "change_percent": ((new_limit - current_limit) / current_limit * 100) if current_limit > 0 else 0.0
                }

            # Adjust risk profiles based on drawdown experience
            profile_adjustments = {}
            for name, profile in self.risk_profiles.items():
                # If max drawdown was exceeded, reduce max drawdown percent
                if portfolio_drawdown_max > profile.max_drawdown_percent:
                    original_drawdown = profile.max_drawdown_percent
                    # Increase max drawdown to slightly above observed maximum
                    new_drawdown = portfolio_drawdown_max * 1.1
                    profile.max_drawdown_percent = new_drawdown

                    profile_adjustments[name] = {
                        "parameter": "max_drawdown_percent",
                        "original": original_drawdown,
                        "new": new_drawdown
                    }

            # Save updated configuration
            for risk_level, limit in self.risk_allocation_limits.items():
                config_key = f"max_{risk_level}_risk_allocation_percent"
                update_config(config_key, str(limit))

            for name, profile in self.risk_profiles.items():
                for param, value in profile.to_dict().items():
                    if param != "name":
                        config_key = f"{name}_{param}"
                        update_config(config_key, str(value))

            logger.info(f"Risk parameters refined based on {len(trades)} trades")

            return {
                "success": True,
                "allocation_adjustments": adjustments,
                "profile_adjustments": profile_adjustments,
                "performance_metrics": risk_level_metrics
            }
        except Exception as e:
            logger.error(f"Error refining risk parameters: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Create a singleton instance
risk_manager = RiskManager()
