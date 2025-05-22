"""
Enhanced risk management module for the Solana Memecoin Trading Bot.
Implements dynamic position sizing, advanced stop-loss mechanisms, and risk scoring.
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
import numpy as np
from dataclasses import dataclass

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.trading.token_analytics import token_analytics
from src.trading.position_manager import RiskLevel
from src.wallet.wallet import wallet_manager
from src.utils.performance_optimizer import cache_result, parallel_processing
from src.trading.technical_analysis import technical_analyzer

# Get logger for this module
logger = get_logger(__name__)

@dataclass
class RiskMetrics:
    volatility: float
    liquidity_score: float
    holder_distribution: float
    contract_risk: float
    social_sentiment: float
    overall_risk_score: float

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

        self.risk_metrics_cache: Dict[str, RiskMetrics] = {}
        self.position_limits: Dict[str, float] = {}
        self.stop_loss_levels: Dict[str, List[float]] = {}
        self.risk_scores: Dict[str, float] = {}

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

    def calculate_position_size(self, token_address: str, portfolio_value: float) -> float:
        """
        Calculate recommended position size based on risk metrics.
        
        Args:
            token_address: The token's address
            portfolio_value: Total portfolio value in SOL
            
        Returns:
            Recommended position size in SOL
        """
        metrics = self.calculate_token_risk_metrics(token_address)
        
        # Base position size (5% of portfolio)
        base_size = portfolio_value * 0.05
        
        # Adjust based on risk score
        risk_adjustment = 1 - metrics.overall_risk_score
        
        # Apply minimum and maximum limits
        min_size = portfolio_value * 0.01  # 1% minimum
        max_size = portfolio_value * 0.20  # 20% maximum
        
        position_size = base_size * risk_adjustment
        return max(min_size, min(max_size, position_size))

    def calculate_dynamic_stop_loss(self, token_address: str, entry_price: float,
                                  current_price: float) -> List[float]:
        """
        Calculate dynamic stop-loss levels based on risk metrics and price action.
        
        Args:
            token_address: The token's address
            entry_price: Entry price
            current_price: Current price
            
        Returns:
            List of stop-loss levels
        """
        metrics = self.calculate_token_risk_metrics(token_address)
        
        # Calculate price change percentage
        price_change = (current_price - entry_price) / entry_price
        
        # Base stop-loss levels
        if price_change >= 0.2:  # 20% profit
            levels = [0.15, 0.10, 0.05]  # Tighter stops for profitable positions
        elif price_change >= 0.1:  # 10% profit
            levels = [0.08, 0.05, 0.03]
        else:
            levels = [0.05, 0.03, 0.02]  # Wider stops for new positions
        
        # Adjust based on risk score
        risk_adjustment = 1 + metrics.overall_risk_score
        levels = [level * risk_adjustment for level in levels]
        
        # Calculate actual stop prices
        stop_prices = [current_price * (1 - level) for level in levels]
        
        self.stop_loss_levels[token_address] = stop_prices
        return stop_prices

    def should_close_position(self, token_address: str, entry_price: float,
                            current_price: float) -> Tuple[bool, str]:
        """
        Determine if a position should be closed based on risk metrics and price action.
        
        Args:
            token_address: The token's address
            entry_price: Entry price
            current_price: Current price
            
        Returns:
            Tuple of (should_close, reason)
        """
        metrics = self.calculate_token_risk_metrics(token_address)
        
        # Check stop-loss levels
        if token_address in self.stop_loss_levels:
            for i, stop_price in enumerate(self.stop_loss_levels[token_address]):
                if current_price <= stop_price:
                    return True, f"Stop-loss level {i+1} triggered"
        
        # Check risk metrics
        if metrics.overall_risk_score > 0.8:
            return True, "Risk score too high"
        
        # Check price action
        price_change = (current_price - entry_price) / entry_price
        if price_change <= -0.1 and metrics.volatility > 0.3:
            return True, "High volatility with significant loss"
        
        return False, ""

    @cache_result(ttl=300)
    def calculate_token_risk_metrics(self, token_address: str) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a token.
        
        Args:
            token_address: The token's address
            
        Returns:
            RiskMetrics object containing various risk indicators
        """
        try:
            # Get token analytics
            analytics = token_analytics.get_token_analytics(token_address)
            
            # Calculate volatility (last 24h)
            price_history = analytics.get('price_history', [])
            volatility = np.std(price_history) if price_history else 1.0
            
            # Calculate liquidity score
            liquidity = analytics.get('liquidity', 0)
            liquidity_score = min(1.0, liquidity / 1000)  # Normalize to 0-1
            
            # Calculate holder distribution score
            holders = analytics.get('holders', {})
            holder_distribution = self._calculate_holder_distribution(holders)
            
            # Calculate contract risk
            contract_risk = self._calculate_contract_risk(token_address)
            
            # Calculate social sentiment
            sentiment = analytics.get('social_sentiment', 0.5)
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(
                volatility, liquidity_score, holder_distribution,
                contract_risk, sentiment
            )
            
            metrics = RiskMetrics(
                volatility=volatility,
                liquidity_score=liquidity_score,
                holder_distribution=holder_distribution,
                contract_risk=contract_risk,
                social_sentiment=sentiment,
                overall_risk_score=overall_risk_score
            )
            
            self.risk_metrics_cache[token_address] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {token_address}: {str(e)}")
            return RiskMetrics(1.0, 0.0, 0.0, 1.0, 0.5, 1.0)  # Return high risk metrics
    
    def _calculate_holder_distribution(self, holders: Dict[str, float]) -> float:
        """Calculate holder distribution score (0-1, higher is better)."""
        if not holders:
            return 0.0
            
        # Calculate Gini coefficient
        values = sorted(holders.values())
        n = len(values)
        if n == 0:
            return 0.0
            
        index = np.arange(1, n + 1)
        return 1 - (2 * np.sum(index * values) / (n * np.sum(values)))
    
    def _calculate_contract_risk(self, token_address: str) -> float:
        """Calculate contract risk score (0-1, higher is worse)."""
        try:
            # Get contract analysis
            contract_analysis = token_analytics.get_contract_analysis(token_address)
            
            risk_factors = {
                'has_blacklist': 0.3,
                'has_whitelist': 0.2,
                'has_pause': 0.2,
                'has_freeze': 0.3,
                'has_owner_controls': 0.2,
                'is_verified': -0.3,
                'has_locked_liquidity': -0.2
            }
            
            risk_score = 0.0
            for factor, weight in risk_factors.items():
                if contract_analysis.get(factor, False):
                    risk_score += weight
                    
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating contract risk: {str(e)}")
            return 1.0
    
    def _calculate_overall_risk_score(self, volatility: float, liquidity_score: float,
                                    holder_distribution: float, contract_risk: float,
                                    sentiment: float) -> float:
        """Calculate overall risk score (0-1, higher is worse)."""
        weights = {
            'volatility': 0.3,
            'liquidity': 0.2,
            'holder_distribution': 0.15,
            'contract_risk': 0.25,
            'sentiment': 0.1
        }
        
        # Normalize volatility to 0-1
        norm_volatility = min(1.0, volatility / 0.5)
        
        # Calculate weighted score
        score = (
            weights['volatility'] * norm_volatility +
            weights['liquidity'] * (1 - liquidity_score) +
            weights['holder_distribution'] * (1 - holder_distribution) +
            weights['contract_risk'] * contract_risk +
            weights['sentiment'] * (1 - sentiment)
        )
        
        return max(0.0, min(1.0, score))

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
