"""
Exit strategies module for the Solana Memecoin Trading Bot.
Provides various exit strategies for managing positions.
"""

import json
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

from config import get_config_value, update_config
from src.trading.technical_analysis import technical_analyzer
from src.trading.token_analytics import token_analytics
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class ExitStrategies:
    """Provides various exit strategies for managing positions."""
    
    def __init__(self):
        """Initialize the exit strategies module."""
        self.enabled = True
        self.exit_strategy_config = get_config_value("exit_strategy", {})
        self.strategy_type = self.exit_strategy_config.get("strategy_type", "tiered")
        
        # Load trailing stop settings
        self.trailing_stop_enabled = self.exit_strategy_config.get("trailing_stop", {}).get("enabled", True)
        self.trailing_stop_percentage = self.exit_strategy_config.get("trailing_stop", {}).get("trail_percentage", 15.0)
        self.trailing_stop_activation = self.exit_strategy_config.get("trailing_stop", {}).get("activation_percentage", 20.0)
        
        # Load time-based exit settings
        self.time_based_exit_enabled = self.exit_strategy_config.get("time_based_exit", {}).get("enabled", True)
        self.time_based_exit_hours = self.exit_strategy_config.get("time_based_exit", {}).get("hold_hours", 48)
        self.time_based_exit_min_profit = self.exit_strategy_config.get("time_based_exit", {}).get("min_profit_percentage", 5.0)
        
        # Load indicator-based exit settings
        self.indicator_exit_enabled = self.exit_strategy_config.get("indicator_based_exit", {}).get("enabled", True)
        self.rsi_overbought = self.exit_strategy_config.get("indicator_based_exit", {}).get("rsi_overbought", 75.0)
        self.macd_signal_exit = self.exit_strategy_config.get("indicator_based_exit", {}).get("macd_signal", True)
        self.bollinger_band_exit = self.exit_strategy_config.get("indicator_based_exit", {}).get("bollinger_band", True)
        
        # Load dynamic stop loss settings
        self.dynamic_stop_loss_enabled = self.exit_strategy_config.get("dynamic_stop_loss", {}).get("enabled", True)
        self.initial_stop_loss = self.exit_strategy_config.get("dynamic_stop_loss", {}).get("initial_percentage", 5.0)
        self.profit_step_percentage = self.exit_strategy_config.get("dynamic_stop_loss", {}).get("profit_step_percentage", 10.0)
        self.stop_loss_step_percentage = self.exit_strategy_config.get("dynamic_stop_loss", {}).get("stop_loss_step_percentage", 2.5)
        
        # Load take profit tiers
        self.take_profit_tiers = get_config_value("take_profit_tiers", [
            {"percentage": 20.0, "sell_percentage": 30.0},
            {"percentage": 50.0, "sell_percentage": 40.0},
            {"percentage": 100.0, "sell_percentage": 30.0},
        ])
        
        # Position tracking
        self.position_high_prices: Dict[str, float] = {}
        self.position_entry_times: Dict[str, datetime] = {}
    
    def should_exit_position(self, token_mint: str, entry_price: float, current_price: float, 
                           entry_time: Optional[datetime] = None, position_size: float = 0.0,
                           already_sold_percentage: float = 0.0) -> Tuple[bool, str, float]:
        """
        Determine if a position should be exited based on the configured exit strategy.
        
        Args:
            token_mint: The token mint address
            entry_price: The entry price
            current_price: The current price
            entry_time: The entry time (optional)
            position_size: The position size (optional)
            already_sold_percentage: Percentage of position already sold (optional)
            
        Returns:
            Tuple of (should exit, reason, percentage to sell)
        """
        if not self.enabled:
            return (False, "Exit strategies disabled", 0.0)
        
        # Track highest price seen for this position for trailing stop
        if token_mint not in self.position_high_prices:
            self.position_high_prices[token_mint] = current_price
        else:
            self.position_high_prices[token_mint] = max(self.position_high_prices[token_mint], current_price)
        
        # Track entry time for time-based exit
        if entry_time and token_mint not in self.position_entry_times:
            self.position_entry_times[token_mint] = entry_time
        
        # Calculate current profit percentage
        profit_percentage = ((current_price / entry_price) - 1) * 100
        
        # Determine which exit strategy to use
        if self.strategy_type == "simple":
            return self._simple_exit_strategy(token_mint, entry_price, current_price, profit_percentage)
        elif self.strategy_type == "tiered":
            return self._tiered_exit_strategy(token_mint, entry_price, current_price, profit_percentage, already_sold_percentage)
        elif self.strategy_type == "trailing":
            return self._trailing_exit_strategy(token_mint, entry_price, current_price, profit_percentage)
        elif self.strategy_type == "smart":
            return self._smart_exit_strategy(token_mint, entry_price, current_price, profit_percentage, entry_time, position_size, already_sold_percentage)
        else:
            # Default to tiered strategy
            return self._tiered_exit_strategy(token_mint, entry_price, current_price, profit_percentage, already_sold_percentage)
    
    def _simple_exit_strategy(self, token_mint: str, entry_price: float, current_price: float, 
                            profit_percentage: float) -> Tuple[bool, str, float]:
        """
        Simple exit strategy with fixed stop loss and take profit levels.
        
        Args:
            token_mint: The token mint address
            entry_price: The entry price
            current_price: The current price
            profit_percentage: The current profit percentage
            
        Returns:
            Tuple of (should exit, reason, percentage to sell)
        """
        # Get stop loss and take profit percentages
        stop_loss_percentage = get_config_value("stop_loss_percentage", 5.0)
        take_profit_percentage = get_config_value("take_profit_percentage", 20.0)
        
        # Check stop loss
        if profit_percentage <= -stop_loss_percentage:
            return (True, f"Stop loss triggered at {profit_percentage:.2f}%", 100.0)
        
        # Check take profit
        if profit_percentage >= take_profit_percentage:
            return (True, f"Take profit triggered at {profit_percentage:.2f}%", 100.0)
        
        return (False, "No exit signal", 0.0)
    
    def _tiered_exit_strategy(self, token_mint: str, entry_price: float, current_price: float, 
                            profit_percentage: float, already_sold_percentage: float) -> Tuple[bool, str, float]:
        """
        Tiered exit strategy with multiple take profit levels.
        
        Args:
            token_mint: The token mint address
            entry_price: The entry price
            current_price: The current price
            profit_percentage: The current profit percentage
            already_sold_percentage: Percentage of position already sold
            
        Returns:
            Tuple of (should exit, reason, percentage to sell)
        """
        # Get stop loss percentage
        stop_loss_percentage = get_config_value("stop_loss_percentage", 5.0)
        
        # Check stop loss
        if profit_percentage <= -stop_loss_percentage:
            remaining_percentage = 100.0 - already_sold_percentage
            return (True, f"Stop loss triggered at {profit_percentage:.2f}%", remaining_percentage)
        
        # Check take profit tiers
        for tier in self.take_profit_tiers:
            tier_percentage = tier.get("percentage", 0.0)
            sell_percentage = tier.get("sell_percentage", 0.0)
            
            if profit_percentage >= tier_percentage:
                # Calculate how much to sell based on what's already been sold
                adjusted_sell_percentage = min(sell_percentage, 100.0 - already_sold_percentage)
                
                if adjusted_sell_percentage > 0:
                    return (True, f"Take profit tier triggered at {profit_percentage:.2f}% (tier: {tier_percentage}%)", adjusted_sell_percentage)
        
        # Check if we should also apply trailing stop
        if self.trailing_stop_enabled and profit_percentage >= self.trailing_stop_activation:
            return self._check_trailing_stop(token_mint, entry_price, current_price, profit_percentage, already_sold_percentage)
        
        return (False, "No exit signal", 0.0)
    
    def _trailing_exit_strategy(self, token_mint: str, entry_price: float, current_price: float, 
                              profit_percentage: float) -> Tuple[bool, str, float]:
        """
        Trailing stop exit strategy.
        
        Args:
            token_mint: The token mint address
            entry_price: The entry price
            current_price: The current price
            profit_percentage: The current profit percentage
            
        Returns:
            Tuple of (should exit, reason, percentage to sell)
        """
        # Get stop loss percentage for initial protection
        stop_loss_percentage = get_config_value("stop_loss_percentage", 5.0)
        
        # Check stop loss before profit
        if profit_percentage <= -stop_loss_percentage:
            return (True, f"Stop loss triggered at {profit_percentage:.2f}%", 100.0)
        
        # Check if we've reached the activation threshold for trailing stop
        if profit_percentage >= self.trailing_stop_activation:
            return self._check_trailing_stop(token_mint, entry_price, current_price, profit_percentage, 0.0)
        
        return (False, "No exit signal", 0.0)
    
    def _smart_exit_strategy(self, token_mint: str, entry_price: float, current_price: float, 
                           profit_percentage: float, entry_time: Optional[datetime], 
                           position_size: float, already_sold_percentage: float) -> Tuple[bool, str, float]:
        """
        Smart exit strategy combining multiple exit methods.
        
        Args:
            token_mint: The token mint address
            entry_price: The entry price
            current_price: The current price
            profit_percentage: The current profit percentage
            entry_time: The entry time
            position_size: The position size
            already_sold_percentage: Percentage of position already sold
            
        Returns:
            Tuple of (should exit, reason, percentage to sell)
        """
        # First check dynamic stop loss
        if self.dynamic_stop_loss_enabled:
            dynamic_result = self._check_dynamic_stop_loss(token_mint, entry_price, current_price, profit_percentage, already_sold_percentage)
            if dynamic_result[0]:
                return dynamic_result
        
        # Then check tiered take profit
        tiered_result = self._tiered_exit_strategy(token_mint, entry_price, current_price, profit_percentage, already_sold_percentage)
        if tiered_result[0]:
            return tiered_result
        
        # Check trailing stop if activated
        if self.trailing_stop_enabled and profit_percentage >= self.trailing_stop_activation:
            trailing_result = self._check_trailing_stop(token_mint, entry_price, current_price, profit_percentage, already_sold_percentage)
            if trailing_result[0]:
                return trailing_result
        
        # Check time-based exit
        if self.time_based_exit_enabled and entry_time:
            time_result = self._check_time_based_exit(token_mint, entry_time, profit_percentage, already_sold_percentage)
            if time_result[0]:
                return time_result
        
        # Check indicator-based exit
        if self.indicator_exit_enabled:
            indicator_result = self._check_indicator_based_exit(token_mint, entry_price, current_price, already_sold_percentage)
            if indicator_result[0]:
                return indicator_result
        
        # Check token analytics for risk-based exit
        analytics_result = self._check_analytics_based_exit(token_mint, entry_price, current_price, profit_percentage, already_sold_percentage)
        if analytics_result[0]:
            return analytics_result
        
        return (False, "No exit signal", 0.0)
    
    def _check_trailing_stop(self, token_mint: str, entry_price: float, current_price: float, 
                           profit_percentage: float, already_sold_percentage: float) -> Tuple[bool, str, float]:
        """
        Check if trailing stop has been triggered.
        
        Args:
            token_mint: The token mint address
            entry_price: The entry price
            current_price: The current price
            profit_percentage: The current profit percentage
            already_sold_percentage: Percentage of position already sold
            
        Returns:
            Tuple of (should exit, reason, percentage to sell)
        """
        # Get highest price seen for this position
        highest_price = self.position_high_prices.get(token_mint, current_price)
        
        # Calculate trailing stop price
        trail_percentage = self.trailing_stop_percentage
        trailing_stop_price = highest_price * (1 - trail_percentage / 100)
        
        # Check if current price is below trailing stop
        if current_price <= trailing_stop_price:
            remaining_percentage = 100.0 - already_sold_percentage
            return (True, f"Trailing stop triggered at {profit_percentage:.2f}% (drop from high: {((highest_price - current_price) / highest_price * 100):.2f}%)", remaining_percentage)
        
        return (False, "No trailing stop exit signal", 0.0)
    
    def _check_time_based_exit(self, token_mint: str, entry_time: datetime, 
                             profit_percentage: float, already_sold_percentage: float) -> Tuple[bool, str, float]:
        """
        Check if time-based exit has been triggered.
        
        Args:
            token_mint: The token mint address
            entry_time: The entry time
            profit_percentage: The current profit percentage
            already_sold_percentage: Percentage of position already sold
            
        Returns:
            Tuple of (should exit, reason, percentage to sell)
        """
        # Calculate how long we've been in the position
        current_time = datetime.now()
        hold_duration = current_time - entry_time
        hold_hours = hold_duration.total_seconds() / 3600
        
        # Check if we've held the position long enough
        if hold_hours >= self.time_based_exit_hours:
            # Only exit if we've reached the minimum profit
            if profit_percentage >= self.time_based_exit_min_profit:
                remaining_percentage = 100.0 - already_sold_percentage
                return (True, f"Time-based exit triggered after {hold_hours:.1f} hours with {profit_percentage:.2f}% profit", remaining_percentage)
        
        return (False, "No time-based exit signal", 0.0)
    
    def _check_indicator_based_exit(self, token_mint: str, entry_price: float, 
                                  current_price: float, already_sold_percentage: float) -> Tuple[bool, str, float]:
        """
        Check if indicator-based exit has been triggered.
        
        Args:
            token_mint: The token mint address
            entry_price: The entry price
            current_price: The current price
            already_sold_percentage: Percentage of position already sold
            
        Returns:
            Tuple of (should exit, reason, percentage to sell)
        """
        # Check if technical analysis is enabled
        if not technical_analyzer.enabled:
            return (False, "Technical analysis not enabled", 0.0)
        
        # Check RSI for overbought condition
        if self.rsi_overbought > 0:
            rsi = technical_analyzer.get_rsi(token_mint)
            if rsi is not None and rsi > self.rsi_overbought:
                remaining_percentage = 100.0 - already_sold_percentage
                return (True, f"RSI overbought exit triggered (RSI: {rsi:.1f})", remaining_percentage)
        
        # Check MACD for bearish crossover
        if self.macd_signal_exit:
            macd = technical_analyzer.get_macd(token_mint)
            if macd is not None:
                macd_line, signal_line = macd
                if macd_line < signal_line and macd_line > 0:  # Bearish crossover in positive territory
                    remaining_percentage = 100.0 - already_sold_percentage
                    return (True, f"MACD bearish crossover exit triggered", remaining_percentage)
        
        # Check Bollinger Bands for price above upper band
        if self.bollinger_band_exit:
            bb = technical_analyzer.get_bollinger_bands(token_mint)
            if bb is not None:
                upper_band, _, _ = bb
                if current_price > upper_band * 1.05:  # Price significantly above upper band
                    remaining_percentage = 100.0 - already_sold_percentage
                    return (True, f"Bollinger Band exit triggered (price above upper band)", remaining_percentage)
        
        # Use the comprehensive indicator check
        indicator_result = technical_analyzer.should_exit_based_on_indicators(token_mint, entry_price, current_price)
        if indicator_result[0]:
            remaining_percentage = 100.0 - already_sold_percentage
            return (True, f"Technical indicator exit: {indicator_result[1]}", remaining_percentage)
        
        return (False, "No indicator-based exit signal", 0.0)
    
    def _check_dynamic_stop_loss(self, token_mint: str, entry_price: float, current_price: float, 
                               profit_percentage: float, already_sold_percentage: float) -> Tuple[bool, str, float]:
        """
        Check if dynamic stop loss has been triggered.
        
        Args:
            token_mint: The token mint address
            entry_price: The entry price
            current_price: The current price
            profit_percentage: The current profit percentage
            already_sold_percentage: Percentage of position already sold
            
        Returns:
            Tuple of (should exit, reason, percentage to sell)
        """
        # Calculate dynamic stop loss based on profit
        stop_loss_percentage = self.initial_stop_loss
        
        # Adjust stop loss based on profit steps
        if profit_percentage > 0:
            profit_steps = int(profit_percentage / self.profit_step_percentage)
            stop_loss_adjustment = profit_steps * self.stop_loss_step_percentage
            stop_loss_percentage = max(0, self.initial_stop_loss - stop_loss_adjustment)
        
        # Check if current price is below dynamic stop loss
        if profit_percentage <= -stop_loss_percentage:
            remaining_percentage = 100.0 - already_sold_percentage
            return (True, f"Dynamic stop loss triggered at {profit_percentage:.2f}% (adjusted stop: {stop_loss_percentage:.2f}%)", remaining_percentage)
        
        return (False, "No dynamic stop loss exit signal", 0.0)
    
    def _check_analytics_based_exit(self, token_mint: str, entry_price: float, current_price: float, 
                                  profit_percentage: float, already_sold_percentage: float) -> Tuple[bool, str, float]:
        """
        Check if analytics-based exit has been triggered.
        
        Args:
            token_mint: The token mint address
            entry_price: The entry price
            current_price: The current price
            profit_percentage: The current profit percentage
            already_sold_percentage: Percentage of position already sold
            
        Returns:
            Tuple of (should exit, reason, percentage to sell)
        """
        # Check if token analytics is enabled
        if not token_analytics.enabled:
            return (False, "Token analytics not enabled", 0.0)
        
        try:
            # Get risk level
            risk_level = token_analytics.get_risk_level(token_mint)
            
            # For high risk tokens, exit earlier
            if risk_level == "very_high" and profit_percentage >= 30:
                remaining_percentage = 100.0 - already_sold_percentage
                return (True, f"Risk-based exit triggered for very high risk token at {profit_percentage:.2f}% profit", remaining_percentage)
            
            if risk_level == "high" and profit_percentage >= 50:
                remaining_percentage = 100.0 - already_sold_percentage
                return (True, f"Risk-based exit triggered for high risk token at {profit_percentage:.2f}% profit", remaining_percentage)
            
            # Check analytics data for warning signs
            analytics = token_analytics.get_token_analytics(token_mint)
            
            # Check for negative sentiment trend
            sentiment_trend = analytics.get("sentiment_trend", 0.0)
            if sentiment_trend < -0.3 and profit_percentage > 20:
                remaining_percentage = 100.0 - already_sold_percentage
                return (True, f"Sentiment-based exit triggered (negative trend: {sentiment_trend:.2f})", remaining_percentage)
            
            # Check for declining social activity
            twitter_mentions = analytics.get("twitter_mentions_24h", 0)
            if twitter_mentions == 0 and profit_percentage > 30:
                remaining_percentage = 100.0 - already_sold_percentage
                return (True, f"Social activity-based exit triggered (no recent mentions)", remaining_percentage)
        
        except Exception as e:
            logger.error(f"Error in analytics-based exit check: {e}")
        
        return (False, "No analytics-based exit signal", 0.0)
    
    def get_recommended_exit_strategy(self, token_mint: str) -> Dict[str, Any]:
        """
        Get recommended exit strategy for a token.
        
        Args:
            token_mint: The token mint address
            
        Returns:
            Recommended exit strategy
        """
        try:
            # Get token analytics
            if token_analytics.enabled:
                return token_analytics.get_exit_strategy_recommendation(token_mint)
            
            # Fallback to default strategy
            return {
                "strategy_type": "tiered",
                "take_profit_levels": self.take_profit_tiers,
                "stop_loss_percentage": self.initial_stop_loss,
                "trailing_stop": {
                    "enabled": self.trailing_stop_enabled,
                    "trail_percentage": self.trailing_stop_percentage,
                    "activation_percentage": self.trailing_stop_activation
                },
                "time_based_exit": {
                    "enabled": self.time_based_exit_enabled,
                    "hold_hours": self.time_based_exit_hours,
                    "min_profit_percentage": self.time_based_exit_min_profit
                }
            }
        except Exception as e:
            logger.error(f"Error getting recommended exit strategy: {e}")
            return {
                "strategy_type": "tiered",
                "take_profit_levels": self.take_profit_tiers,
                "stop_loss_percentage": self.initial_stop_loss
            }
    
    def reset_position_tracking(self, token_mint: str) -> None:
        """
        Reset position tracking for a token.
        
        Args:
            token_mint: The token mint address
        """
        if token_mint in self.position_high_prices:
            del self.position_high_prices[token_mint]
        
        if token_mint in self.position_entry_times:
            del self.position_entry_times[token_mint]
    
    def update_position_tracking(self, token_mint: str, current_price: float, entry_time: Optional[datetime] = None) -> None:
        """
        Update position tracking for a token.
        
        Args:
            token_mint: The token mint address
            current_price: The current price
            entry_time: The entry time (optional)
        """
        # Update highest price
        if token_mint in self.position_high_prices:
            self.position_high_prices[token_mint] = max(self.position_high_prices[token_mint], current_price)
        else:
            self.position_high_prices[token_mint] = current_price
        
        # Update entry time if provided
        if entry_time and token_mint not in self.position_entry_times:
            self.position_entry_times[token_mint] = entry_time


# Create singleton instance
exit_strategies = ExitStrategies()
