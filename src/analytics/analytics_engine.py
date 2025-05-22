"""
Analytics Engine for Solana Memecoin Trading Bot.
Provides comprehensive trading analytics and reporting capabilities.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logging_utils import get_logger
from src.trading.position_manager import position_manager
from src.trading.token_analytics import token_analytics
from src.trading.technical_analysis import technical_analyzer
from src.trading.sentiment_analysis import sentiment_analyzer
from src.utils.performance_tracker import performance_tracker
from src.trading.strategy_engine import strategy_engine

logger = get_logger(__name__)

@dataclass
class AnalyticsConfig:
    """Analytics configuration settings."""
    data_dir: str = "data/analytics"
    max_history_days: int = 365
    update_interval: int = 300  # seconds
    enable_advanced_metrics: bool = True
    enable_correlation_analysis: bool = True
    enable_market_impact: bool = True

class AnalyticsEngine:
    def __init__(self):
        self.config = AnalyticsConfig()
        self.data_cache: Dict[str, Any] = {}
        self.last_update = datetime.now()
        
        # Ensure data directory exists
        os.makedirs(self.config.data_dir, exist_ok=True)
        
        # Initialize data structures
        self.performance_history: pd.DataFrame = pd.DataFrame()
        self.token_correlations: pd.DataFrame = pd.DataFrame()
        self.market_impact_data: Dict[str, List[Dict]] = {}
        
    def update_analytics(self) -> None:
        """Update all analytics data."""
        try:
            # Update performance history
            self._update_performance_history()
            
            # Update token correlations
            if self.config.enable_correlation_analysis:
                self._update_token_correlations()
            
            # Update market impact data
            if self.config.enable_market_impact:
                self._update_market_impact()
            
            self.last_update = datetime.now()
            logger.info("Analytics data updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating analytics: {e}")
            raise
            
    def _update_performance_history(self) -> None:
        """Update performance history data."""
        try:
            # Get performance data from tracker
            performance_data = performance_tracker.get_performance_history(
                days=self.config.max_history_days
            )
            
            # Convert to DataFrame
            self.performance_history = pd.DataFrame(performance_data)
            
            # Calculate additional metrics
            if self.config.enable_advanced_metrics:
                self._calculate_advanced_metrics()
                
            # Save to file
            self._save_performance_history()
            
        except Exception as e:
            logger.error(f"Error updating performance history: {e}")
            raise
            
    def _calculate_advanced_metrics(self) -> None:
        """Calculate advanced performance metrics."""
        try:
            # Calculate rolling metrics
            self.performance_history['rolling_volatility'] = (
                self.performance_history['portfolio_value']
                .pct_change()
                .rolling(window=20)
                .std()
                * np.sqrt(252)
            )
            
            # Calculate drawdown
            self.performance_history['drawdown'] = (
                self.performance_history['portfolio_value']
                .cummax()
                - self.performance_history['portfolio_value']
            )
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            excess_returns = (
                self.performance_history['portfolio_value']
                .pct_change()
                - risk_free_rate/252
            )
            self.performance_history['sharpe_ratio'] = (
                excess_returns.rolling(window=20).mean()
                / excess_returns.rolling(window=20).std()
                * np.sqrt(252)
            )
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            raise
            
    def _update_token_correlations(self) -> None:
        """Update token price correlations."""
        try:
            # Get token price data
            token_prices = {}
            for position in position_manager.get_all_positions():
                token_prices[position['token_address']] = (
                    token_analytics.get_token_price_history(
                        position['token_address'],
                        days=30
                    )
                )
            
            # Calculate correlations
            price_df = pd.DataFrame(token_prices)
            self.token_correlations = price_df.corr()
            
            # Save to file
            self._save_token_correlations()
            
        except Exception as e:
            logger.error(f"Error updating token correlations: {e}")
            raise
            
    def _update_market_impact(self) -> None:
        """Update market impact analysis."""
        try:
            for position in position_manager.get_all_positions():
                token_address = position['token_address']
                
                # Get trade history
                trades = performance_tracker.get_token_trade_history(token_address)
                
                # Calculate market impact
                impact_data = []
                for trade in trades:
                    impact = self._calculate_trade_impact(trade)
                    impact_data.append(impact)
                
                self.market_impact_data[token_address] = impact_data
                
            # Save to file
            self._save_market_impact()
            
        except Exception as e:
            logger.error(f"Error updating market impact: {e}")
            raise
            
    def _calculate_trade_impact(self, trade: Dict) -> Dict:
        """Calculate market impact of a single trade."""
        try:
            # Get price data around trade
            price_data = token_analytics.get_token_price_history(
                trade['token_address'],
                start_time=trade['timestamp'] - timedelta(hours=1),
                end_time=trade['timestamp'] + timedelta(hours=1)
            )
            
            # Calculate price impact
            pre_trade_price = price_data[0]['price']
            post_trade_price = price_data[-1]['price']
            price_impact = (post_trade_price - pre_trade_price) / pre_trade_price
            
            return {
                'timestamp': trade['timestamp'],
                'trade_size': trade['amount'],
                'price_impact': price_impact,
                'pre_trade_price': pre_trade_price,
                'post_trade_price': post_trade_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade impact: {e}")
            raise
            
    def _save_performance_history(self) -> None:
        """Save performance history to file."""
        try:
            file_path = os.path.join(self.config.data_dir, 'performance_history.csv')
            self.performance_history.to_csv(file_path, index=False)
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
            raise
            
    def _save_token_correlations(self) -> None:
        """Save token correlations to file."""
        try:
            file_path = os.path.join(self.config.data_dir, 'token_correlations.csv')
            self.token_correlations.to_csv(file_path)
        except Exception as e:
            logger.error(f"Error saving token correlations: {e}")
            raise
            
    def _save_market_impact(self) -> None:
        """Save market impact data to file."""
        try:
            file_path = os.path.join(self.config.data_dir, 'market_impact.json')
            with open(file_path, 'w') as f:
                json.dump(self.market_impact_data, f)
        except Exception as e:
            logger.error(f"Error saving market impact data: {e}")
            raise
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            if self.performance_history.empty:
                return {}
                
            latest = self.performance_history.iloc[-1]
            
            return {
                'total_return': (
                    latest['portfolio_value'] / self.performance_history.iloc[0]['portfolio_value'] - 1
                ),
                'annualized_return': (
                    (1 + latest['portfolio_value'] / self.performance_history.iloc[0]['portfolio_value'])
                    ** (252 / len(self.performance_history)) - 1
                ),
                'volatility': latest['rolling_volatility'],
                'sharpe_ratio': latest['sharpe_ratio'],
                'max_drawdown': self.performance_history['drawdown'].max(),
                'win_rate': performance_tracker.get_win_rate(),
                'profit_factor': performance_tracker.get_profit_factor(),
                'average_trade': performance_tracker.get_average_trade(),
                'best_trade': performance_tracker.get_best_trade(),
                'worst_trade': performance_tracker.get_worst_trade()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
            
    def get_token_analytics(self, token_address: str) -> Dict[str, Any]:
        """Get comprehensive token analytics."""
        try:
            # Get basic analytics
            analytics = token_analytics.get_token_analytics(token_address)
            
            # Add correlation data
            if not self.token_correlations.empty:
                analytics['correlations'] = self.token_correlations[token_address].to_dict()
            
            # Add market impact data
            if token_address in self.market_impact_data:
                analytics['market_impact'] = self.market_impact_data[token_address]
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting token analytics: {e}")
            return {}
            
    def get_portfolio_analytics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio analytics."""
        try:
            positions = position_manager.get_all_positions()
            
            # Calculate portfolio metrics
            total_value = sum(p['current_value'] for p in positions)
            position_weights = {
                p['token_address']: p['current_value'] / total_value
                for p in positions
            }
            
            # Calculate portfolio risk metrics
            portfolio_volatility = np.sqrt(
                sum(
                    position_weights[token] ** 2 * self.token_correlations.loc[token, token]
                    for token in position_weights
                )
            )
            
            return {
                'total_value': total_value,
                'position_weights': position_weights,
                'portfolio_volatility': portfolio_volatility,
                'diversification_score': 1 - sum(w ** 2 for w in position_weights.values()),
                'correlation_matrix': self.token_correlations.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio analytics: {e}")
            return {}
            
    def generate_report(self, report_type: str = 'daily') -> Dict[str, Any]:
        """Generate a comprehensive trading report."""
        try:
            if report_type == 'daily':
                start_time = datetime.now() - timedelta(days=1)
            elif report_type == 'weekly':
                start_time = datetime.now() - timedelta(weeks=1)
            elif report_type == 'monthly':
                start_time = datetime.now() - timedelta(days=30)
            else:
                raise ValueError(f"Invalid report type: {report_type}")
                
            # Get performance data for period
            period_data = self.performance_history[
                self.performance_history['timestamp'] >= start_time
            ]
            
            # Calculate period metrics
            period_return = (
                period_data.iloc[-1]['portfolio_value']
                / period_data.iloc[0]['portfolio_value'] - 1
            )
            
            # Get trade data for period
            trades = performance_tracker.get_trade_history(start_time=start_time)
            
            return {
                'period': report_type,
                'start_time': start_time,
                'end_time': datetime.now(),
                'period_return': period_return,
                'trades': len(trades),
                'win_rate': sum(1 for t in trades if t['pnl'] > 0) / len(trades) if trades else 0,
                'average_trade': sum(t['pnl'] for t in trades) / len(trades) if trades else 0,
                'best_trade': max((t['pnl'] for t in trades), default=0),
                'worst_trade': min((t['pnl'] for t in trades), default=0),
                'performance_metrics': self.get_performance_metrics(),
                'portfolio_analytics': self.get_portfolio_analytics()
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {}

# Global instance
analytics_engine = AnalyticsEngine() 