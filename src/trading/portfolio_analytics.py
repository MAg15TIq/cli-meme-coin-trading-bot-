"""
Portfolio analytics dashboard for the Solana Memecoin Trading Bot.
Provides comprehensive portfolio analysis and visualization.
"""

import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.trading.jupiter_api import jupiter_api
from src.trading.position_manager import position_manager
from src.wallet.wallet import wallet_manager

# Get logger for this module
logger = get_logger(__name__)


@dataclass
class PortfolioSnapshot:
    """Represents a portfolio snapshot at a point in time."""
    timestamp: datetime
    total_value_sol: float
    sol_balance: float
    token_value_sol: float
    positions: Dict[str, Dict[str, Any]]
    pnl_24h: float = 0.0
    pnl_7d: float = 0.0
    pnl_30d: float = 0.0
    pnl_total: float = 0.0


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional Value at Risk (95%)


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    portfolio_beta: float = 0.0
    correlation_with_sol: float = 0.0
    concentration_risk: float = 0.0
    liquidity_risk: float = 0.0
    sector_exposure: Dict[str, float] = None
    
    def __post_init__(self):
        if self.sector_exposure is None:
            self.sector_exposure = {}


class PortfolioAnalytics:
    """Portfolio analytics and dashboard manager."""
    
    def __init__(self):
        """Initialize the portfolio analytics."""
        self.enabled = get_config_value("portfolio_analytics_enabled", False)
        self.data_path = get_config_value("portfolio_analytics_data_path", "portfolio_data")
        
        # Historical data
        self.snapshots = []  # List of PortfolioSnapshot
        self.price_history = defaultdict(list)  # token_mint -> [(timestamp, price)]
        self.trade_history = []
        
        # Cache
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Benchmark data (SOL price for comparison)
        self.benchmark_history = []  # [(timestamp, sol_price)]
        
        logger.info("Portfolio analytics initialized")
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable portfolio analytics.
        
        Args:
            enabled: Whether portfolio analytics should be enabled
        """
        self.enabled = enabled
        update_config("portfolio_analytics_enabled", enabled)
        logger.info(f"Portfolio analytics {'enabled' if enabled else 'disabled'}")
    
    def capture_snapshot(self) -> PortfolioSnapshot:
        """
        Capture a current portfolio snapshot.
        
        Returns:
            Portfolio snapshot
        """
        try:
            current_time = datetime.now()
            
            # Get SOL balance
            sol_balance = wallet_manager.get_sol_balance() if wallet_manager.current_keypair else 0.0
            
            # Get all positions and their current values
            positions = {}
            total_token_value = 0.0
            
            for token_mint, position in position_manager.positions.items():
                try:
                    current_price = jupiter_api.get_token_price(token_mint)
                    position_value = position.amount * current_price
                    total_token_value += position_value
                    
                    positions[token_mint] = {
                        'token_name': position.token_name,
                        'amount': position.amount,
                        'current_price': current_price,
                        'value_sol': position_value,
                        'entry_price': position.entry_price,
                        'pnl': (current_price - position.entry_price) * position.amount,
                        'pnl_percentage': ((current_price - position.entry_price) / position.entry_price) * 100
                    }
                except Exception as e:
                    logger.warning(f"Error getting price for {token_mint}: {e}")
                    positions[token_mint] = {
                        'token_name': position.token_name,
                        'amount': position.amount,
                        'current_price': 0.0,
                        'value_sol': 0.0,
                        'entry_price': position.entry_price,
                        'pnl': 0.0,
                        'pnl_percentage': 0.0
                    }
            
            total_value = sol_balance + total_token_value
            
            # Calculate P&L for different periods
            pnl_24h = self._calculate_pnl_for_period(timedelta(days=1))
            pnl_7d = self._calculate_pnl_for_period(timedelta(days=7))
            pnl_30d = self._calculate_pnl_for_period(timedelta(days=30))
            pnl_total = self._calculate_total_pnl()
            
            snapshot = PortfolioSnapshot(
                timestamp=current_time,
                total_value_sol=total_value,
                sol_balance=sol_balance,
                token_value_sol=total_token_value,
                positions=positions,
                pnl_24h=pnl_24h,
                pnl_7d=pnl_7d,
                pnl_30d=pnl_30d,
                pnl_total=pnl_total
            )
            
            # Store snapshot
            self.snapshots.append(snapshot)
            
            # Keep only last 1000 snapshots to manage memory
            if len(self.snapshots) > 1000:
                self.snapshots = self.snapshots[-1000:]
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error capturing portfolio snapshot: {e}")
            raise
    
    def get_current_allocation(self) -> Dict[str, float]:
        """
        Get current portfolio allocation by token.
        
        Returns:
            Dictionary mapping token names to allocation percentages
        """
        snapshot = self.capture_snapshot()
        
        if snapshot.total_value_sol == 0:
            return {}
        
        allocation = {}
        
        # SOL allocation
        if snapshot.sol_balance > 0:
            allocation['SOL'] = (snapshot.sol_balance / snapshot.total_value_sol) * 100
        
        # Token allocations
        for token_mint, position_data in snapshot.positions.items():
            if position_data['value_sol'] > 0:
                allocation[position_data['token_name']] = (position_data['value_sol'] / snapshot.total_value_sol) * 100
        
        return allocation
    
    def calculate_performance_metrics(self, days: int = 30) -> PerformanceMetrics:
        """
        Calculate portfolio performance metrics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Performance metrics
        """
        if len(self.snapshots) < 2:
            return PerformanceMetrics()
        
        # Get snapshots for the specified period
        cutoff_date = datetime.now() - timedelta(days=days)
        period_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_date]
        
        if len(period_snapshots) < 2:
            period_snapshots = self.snapshots[-min(len(self.snapshots), 30):]
        
        # Calculate returns
        values = [s.total_value_sol for s in period_snapshots]
        returns = np.diff(values) / values[:-1]
        
        metrics = PerformanceMetrics()
        
        if len(returns) > 0:
            # Total return
            metrics.total_return = (values[-1] - values[0]) / values[0]
            
            # Annualized return
            days_actual = (period_snapshots[-1].timestamp - period_snapshots[0].timestamp).days
            if days_actual > 0:
                metrics.annualized_return = (1 + metrics.total_return) ** (365 / days_actual) - 1
            
            # Volatility (annualized)
            if len(returns) > 1:
                metrics.volatility = np.std(returns) * np.sqrt(365)
            
            # Sharpe ratio (assuming 0% risk-free rate)
            if metrics.volatility > 0:
                metrics.sharpe_ratio = metrics.annualized_return / metrics.volatility
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 1:
                downside_deviation = np.std(negative_returns) * np.sqrt(365)
                if downside_deviation > 0:
                    metrics.sortino_ratio = metrics.annualized_return / downside_deviation
            
            # Maximum drawdown
            peak = np.maximum.accumulate(values)
            drawdown = (peak - values) / peak
            metrics.max_drawdown = np.max(drawdown)
            
            # Calmar ratio
            if metrics.max_drawdown > 0:
                metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
            
            # Value at Risk (95%)
            if len(returns) > 0:
                metrics.var_95 = np.percentile(returns, 5)
                
                # Conditional Value at Risk (95%)
                var_returns = returns[returns <= metrics.var_95]
                if len(var_returns) > 0:
                    metrics.cvar_95 = np.mean(var_returns)
        
        # Calculate win rate and profit factor from trades
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
            losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]
            
            total_trades = len(self.trade_history)
            if total_trades > 0:
                metrics.win_rate = len(winning_trades) / total_trades
            
            total_wins = sum(t.get('pnl', 0) for t in winning_trades)
            total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
            
            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses
        
        return metrics
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """
        Calculate portfolio risk metrics.
        
        Returns:
            Risk metrics
        """
        metrics = RiskMetrics()
        
        if len(self.snapshots) < 10:
            return metrics
        
        try:
            # Get recent snapshots
            recent_snapshots = self.snapshots[-30:]  # Last 30 snapshots
            
            # Calculate concentration risk (Herfindahl index)
            latest_snapshot = recent_snapshots[-1]
            total_value = latest_snapshot.total_value_sol
            
            if total_value > 0:
                allocations = []
                
                # SOL allocation
                if latest_snapshot.sol_balance > 0:
                    allocations.append(latest_snapshot.sol_balance / total_value)
                
                # Token allocations
                for position_data in latest_snapshot.positions.values():
                    if position_data['value_sol'] > 0:
                        allocations.append(position_data['value_sol'] / total_value)
                
                # Herfindahl index (0 = perfectly diversified, 1 = concentrated)
                metrics.concentration_risk = sum(w**2 for w in allocations)
            
            # Calculate correlation with SOL (if we have benchmark data)
            if len(self.benchmark_history) >= len(recent_snapshots):
                portfolio_values = [s.total_value_sol for s in recent_snapshots]
                sol_prices = [p[1] for p in self.benchmark_history[-len(recent_snapshots):]]
                
                if len(portfolio_values) == len(sol_prices) and len(portfolio_values) > 1:
                    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
                    sol_returns = np.diff(sol_prices) / sol_prices[:-1]
                    
                    if len(portfolio_returns) > 1 and len(sol_returns) > 1:
                        correlation_matrix = np.corrcoef(portfolio_returns, sol_returns)
                        metrics.correlation_with_sol = correlation_matrix[0, 1]
                        
                        # Calculate beta
                        if np.var(sol_returns) > 0:
                            metrics.portfolio_beta = np.cov(portfolio_returns, sol_returns)[0, 1] / np.var(sol_returns)
            
            # Liquidity risk (simplified - based on position sizes)
            large_positions = 0
            total_positions = len(latest_snapshot.positions)
            
            for position_data in latest_snapshot.positions.values():
                allocation = position_data['value_sol'] / total_value if total_value > 0 else 0
                if allocation > 0.1:  # Positions > 10% are considered large
                    large_positions += 1
            
            if total_positions > 0:
                metrics.liquidity_risk = large_positions / total_positions
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
        
        return metrics
    
    def _calculate_pnl_for_period(self, period: timedelta) -> float:
        """Calculate P&L for a specific time period."""
        if len(self.snapshots) < 2:
            return 0.0
        
        cutoff_time = datetime.now() - period
        
        # Find snapshot closest to cutoff time
        past_snapshot = None
        for snapshot in reversed(self.snapshots):
            if snapshot.timestamp <= cutoff_time:
                past_snapshot = snapshot
                break
        
        if past_snapshot is None:
            past_snapshot = self.snapshots[0]
        
        current_snapshot = self.snapshots[-1]
        
        return current_snapshot.total_value_sol - past_snapshot.total_value_sol
    
    def _calculate_total_pnl(self) -> float:
        """Calculate total P&L since tracking began."""
        if len(self.snapshots) < 2:
            return 0.0
        
        return self.snapshots[-1].total_value_sol - self.snapshots[0].total_value_sol
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data.
        
        Returns:
            Dashboard data dictionary
        """
        try:
            current_snapshot = self.capture_snapshot()
            allocation = self.get_current_allocation()
            performance = self.calculate_performance_metrics()
            risk = self.calculate_risk_metrics()
            
            return {
                'portfolio': {
                    'total_value_sol': current_snapshot.total_value_sol,
                    'sol_balance': current_snapshot.sol_balance,
                    'token_value_sol': current_snapshot.token_value_sol,
                    'position_count': len(current_snapshot.positions),
                    'pnl_24h': current_snapshot.pnl_24h,
                    'pnl_7d': current_snapshot.pnl_7d,
                    'pnl_30d': current_snapshot.pnl_30d,
                    'pnl_total': current_snapshot.pnl_total
                },
                'allocation': allocation,
                'performance': asdict(performance),
                'risk': asdict(risk),
                'positions': current_snapshot.positions,
                'last_updated': current_snapshot.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}


# Create a singleton instance
portfolio_analytics = PortfolioAnalytics()
