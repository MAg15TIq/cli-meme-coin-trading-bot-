"""
Performance Attribution Analysis for the Solana Memecoin Trading Bot.
Implements Brinson-Hood-Beebower attribution model and factor-based analysis.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from config import get_config_value
from src.utils.logging_utils import get_logger
from src.trading.portfolio_analytics import portfolio_analytics

# Get logger for this module
logger = get_logger(__name__)


@dataclass
class AttributionResult:
    """Performance attribution analysis result."""
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_active_return: float
    period_start: datetime
    period_end: datetime
    method: str


@dataclass
class FactorAttribution:
    """Factor-based attribution result."""
    factor_name: str
    factor_return: float
    factor_exposure: float
    factor_contribution: float


@dataclass
class BenchmarkComparison:
    """Benchmark comparison result."""
    benchmark_name: str
    portfolio_return: float
    benchmark_return: float
    active_return: float
    tracking_error: float
    information_ratio: float
    beta: float
    alpha: float
    correlation: float


class PerformanceAttributionAnalyzer:
    """Advanced performance attribution analysis."""

    def __init__(self):
        """Initialize the performance attribution analyzer."""
        self.attribution_models = {
            'brinson': self._brinson_attribution,
            'factor_based': self._factor_attribution,
            'time_based': self._time_attribution
        }
        
        # Common risk factors for crypto/DeFi
        self.risk_factors = {
            'market': 'SOL price movement',
            'size': 'Market cap factor',
            'momentum': 'Price momentum factor',
            'volatility': 'Volatility factor',
            'liquidity': 'Liquidity factor'
        }
        
        logger.info("Performance attribution analyzer initialized")

    def analyze_performance_attribution(self, portfolio_returns: pd.Series,
                                      benchmark_returns: pd.Series,
                                      method: str = 'brinson',
                                      period_days: int = 30) -> AttributionResult:
        """
        Perform detailed performance attribution analysis.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            method: Attribution method to use
            period_days: Analysis period in days

        Returns:
            Attribution analysis result
        """
        if method not in self.attribution_models:
            raise ValueError(f"Unknown attribution method: {method}")

        # Align time series and get recent period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Filter returns for the analysis period
        portfolio_period = portfolio_returns[portfolio_returns.index >= start_date]
        benchmark_period = benchmark_returns[benchmark_returns.index >= start_date]
        
        if len(portfolio_period) < 5 or len(benchmark_period) < 5:
            logger.warning("Insufficient data for attribution analysis")
            return AttributionResult(0.0, 0.0, 0.0, 0.0, start_date, end_date, method)

        attribution_func = self.attribution_models[method]
        return attribution_func(portfolio_period, benchmark_period, start_date, end_date)

    def _brinson_attribution(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                           start_date: datetime, end_date: datetime) -> AttributionResult:
        """
        Brinson-Hood-Beebower attribution model.
        
        This model decomposes active return into:
        - Allocation effect: Return from over/under-weighting sectors
        - Selection effect: Return from security selection within sectors
        - Interaction effect: Combined effect of allocation and selection
        """
        try:
            # Get portfolio and benchmark data
            portfolio_data = self._get_portfolio_sector_data(start_date, end_date)
            benchmark_data = self._get_benchmark_sector_data(start_date, end_date)
            
            if not portfolio_data or not benchmark_data:
                logger.warning("Insufficient sector data for Brinson attribution")
                return AttributionResult(0.0, 0.0, 0.0, 0.0, start_date, end_date, "brinson")

            # Calculate allocation effect
            allocation_effect = self._calculate_allocation_effect(portfolio_data, benchmark_data)
            
            # Calculate selection effect
            selection_effect = self._calculate_selection_effect(portfolio_data, benchmark_data)
            
            # Calculate interaction effect
            interaction_effect = self._calculate_interaction_effect(portfolio_data, benchmark_data)
            
            total_active_return = allocation_effect + selection_effect + interaction_effect

            return AttributionResult(
                allocation_effect=allocation_effect,
                selection_effect=selection_effect,
                interaction_effect=interaction_effect,
                total_active_return=total_active_return,
                period_start=start_date,
                period_end=end_date,
                method="brinson"
            )

        except Exception as e:
            logger.error(f"Error in Brinson attribution: {e}")
            return AttributionResult(0.0, 0.0, 0.0, 0.0, start_date, end_date, "brinson")

    def _calculate_allocation_effect(self, portfolio_data: Dict, benchmark_data: Dict) -> float:
        """Calculate allocation effect: (wp - wb) * rb"""
        allocation_effect = 0.0
        
        for sector in benchmark_data.keys():
            wp = portfolio_data.get(sector, {}).get('weight', 0.0)  # Portfolio weight
            wb = benchmark_data.get(sector, {}).get('weight', 0.0)  # Benchmark weight
            rb = benchmark_data.get(sector, {}).get('return', 0.0)  # Benchmark sector return
            
            allocation_effect += (wp - wb) * rb
            
        return allocation_effect

    def _calculate_selection_effect(self, portfolio_data: Dict, benchmark_data: Dict) -> float:
        """Calculate selection effect: wb * (rp - rb)"""
        selection_effect = 0.0
        
        for sector in benchmark_data.keys():
            wb = benchmark_data.get(sector, {}).get('weight', 0.0)  # Benchmark weight
            rp = portfolio_data.get(sector, {}).get('return', 0.0)  # Portfolio sector return
            rb = benchmark_data.get(sector, {}).get('return', 0.0)  # Benchmark sector return
            
            selection_effect += wb * (rp - rb)
            
        return selection_effect

    def _calculate_interaction_effect(self, portfolio_data: Dict, benchmark_data: Dict) -> float:
        """Calculate interaction effect: (wp - wb) * (rp - rb)"""
        interaction_effect = 0.0
        
        for sector in benchmark_data.keys():
            wp = portfolio_data.get(sector, {}).get('weight', 0.0)  # Portfolio weight
            wb = benchmark_data.get(sector, {}).get('weight', 0.0)  # Benchmark weight
            rp = portfolio_data.get(sector, {}).get('return', 0.0)  # Portfolio sector return
            rb = benchmark_data.get(sector, {}).get('return', 0.0)  # Benchmark sector return
            
            interaction_effect += (wp - wb) * (rp - rb)
            
        return interaction_effect

    def _get_portfolio_sector_data(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get portfolio sector allocation and returns."""
        try:
            # Get current positions
            from src.trading.position_manager import position_manager
            positions = position_manager.get_all_positions()
            
            # Categorize positions by sector (simplified)
            sectors = {
                'defi': {'weight': 0.0, 'return': 0.0, 'positions': []},
                'meme': {'weight': 0.0, 'return': 0.0, 'positions': []},
                'gaming': {'weight': 0.0, 'return': 0.0, 'positions': []},
                'infrastructure': {'weight': 0.0, 'return': 0.0, 'positions': []},
                'other': {'weight': 0.0, 'return': 0.0, 'positions': []}
            }
            
            total_value = sum(pos.current_value for pos in positions.values())
            
            if total_value == 0:
                return {}
            
            # Categorize positions (simplified categorization)
            for token_mint, position in positions.items():
                sector = self._categorize_token(position.token_name)
                sectors[sector]['positions'].append(position)
                sectors[sector]['weight'] += position.current_value / total_value
                
                # Calculate position return (simplified)
                if position.entry_price > 0:
                    position_return = (position.current_price - position.entry_price) / position.entry_price
                    sectors[sector]['return'] += position_return * (position.current_value / total_value)
            
            return sectors
            
        except Exception as e:
            logger.error(f"Error getting portfolio sector data: {e}")
            return {}

    def _get_benchmark_sector_data(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get benchmark sector allocation and returns."""
        # Simplified benchmark sector data (would need real benchmark data)
        return {
            'defi': {'weight': 0.4, 'return': 0.05},
            'meme': {'weight': 0.3, 'return': 0.10},
            'gaming': {'weight': 0.15, 'return': 0.03},
            'infrastructure': {'weight': 0.10, 'return': 0.02},
            'other': {'weight': 0.05, 'return': 0.01}
        }

    def _categorize_token(self, token_name: str) -> str:
        """Categorize token by sector (simplified)."""
        token_lower = token_name.lower()
        
        if any(keyword in token_lower for keyword in ['swap', 'dex', 'lend', 'yield', 'farm']):
            return 'defi'
        elif any(keyword in token_lower for keyword in ['dog', 'cat', 'pepe', 'shib', 'meme']):
            return 'meme'
        elif any(keyword in token_lower for keyword in ['game', 'play', 'nft', 'meta']):
            return 'gaming'
        elif any(keyword in token_lower for keyword in ['sol', 'validator', 'stake', 'node']):
            return 'infrastructure'
        else:
            return 'other'

    def _factor_attribution(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                          start_date: datetime, end_date: datetime) -> AttributionResult:
        """Factor-based attribution analysis."""
        try:
            # This would implement factor model attribution
            # For now, return simplified result
            total_return = portfolio_returns.sum()
            benchmark_return = benchmark_returns.sum()
            active_return = total_return - benchmark_return
            
            # Simplified factor attribution
            allocation_effect = active_return * 0.4  # 40% from allocation
            selection_effect = active_return * 0.6   # 60% from selection
            interaction_effect = 0.0
            
            return AttributionResult(
                allocation_effect=allocation_effect,
                selection_effect=selection_effect,
                interaction_effect=interaction_effect,
                total_active_return=active_return,
                period_start=start_date,
                period_end=end_date,
                method="factor_based"
            )
            
        except Exception as e:
            logger.error(f"Error in factor attribution: {e}")
            return AttributionResult(0.0, 0.0, 0.0, 0.0, start_date, end_date, "factor_based")

    def _time_attribution(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                        start_date: datetime, end_date: datetime) -> AttributionResult:
        """Time-based attribution analysis."""
        try:
            # Analyze performance over different time periods
            active_returns = portfolio_returns - benchmark_returns
            
            # Split into time periods and analyze
            total_active_return = active_returns.sum()
            
            # Simplified time attribution
            allocation_effect = total_active_return * 0.3
            selection_effect = total_active_return * 0.5
            interaction_effect = total_active_return * 0.2
            
            return AttributionResult(
                allocation_effect=allocation_effect,
                selection_effect=selection_effect,
                interaction_effect=interaction_effect,
                total_active_return=total_active_return,
                period_start=start_date,
                period_end=end_date,
                method="time_based"
            )
            
        except Exception as e:
            logger.error(f"Error in time attribution: {e}")
            return AttributionResult(0.0, 0.0, 0.0, 0.0, start_date, end_date, "time_based")


# Create a singleton instance
performance_attribution_analyzer = PerformanceAttributionAnalyzer()
