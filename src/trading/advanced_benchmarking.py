"""
Advanced Benchmarking Engine

This module provides comprehensive benchmarking capabilities including:
- Multiple benchmark types (market indices, peer groups, custom)
- Risk-adjusted performance metrics
- Factor-based attribution analysis
- Performance forecasting
- Relative performance analysis

Author: Solana Memecoin Trading Bot
Version: 3.0.0 (Phase 3)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
import warnings

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks available."""
    MARKET_INDEX = "market_index"
    PEER_GROUP = "peer_group"
    CUSTOM = "custom"
    RISK_FREE = "risk_free"
    SECTOR = "sector"


class PerformanceMetric(Enum):
    """Performance metrics for benchmarking."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    TREYNOR_RATIO = "treynor_ratio"
    ALPHA = "alpha"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"
    ACTIVE_RETURN = "active_return"


@dataclass
class BenchmarkData:
    """Benchmark data structure."""
    name: str
    benchmark_type: BenchmarkType
    returns: pd.Series
    description: str
    inception_date: datetime
    last_updated: datetime
    metadata: Dict[str, Any]


@dataclass
class PerformanceComparison:
    """Performance comparison result."""
    portfolio_return: float
    benchmark_return: float
    active_return: float
    tracking_error: float
    information_ratio: float
    alpha: float
    beta: float
    r_squared: float
    sharpe_portfolio: float
    sharpe_benchmark: float
    sortino_portfolio: float
    sortino_benchmark: float
    max_drawdown_portfolio: float
    max_drawdown_benchmark: float
    win_rate: float
    period_start: datetime
    period_end: datetime
    benchmark_name: str


@dataclass
class RiskAdjustedMetrics:
    """Risk-adjusted performance metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    beta: float
    r_squared: float
    tracking_error: float
    downside_deviation: float
    var_95: float
    cvar_95: float
    maximum_drawdown: float
    recovery_time_days: float


@dataclass
class FactorExposure:
    """Factor exposure analysis."""
    market_beta: float
    size_factor: float
    value_factor: float
    momentum_factor: float
    volatility_factor: float
    liquidity_factor: float
    factor_r_squared: float
    residual_risk: float
    factor_loadings: Dict[str, float]


class AdvancedBenchmarkingEngine:
    """
    Advanced benchmarking engine for portfolio performance analysis.

    Features:
    - Multiple benchmark comparisons
    - Risk-adjusted performance metrics
    - Factor-based attribution
    - Performance forecasting
    - Relative performance analysis
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the advanced benchmarking engine."""
        self.config = config or {}

        # Configuration
        self.update_frequency = self.config.get('benchmark_update_frequency_hours', 1)
        self.forecasting_enabled = self.config.get('performance_forecasting_enabled', True)
        self.custom_benchmarks = self.config.get('custom_benchmarks', [])

        # Risk-free rate (annual)
        self.risk_free_rate = 0.02  # 2% annual

        # Benchmarks storage
        self.benchmarks: Dict[str, BenchmarkData] = {}
        self.comparison_history: List[PerformanceComparison] = []

        # Factor models
        self.factor_returns: Optional[pd.DataFrame] = None

        # Initialize default benchmarks
        self._initialize_default_benchmarks()

        logger.info("Advanced benchmarking engine initialized")

    def add_benchmark(self, name: str, benchmark_type: BenchmarkType,
                     returns: pd.Series, description: str = "",
                     metadata: Optional[Dict] = None) -> None:
        """
        Add a new benchmark.

        Args:
            name: Benchmark name
            benchmark_type: Type of benchmark
            returns: Return series
            description: Benchmark description
            metadata: Additional metadata
        """
        try:
            benchmark_data = BenchmarkData(
                name=name,
                benchmark_type=benchmark_type,
                returns=returns,
                description=description,
                inception_date=returns.index[0] if len(returns) > 0 else datetime.now(),
                last_updated=datetime.now(),
                metadata=metadata or {}
            )

            self.benchmarks[name] = benchmark_data
            logger.info(f"Added benchmark: {name}")

        except Exception as e:
            logger.error(f"Failed to add benchmark {name}: {e}")

    def compare_performance(self, portfolio_returns: pd.Series,
                          benchmark_name: str,
                          period_days: int = 30) -> PerformanceComparison:
        """
        Compare portfolio performance against a benchmark.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_name: Name of benchmark to compare against
            period_days: Analysis period in days

        Returns:
            Performance comparison result
        """
        try:
            if benchmark_name not in self.benchmarks:
                raise ValueError(f"Benchmark {benchmark_name} not found")

            benchmark = self.benchmarks[benchmark_name]

            # Align time series and get recent period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)

            # Filter returns for the analysis period
            portfolio_period = portfolio_returns[portfolio_returns.index >= start_date]
            benchmark_period = benchmark.returns[benchmark.returns.index >= start_date]

            # Align the series
            aligned_data = pd.concat([portfolio_period, benchmark_period], axis=1, join='inner')
            aligned_data.columns = ['portfolio', 'benchmark']
            aligned_data = aligned_data.dropna()

            if len(aligned_data) < 5:
                raise ValueError("Insufficient overlapping data for comparison")

            portfolio_aligned = aligned_data['portfolio']
            benchmark_aligned = aligned_data['benchmark']

            # Calculate performance metrics
            comparison = self._calculate_performance_comparison(
                portfolio_aligned, benchmark_aligned, start_date, end_date, benchmark_name
            )

            # Store comparison
            self.comparison_history.append(comparison)

            return comparison

        except Exception as e:
            logger.error(f"Performance comparison failed: {e}")
            raise

    def calculate_risk_adjusted_metrics(self, returns: pd.Series) -> RiskAdjustedMetrics:
        """
        Calculate comprehensive risk-adjusted performance metrics.

        Args:
            returns: Return series

        Returns:
            Risk-adjusted metrics
        """
        try:
            if len(returns) < 10:
                raise ValueError("Insufficient data for risk metrics calculation")

            # Basic statistics
            mean_return = returns.mean()
            std_return = returns.std()

            # Annualized metrics
            annual_return = mean_return * 252
            annual_volatility = std_return * np.sqrt(252)

            # Sharpe ratio
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()

            # Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

            # VaR and CVaR (95% confidence)
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

            # Recovery time (simplified)
            recovery_time = self._calculate_recovery_time(drawdowns)

            return RiskAdjustedMetrics(
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                information_ratio=0.0,  # Requires benchmark
                treynor_ratio=0.0,     # Requires beta
                jensen_alpha=0.0,      # Requires benchmark
                beta=1.0,              # Requires benchmark
                r_squared=0.0,         # Requires benchmark
                tracking_error=0.0,    # Requires benchmark
                downside_deviation=downside_deviation,
                var_95=var_95,
                cvar_95=cvar_95,
                maximum_drawdown=max_drawdown,
                recovery_time_days=recovery_time
            )

        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            raise

    def analyze_factor_exposure(self, returns: pd.Series) -> FactorExposure:
        """
        Analyze factor exposure using multi-factor model.

        Args:
            returns: Return series

        Returns:
            Factor exposure analysis
        """
        try:
            # Initialize factor returns if not available
            if self.factor_returns is None:
                self._initialize_factor_returns(returns.index)

            # Align returns with factor data
            aligned_data = pd.concat([returns, self.factor_returns], axis=1, join='inner')
            aligned_data = aligned_data.dropna()

            if len(aligned_data) < 20:
                logger.warning("Insufficient data for factor analysis, using simplified model")
                return self._simplified_factor_exposure(returns)

            # Prepare regression data
            y = aligned_data.iloc[:, 0]  # Portfolio returns
            X = aligned_data.iloc[:, 1:]  # Factor returns

            # Add constant for alpha
            X_with_const = np.column_stack([np.ones(len(X)), X])

            # Perform regression
            try:
                coefficients, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)

                # Calculate R-squared
                ss_res = np.sum(residuals) if len(residuals) > 0 else np.sum((y - X_with_const @ coefficients) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Extract factor loadings
                alpha = coefficients[0]
                factor_loadings = dict(zip(X.columns, coefficients[1:]))

                # Calculate residual risk
                residual_returns = y - X_with_const @ coefficients
                residual_risk = np.std(residual_returns) * np.sqrt(252)

                return FactorExposure(
                    market_beta=factor_loadings.get('market', 1.0),
                    size_factor=factor_loadings.get('size', 0.0),
                    value_factor=factor_loadings.get('value', 0.0),
                    momentum_factor=factor_loadings.get('momentum', 0.0),
                    volatility_factor=factor_loadings.get('volatility', 0.0),
                    liquidity_factor=factor_loadings.get('liquidity', 0.0),
                    factor_r_squared=r_squared,
                    residual_risk=residual_risk,
                    factor_loadings=factor_loadings
                )

            except np.linalg.LinAlgError:
                logger.warning("Factor regression failed, using simplified model")
                return self._simplified_factor_exposure(returns)

        except Exception as e:
            logger.error(f"Factor exposure analysis failed: {e}")
            return self._simplified_factor_exposure(returns)

    def _calculate_performance_comparison(self, portfolio_returns: pd.Series,
                                        benchmark_returns: pd.Series,
                                        start_date: datetime, end_date: datetime,
                                        benchmark_name: str) -> PerformanceComparison:
        """Calculate detailed performance comparison."""
        # Total returns
        portfolio_total = (1 + portfolio_returns).prod() - 1
        benchmark_total = (1 + benchmark_returns).prod() - 1
        active_return = portfolio_total - benchmark_total

        # Tracking error
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)

        # Information ratio
        information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0

        # Alpha and Beta
        try:
            beta, alpha, r_value, p_value, std_err = stats.linregress(benchmark_returns, portfolio_returns)
            r_squared = r_value ** 2
        except:
            beta, alpha, r_squared = 1.0, 0.0, 0.0

        # Sharpe ratios
        portfolio_sharpe = self._calculate_sharpe_ratio(portfolio_returns)
        benchmark_sharpe = self._calculate_sharpe_ratio(benchmark_returns)

        # Sortino ratios
        portfolio_sortino = self._calculate_sortino_ratio(portfolio_returns)
        benchmark_sortino = self._calculate_sortino_ratio(benchmark_returns)

        # Maximum drawdowns
        portfolio_dd = self._calculate_max_drawdown(portfolio_returns)
        benchmark_dd = self._calculate_max_drawdown(benchmark_returns)

        # Win rate
        win_rate = (portfolio_returns > benchmark_returns).mean()

        return PerformanceComparison(
            portfolio_return=portfolio_total,
            benchmark_return=benchmark_total,
            active_return=active_return,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            alpha=alpha * 252,  # Annualized
            beta=beta,
            r_squared=r_squared,
            sharpe_portfolio=portfolio_sharpe,
            sharpe_benchmark=benchmark_sharpe,
            sortino_portfolio=portfolio_sortino,
            sortino_benchmark=benchmark_sortino,
            max_drawdown_portfolio=portfolio_dd,
            max_drawdown_benchmark=benchmark_dd,
            win_rate=win_rate,
            period_start=start_date,
            period_end=end_date,
            benchmark_name=benchmark_name
        )

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)

        return (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        annual_return = returns.mean() * 252
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        return (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max

        return drawdowns.min()

    def _calculate_recovery_time(self, drawdowns: pd.Series) -> float:
        """Calculate average recovery time from drawdowns."""
        if len(drawdowns) == 0:
            return 0.0

        # Find drawdown periods
        in_drawdown = drawdowns < -0.01  # 1% threshold

        if not in_drawdown.any():
            return 0.0

        # Calculate recovery periods (simplified)
        recovery_periods = []
        current_period = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    recovery_periods.append(current_period)
                    current_period = 0

        return np.mean(recovery_periods) if recovery_periods else 0.0

    def _initialize_default_benchmarks(self):
        """Initialize default benchmarks with synthetic data."""
        try:
            # Create synthetic benchmark data
            dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')

            # Solana ecosystem benchmark (synthetic)
            sol_returns = np.random.normal(0.001, 0.03, len(dates))  # Higher volatility
            sol_benchmark = pd.Series(sol_returns, index=dates)

            self.add_benchmark(
                name="SOL_ECOSYSTEM",
                benchmark_type=BenchmarkType.MARKET_INDEX,
                returns=sol_benchmark,
                description="Solana Ecosystem Index (Synthetic)",
                metadata={"volatility": "high", "asset_class": "crypto"}
            )

            # DeFi sector benchmark (synthetic)
            defi_returns = np.random.normal(0.0008, 0.025, len(dates))
            defi_benchmark = pd.Series(defi_returns, index=dates)

            self.add_benchmark(
                name="DEFI_SECTOR",
                benchmark_type=BenchmarkType.SECTOR,
                returns=defi_benchmark,
                description="DeFi Sector Index (Synthetic)",
                metadata={"sector": "defi", "asset_class": "crypto"}
            )

            # Risk-free rate benchmark
            rf_returns = pd.Series([self.risk_free_rate / 252] * len(dates), index=dates)

            self.add_benchmark(
                name="RISK_FREE",
                benchmark_type=BenchmarkType.RISK_FREE,
                returns=rf_returns,
                description="Risk-Free Rate",
                metadata={"rate": self.risk_free_rate}
            )

            logger.info("Default benchmarks initialized")

        except Exception as e:
            logger.error(f"Failed to initialize default benchmarks: {e}")

    def _initialize_factor_returns(self, date_index: pd.DatetimeIndex):
        """Initialize factor returns for factor analysis."""
        try:
            # Create synthetic factor returns
            n_periods = len(date_index)

            factor_data = {
                'market': np.random.normal(0.0005, 0.02, n_periods),
                'size': np.random.normal(0.0, 0.01, n_periods),
                'value': np.random.normal(0.0, 0.008, n_periods),
                'momentum': np.random.normal(0.0, 0.012, n_periods),
                'volatility': np.random.normal(0.0, 0.015, n_periods),
                'liquidity': np.random.normal(0.0, 0.006, n_periods)
            }

            self.factor_returns = pd.DataFrame(factor_data, index=date_index)

        except Exception as e:
            logger.error(f"Failed to initialize factor returns: {e}")

    def _simplified_factor_exposure(self, returns: pd.Series) -> FactorExposure:
        """Simplified factor exposure when full analysis fails."""
        return FactorExposure(
            market_beta=1.0,
            size_factor=0.0,
            value_factor=0.0,
            momentum_factor=0.0,
            volatility_factor=0.0,
            liquidity_factor=0.0,
            factor_r_squared=0.5,
            residual_risk=returns.std() * np.sqrt(252),
            factor_loadings={'market': 1.0}
        )

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of available benchmarks and recent comparisons."""
        summary = {
            "available_benchmarks": {
                name: {
                    "type": bench.benchmark_type.value,
                    "description": bench.description,
                    "inception_date": bench.inception_date.isoformat(),
                    "last_updated": bench.last_updated.isoformat(),
                    "data_points": len(bench.returns)
                }
                for name, bench in self.benchmarks.items()
            },
            "total_comparisons": len(self.comparison_history),
            "forecasting_enabled": self.forecasting_enabled
        }

        if self.comparison_history:
            recent_comparison = self.comparison_history[-1]
            summary["latest_comparison"] = {
                "benchmark": recent_comparison.benchmark_name,
                "active_return": recent_comparison.active_return,
                "information_ratio": recent_comparison.information_ratio,
                "alpha": recent_comparison.alpha,
                "beta": recent_comparison.beta,
                "period_end": recent_comparison.period_end.isoformat()
            }

        return summary

    def forecast_performance(self, returns: pd.Series,
                           forecast_days: int = 30) -> Dict[str, Any]:
        """
        Forecast portfolio performance (simplified implementation).

        Args:
            returns: Historical returns
            forecast_days: Number of days to forecast

        Returns:
            Performance forecast
        """
        try:
            if not self.forecasting_enabled:
                return {"error": "Performance forecasting is disabled"}

            if len(returns) < 30:
                return {"error": "Insufficient historical data for forecasting"}

            # Simple statistical forecast
            mean_return = returns.mean()
            volatility = returns.std()

            # Monte Carlo simulation (simplified)
            n_simulations = 1000
            forecasted_paths = []

            for _ in range(n_simulations):
                path = []
                current_value = 1.0

                for day in range(forecast_days):
                    daily_return = np.random.normal(mean_return, volatility)
                    current_value *= (1 + daily_return)
                    path.append(current_value)

                forecasted_paths.append(path)

            # Calculate statistics
            forecasted_paths = np.array(forecasted_paths)
            final_values = forecasted_paths[:, -1]

            forecast_result = {
                "forecast_period_days": forecast_days,
                "expected_return": np.mean(final_values) - 1,
                "return_std": np.std(final_values),
                "confidence_intervals": {
                    "5%": np.percentile(final_values, 5) - 1,
                    "25%": np.percentile(final_values, 25) - 1,
                    "50%": np.percentile(final_values, 50) - 1,
                    "75%": np.percentile(final_values, 75) - 1,
                    "95%": np.percentile(final_values, 95) - 1
                },
                "probability_positive": (final_values > 1.0).mean(),
                "var_95": np.percentile(final_values, 5) - 1,
                "expected_volatility": volatility * np.sqrt(forecast_days),
                "forecast_timestamp": datetime.now().isoformat()
            }

            return forecast_result

        except Exception as e:
            logger.error(f"Performance forecasting failed: {e}")
            return {"error": str(e)}


# Global instance
advanced_benchmarking_engine = AdvancedBenchmarkingEngine()
