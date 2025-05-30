"""
Advanced Risk Metrics for the Solana Memecoin Trading Bot.
Implements sophisticated risk assessment including VaR, CVaR, stress testing, and Monte Carlo simulations.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize

from config import get_config_value
from src.utils.logging_utils import get_logger
from src.trading.portfolio_analytics import portfolio_analytics

# Get logger for this module
logger = get_logger(__name__)


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    var: float
    cvar: float
    confidence_level: float
    time_horizon_days: int
    observations: int
    method: str


@dataclass
class StressTestResult:
    """Stress test scenario result."""
    scenario_name: str
    total_pnl: float
    pnl_percentage: float
    worst_position: str
    worst_position_pnl: float
    scenario_details: Dict[str, Any]


@dataclass
class MaxDrawdownPrediction:
    """Maximum drawdown prediction result."""
    predicted_mdd: float
    confidence_level: float
    mean_mdd: float
    std_mdd: float
    simulations: int
    time_horizon_days: int


class AdvancedRiskMetrics:
    """Advanced risk metrics calculation and monitoring."""

    def __init__(self):
        """Initialize the advanced risk metrics calculator."""
        self.confidence_levels = [0.90, 0.95, 0.99]  # For VaR calculations
        self.time_horizons = [1, 7, 30]  # Days for different calculations
        self.stress_scenarios = self._initialize_stress_scenarios()
        
        logger.info("Advanced risk metrics calculator initialized")

    def _initialize_stress_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Initialize predefined stress test scenarios."""
        return {
            "market_crash": {
                "description": "Major market crash scenario",
                "sol_shock": -0.50,  # 50% SOL price drop
                "correlation_increase": 0.8,  # Increased correlation during crisis
                "liquidity_shock": 0.3  # 30% liquidity reduction
            },
            "crypto_winter": {
                "description": "Extended bear market",
                "sol_shock": -0.30,  # 30% SOL price drop
                "correlation_increase": 0.6,
                "liquidity_shock": 0.2
            },
            "flash_crash": {
                "description": "Sudden market flash crash",
                "sol_shock": -0.25,  # 25% SOL price drop
                "correlation_increase": 0.9,  # Very high correlation
                "liquidity_shock": 0.5  # 50% liquidity reduction
            },
            "defi_crisis": {
                "description": "DeFi protocol crisis",
                "sol_shock": -0.15,  # 15% SOL price drop
                "correlation_increase": 0.7,
                "liquidity_shock": 0.4
            },
            "regulatory_shock": {
                "description": "Adverse regulatory news",
                "sol_shock": -0.20,  # 20% SOL price drop
                "correlation_increase": 0.5,
                "liquidity_shock": 0.1
            }
        }

    def calculate_var_cvar_multi_horizon(self, returns: np.array, 
                                       confidence_levels: Optional[List[float]] = None,
                                       time_horizons: Optional[List[int]] = None) -> List[VaRResult]:
        """
        Calculate Value at Risk and Conditional Value at Risk for multiple time horizons.

        Args:
            returns: Array of portfolio returns
            confidence_levels: List of confidence levels (default: [0.90, 0.95, 0.99])
            time_horizons: List of time horizons in days (default: [1, 7, 30])

        Returns:
            List of VaR results for each combination of confidence level and time horizon
        """
        if confidence_levels is None:
            confidence_levels = self.confidence_levels
        if time_horizons is None:
            time_horizons = self.time_horizons

        if len(returns) < 30:
            logger.warning("Insufficient data for reliable VaR calculation")
            return []

        results = []

        for confidence_level in confidence_levels:
            for time_horizon in time_horizons:
                try:
                    # Scale returns for time horizon (assuming daily returns)
                    if time_horizon > 1:
                        scaled_returns = returns * np.sqrt(time_horizon)
                    else:
                        scaled_returns = returns

                    # Calculate VaR using historical simulation
                    var_historical = self._calculate_historical_var(scaled_returns, confidence_level)
                    
                    # Calculate CVaR
                    cvar_historical = self._calculate_cvar(scaled_returns, var_historical)

                    # Calculate parametric VaR (assuming normal distribution)
                    var_parametric = self._calculate_parametric_var(scaled_returns, confidence_level)

                    # Use the more conservative estimate
                    var_final = min(var_historical, var_parametric)
                    
                    results.append(VaRResult(
                        var=var_final,
                        cvar=cvar_historical,
                        confidence_level=confidence_level,
                        time_horizon_days=time_horizon,
                        observations=len(returns),
                        method="historical_simulation"
                    ))

                except Exception as e:
                    logger.error(f"Error calculating VaR for {confidence_level:.0%} confidence, "
                               f"{time_horizon}d horizon: {e}")

        return results

    def _calculate_historical_var(self, returns: np.array, confidence_level: float) -> float:
        """Calculate VaR using historical simulation method."""
        sorted_returns = np.sort(returns)
        var_index = int((1 - confidence_level) * len(sorted_returns))
        return sorted_returns[var_index] if var_index < len(sorted_returns) else sorted_returns[0]

    def _calculate_parametric_var(self, returns: np.array, confidence_level: float) -> float:
        """Calculate VaR using parametric method (normal distribution assumption)."""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence_level)
        return mean_return + z_score * std_return

    def _calculate_cvar(self, returns: np.array, var_threshold: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        tail_returns = returns[returns <= var_threshold]
        return np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold

    def calculate_maximum_drawdown_prediction(self, returns: np.array,
                                            confidence_level: float = 0.95,
                                            time_horizon_days: int = 252,
                                            n_simulations: int = 10000) -> MaxDrawdownPrediction:
        """
        Predict potential maximum drawdown using Monte Carlo simulation.

        Args:
            returns: Historical returns array
            confidence_level: Confidence level for prediction
            time_horizon_days: Time horizon for prediction (default: 1 year)
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Maximum drawdown prediction result
        """
        if len(returns) < 30:
            logger.warning("Insufficient data for MDD prediction")
            return MaxDrawdownPrediction(0.0, confidence_level, 0.0, 0.0, 0, time_horizon_days)

        try:
            # Calculate return statistics
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            # Monte Carlo simulation
            max_drawdowns = []

            for _ in range(n_simulations):
                # Generate random returns using normal distribution
                simulated_returns = np.random.normal(mean_return, std_return, time_horizon_days)
                
                # Calculate cumulative returns
                cumulative_returns = np.cumprod(1 + simulated_returns)
                
                # Calculate drawdown series
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - peak) / peak
                
                # Record maximum drawdown for this simulation
                max_drawdowns.append(np.min(drawdown))

            max_drawdowns = np.array(max_drawdowns)

            # Calculate statistics
            predicted_mdd = np.percentile(max_drawdowns, (1 - confidence_level) * 100)
            mean_mdd = np.mean(max_drawdowns)
            std_mdd = np.std(max_drawdowns)

            return MaxDrawdownPrediction(
                predicted_mdd=abs(predicted_mdd),
                confidence_level=confidence_level,
                mean_mdd=abs(mean_mdd),
                std_mdd=std_mdd,
                simulations=n_simulations,
                time_horizon_days=time_horizon_days
            )

        except Exception as e:
            logger.error(f"Error in maximum drawdown prediction: {e}")
            return MaxDrawdownPrediction(0.0, confidence_level, 0.0, 0.0, 0, time_horizon_days)

    def stress_test_portfolio(self, positions: Dict[str, Any], 
                            scenarios: Optional[Dict[str, Dict[str, Any]]] = None) -> List[StressTestResult]:
        """
        Perform stress testing on portfolio under various scenarios.

        Args:
            positions: Current portfolio positions
            scenarios: Custom stress scenarios (optional)

        Returns:
            List of stress test results
        """
        if scenarios is None:
            scenarios = self.stress_scenarios

        stress_results = []

        try:
            total_portfolio_value = sum(pos.get('value_sol', 0) for pos in positions.values())
            
            if total_portfolio_value == 0:
                logger.warning("Portfolio has no value for stress testing")
                return stress_results

            for scenario_name, scenario in scenarios.items():
                scenario_pnl = 0.0
                position_pnls = {}
                worst_position = ""
                worst_position_pnl = 0.0

                for token_mint, position in positions.items():
                    position_value = position.get('value_sol', 0)
                    
                    # Apply scenario shock
                    if token_mint == "SOL" or position.get('is_sol', False):
                        # Direct SOL exposure
                        shock = scenario.get('sol_shock', 0.0)
                    else:
                        # Token exposure - assume correlation with SOL plus additional volatility
                        base_shock = scenario.get('sol_shock', 0.0)
                        correlation = scenario.get('correlation_increase', 0.5)
                        additional_volatility = np.random.normal(0, 0.1)  # Additional token-specific risk
                        shock = base_shock * correlation + additional_volatility

                    # Apply liquidity shock (reduces ability to exit at fair value)
                    liquidity_shock = scenario.get('liquidity_shock', 0.0)
                    effective_shock = shock * (1 + liquidity_shock)

                    position_pnl = position_value * effective_shock
                    scenario_pnl += position_pnl
                    position_pnls[token_mint] = position_pnl

                    # Track worst performing position
                    if position_pnl < worst_position_pnl:
                        worst_position_pnl = position_pnl
                        worst_position = position.get('token_name', token_mint)

                stress_results.append(StressTestResult(
                    scenario_name=scenario_name,
                    total_pnl=scenario_pnl,
                    pnl_percentage=(scenario_pnl / total_portfolio_value) * 100,
                    worst_position=worst_position,
                    worst_position_pnl=worst_position_pnl,
                    scenario_details=scenario
                ))

        except Exception as e:
            logger.error(f"Error in stress testing: {e}")

        return stress_results


# Create a singleton instance
advanced_risk_metrics = AdvancedRiskMetrics()
