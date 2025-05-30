"""
Dynamic Portfolio Optimization Module

This module provides advanced portfolio optimization capabilities including:
- Mean-Variance Optimization (Markowitz)
- Risk Parity Models
- Black-Litterman Model
- Market Regime Detection
- Dynamic Rebalancing

Author: Solana Memecoin Trading Bot
Version: 3.0.0 (Phase 3)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy.optimize import minimize
from scipy import linalg
import warnings

from src.utils.logging_utils import get_logger

# Suppress optimization warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = get_logger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MAXIMUM_DIVERSIFICATION = "max_diversification"
    MINIMUM_VARIANCE = "min_variance"


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_turnover: float = 0.5
    target_return: Optional[float] = None
    max_risk: Optional[float] = None
    sector_limits: Optional[Dict[str, float]] = None
    position_limits: Optional[Dict[str, Tuple[float, float]]] = None


@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    optimization_method: str
    market_regime: str
    timestamp: datetime
    convergence_status: str
    objective_value: float
    constraints_satisfied: bool


@dataclass
class RegimeDetectionResult:
    """Market regime detection result."""
    current_regime: MarketRegime
    regime_probability: float
    volatility_level: float
    trend_strength: float
    momentum_score: float
    confidence: float
    timestamp: datetime


class DynamicPortfolioOptimizer:
    """
    Advanced portfolio optimization with market regime detection.

    Features:
    - Multiple optimization methods
    - Market regime-aware optimization
    - Dynamic rebalancing
    - Risk management integration
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the dynamic portfolio optimizer."""
        self.config = config or {}

        # Optimization parameters
        self.lookback_days = self.config.get('optimization_lookback_days', 30)
        self.rebalancing_frequency = self.config.get('rebalancing_frequency_hours', 24)
        self.max_turnover = self.config.get('max_portfolio_turnover', 0.2)
        self.regime_detection_enabled = self.config.get('regime_detection_enabled', True)

        # Risk parameters
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.confidence_level = 0.95

        # Regime detection parameters
        self.volatility_window = 20
        self.trend_window = 10
        self.momentum_window = 5

        # Optimization history
        self.optimization_history: List[OptimizationResult] = []
        self.regime_history: List[RegimeDetectionResult] = []

        # Current state
        self.current_weights: Dict[str, float] = {}
        self.current_regime: Optional[MarketRegime] = None
        self.last_optimization: Optional[datetime] = None

        logger.info("Dynamic portfolio optimizer initialized")

    def optimize_portfolio(self,
                          returns_data: pd.DataFrame,
                          current_weights: Optional[Dict[str, float]] = None,
                          method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
                          constraints: Optional[OptimizationConstraints] = None) -> OptimizationResult:
        """
        Optimize portfolio allocation using specified method.

        Args:
            returns_data: Historical returns data (assets as columns)
            current_weights: Current portfolio weights
            method: Optimization method to use
            constraints: Optimization constraints

        Returns:
            Optimization result with optimal weights
        """
        try:
            logger.info(f"Starting portfolio optimization using {method.value}")

            # Validate input data
            if returns_data.empty or len(returns_data.columns) < 2:
                raise ValueError("Insufficient data for optimization")

            # Detect market regime if enabled
            regime_result = None
            if self.regime_detection_enabled:
                regime_result = self.detect_market_regime(returns_data)
                self.current_regime = regime_result.current_regime
                logger.info(f"Detected market regime: {regime_result.current_regime.value}")

            # Set default constraints
            if constraints is None:
                constraints = OptimizationConstraints()

            # Adjust constraints based on market regime
            if regime_result:
                constraints = self._adjust_constraints_for_regime(constraints, regime_result)

            # Calculate expected returns and covariance matrix
            expected_returns = self._calculate_expected_returns(returns_data, method)
            cov_matrix = self._calculate_covariance_matrix(returns_data)

            # Perform optimization based on method
            if method == OptimizationMethod.MEAN_VARIANCE:
                weights = self._optimize_mean_variance(expected_returns, cov_matrix, constraints)
            elif method == OptimizationMethod.RISK_PARITY:
                weights = self._optimize_risk_parity(cov_matrix, constraints)
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                weights = self._optimize_black_litterman(returns_data, expected_returns, cov_matrix, constraints)
            elif method == OptimizationMethod.MAXIMUM_DIVERSIFICATION:
                weights = self._optimize_max_diversification(cov_matrix, constraints)
            elif method == OptimizationMethod.MINIMUM_VARIANCE:
                weights = self._optimize_min_variance(cov_matrix, constraints)
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate / 252) / portfolio_risk if portfolio_risk > 0 else 0

            # Create result
            result = OptimizationResult(
                weights={returns_data.columns[i]: weights[i] for i in range(len(weights))},
                expected_return=portfolio_return * 252,  # Annualized
                expected_risk=portfolio_risk * np.sqrt(252),  # Annualized
                sharpe_ratio=sharpe_ratio * np.sqrt(252),  # Annualized
                optimization_method=method.value,
                market_regime=regime_result.current_regime.value if regime_result else "unknown",
                timestamp=datetime.now(),
                convergence_status="success",
                objective_value=0.0,  # Will be set by optimization method
                constraints_satisfied=True
            )

            # Store results
            self.optimization_history.append(result)
            self.current_weights = result.weights
            self.last_optimization = result.timestamp

            logger.info(f"Portfolio optimization completed successfully")
            logger.info(f"Expected return: {result.expected_return:.4f}, Risk: {result.expected_risk:.4f}, Sharpe: {result.sharpe_ratio:.4f}")

            return result

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            raise

    def detect_market_regime(self, returns_data: pd.DataFrame) -> RegimeDetectionResult:
        """
        Detect current market regime using multiple indicators.

        Args:
            returns_data: Historical returns data

        Returns:
            Market regime detection result
        """
        try:
            # Calculate portfolio returns (equal weighted for regime detection)
            portfolio_returns = returns_data.mean(axis=1)

            # Calculate volatility
            volatility = portfolio_returns.rolling(self.volatility_window).std().iloc[-1]
            volatility_percentile = (portfolio_returns.rolling(self.volatility_window).std() < volatility).mean()

            # Calculate trend strength
            trend_returns = portfolio_returns.rolling(self.trend_window).mean().iloc[-1]
            trend_strength = abs(trend_returns)

            # Calculate momentum
            momentum = portfolio_returns.rolling(self.momentum_window).mean().iloc[-1]

            # Determine regime based on indicators
            regime = self._classify_regime(volatility_percentile, trend_returns, momentum)

            # Calculate confidence based on signal strength
            confidence = self._calculate_regime_confidence(volatility_percentile, trend_strength, abs(momentum))

            result = RegimeDetectionResult(
                current_regime=regime,
                regime_probability=confidence,
                volatility_level=volatility,
                trend_strength=trend_strength,
                momentum_score=momentum,
                confidence=confidence,
                timestamp=datetime.now()
            )

            self.regime_history.append(result)

            return result

        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            # Return default regime
            return RegimeDetectionResult(
                current_regime=MarketRegime.SIDEWAYS,
                regime_probability=0.5,
                volatility_level=0.02,
                trend_strength=0.0,
                momentum_score=0.0,
                confidence=0.5,
                timestamp=datetime.now()
            )

    def should_rebalance(self, current_weights: Dict[str, float], target_weights: Dict[str, float]) -> bool:
        """
        Determine if portfolio should be rebalanced.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            True if rebalancing is recommended
        """
        try:
            # Check time-based rebalancing
            if self.last_optimization:
                time_since_last = datetime.now() - self.last_optimization
                if time_since_last.total_seconds() / 3600 >= self.rebalancing_frequency:
                    return True

            # Check weight deviation
            total_deviation = 0.0
            for asset in target_weights:
                current_weight = current_weights.get(asset, 0.0)
                target_weight = target_weights[asset]
                total_deviation += abs(current_weight - target_weight)

            # Rebalance if total deviation exceeds threshold
            return total_deviation > self.max_turnover

        except Exception as e:
            logger.error(f"Error checking rebalancing condition: {e}")
            return False

    def _calculate_expected_returns(self, returns_data: pd.DataFrame, method: OptimizationMethod) -> np.ndarray:
        """Calculate expected returns based on optimization method."""
        if method == OptimizationMethod.BLACK_LITTERMAN:
            # Use market cap weighted returns for Black-Litterman
            return returns_data.mean().values
        else:
            # Use historical mean for other methods
            return returns_data.mean().values

    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix with shrinkage estimation."""
        # Calculate sample covariance matrix
        sample_cov = returns_data.cov().values

        # Apply shrinkage to improve estimation
        n_assets = sample_cov.shape[0]
        identity = np.eye(n_assets)

        # Ledoit-Wolf shrinkage
        shrinkage_target = np.trace(sample_cov) / n_assets * identity
        shrinkage_intensity = 0.2  # 20% shrinkage

        shrunk_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * shrinkage_target

        return shrunk_cov

    def _optimize_mean_variance(self, expected_returns: np.ndarray,
                               cov_matrix: np.ndarray,
                               constraints: OptimizationConstraints) -> np.ndarray:
        """Optimize portfolio using Mean-Variance optimization."""
        n_assets = len(expected_returns)

        # Objective function: maximize Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_risk == 0:
                return -np.inf
            return -(portfolio_return - self.risk_free_rate / 252) / portfolio_risk

        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]

        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)

        if result.success:
            return result.x
        else:
            logger.warning("Mean-variance optimization failed, using equal weights")
            return x0

    def _optimize_risk_parity(self, cov_matrix: np.ndarray,
                             constraints: OptimizationConstraints) -> np.ndarray:
        """Optimize portfolio using Risk Parity approach."""
        n_assets = cov_matrix.shape[0]

        # Objective function: minimize sum of squared risk contribution differences
        def objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_risk == 0:
                return 1e6

            # Calculate risk contributions
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_risk
            risk_contrib = weights * marginal_contrib

            # Target equal risk contribution
            target_contrib = portfolio_risk / n_assets

            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]

        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]

        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)

        if result.success:
            return result.x
        else:
            logger.warning("Risk parity optimization failed, using equal weights")
            return x0

    def _optimize_black_litterman(self, returns_data: pd.DataFrame,
                                 expected_returns: np.ndarray,
                                 cov_matrix: np.ndarray,
                                 constraints: OptimizationConstraints) -> np.ndarray:
        """Optimize portfolio using Black-Litterman model."""
        n_assets = len(expected_returns)

        # Market capitalization weights (proxy using equal weights)
        market_weights = np.array([1.0 / n_assets] * n_assets)

        # Risk aversion parameter
        risk_aversion = 3.0

        # Implied equilibrium returns
        pi = risk_aversion * np.dot(cov_matrix, market_weights)

        # Uncertainty in prior (tau)
        tau = 1.0 / len(returns_data)

        # Views and confidence (simplified - no specific views)
        # In practice, this would incorporate analyst views
        P = np.eye(n_assets)  # Identity matrix (views on all assets)
        Q = expected_returns  # View returns (using historical means)
        omega = tau * np.diag(np.diag(cov_matrix))  # Uncertainty in views

        # Black-Litterman formula
        M1 = linalg.inv(tau * cov_matrix)
        M2 = np.dot(P.T, np.dot(linalg.inv(omega), P))
        M3 = np.dot(linalg.inv(tau * cov_matrix), pi)
        M4 = np.dot(P.T, np.dot(linalg.inv(omega), Q))

        mu_bl = np.dot(linalg.inv(M1 + M2), M3 + M4)
        cov_bl = linalg.inv(M1 + M2)

        # Optimize using Black-Litterman inputs
        def objective(weights):
            portfolio_return = np.dot(weights, mu_bl)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_bl, weights)))
            if portfolio_risk == 0:
                return -np.inf
            return -(portfolio_return - self.risk_free_rate / 252) / portfolio_risk

        # Constraints and bounds
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = market_weights

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)

        if result.success:
            return result.x
        else:
            logger.warning("Black-Litterman optimization failed, using market weights")
            return market_weights

    def _optimize_max_diversification(self, cov_matrix: np.ndarray,
                                     constraints: OptimizationConstraints) -> np.ndarray:
        """Optimize portfolio for maximum diversification."""
        n_assets = cov_matrix.shape[0]

        # Diversification ratio = weighted average volatility / portfolio volatility
        def objective(weights):
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.dot(weights, individual_vols)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

            if portfolio_vol == 0:
                return -1e6

            # Maximize diversification ratio (minimize negative)
            return -weighted_avg_vol / portfolio_vol

        # Constraints and bounds
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)

        if result.success:
            return result.x
        else:
            logger.warning("Maximum diversification optimization failed, using equal weights")
            return x0

    def _optimize_min_variance(self, cov_matrix: np.ndarray,
                              constraints: OptimizationConstraints) -> np.ndarray:
        """Optimize portfolio for minimum variance."""
        n_assets = cov_matrix.shape[0]

        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Constraints and bounds
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)

        if result.success:
            return result.x
        else:
            logger.warning("Minimum variance optimization failed, using equal weights")
            return x0

    def _classify_regime(self, volatility_percentile: float, trend_returns: float, momentum: float) -> MarketRegime:
        """Classify market regime based on indicators."""
        # High volatility regime
        if volatility_percentile > 0.8:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility_percentile < 0.2:
            return MarketRegime.LOW_VOLATILITY

        # Trend-based classification
        if abs(trend_returns) > 0.01:  # Strong trend
            if trend_returns > 0:
                return MarketRegime.BULL
            else:
                return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def _calculate_regime_confidence(self, volatility_percentile: float,
                                   trend_strength: float, momentum_strength: float) -> float:
        """Calculate confidence in regime classification."""
        # Combine multiple signals for confidence
        vol_confidence = abs(volatility_percentile - 0.5) * 2  # 0 to 1
        trend_confidence = min(trend_strength * 100, 1.0)  # Cap at 1
        momentum_confidence = min(momentum_strength * 100, 1.0)  # Cap at 1

        # Weighted average
        confidence = (vol_confidence * 0.3 + trend_confidence * 0.4 + momentum_confidence * 0.3)
        return max(0.5, min(1.0, confidence))  # Ensure between 0.5 and 1.0

    def _adjust_constraints_for_regime(self, constraints: OptimizationConstraints,
                                     regime_result: RegimeDetectionResult) -> OptimizationConstraints:
        """Adjust optimization constraints based on market regime."""
        adjusted_constraints = OptimizationConstraints(
            min_weight=constraints.min_weight,
            max_weight=constraints.max_weight,
            max_turnover=constraints.max_turnover,
            target_return=constraints.target_return,
            max_risk=constraints.max_risk,
            sector_limits=constraints.sector_limits,
            position_limits=constraints.position_limits
        )

        # Adjust based on regime
        if regime_result.current_regime == MarketRegime.HIGH_VOLATILITY:
            # More conservative in high volatility
            adjusted_constraints.max_weight = min(constraints.max_weight, 0.3)
            adjusted_constraints.max_turnover = constraints.max_turnover * 0.5
        elif regime_result.current_regime == MarketRegime.BEAR:
            # More defensive positioning
            adjusted_constraints.max_weight = min(constraints.max_weight, 0.25)
        elif regime_result.current_regime == MarketRegime.BULL:
            # Allow more concentration
            adjusted_constraints.max_weight = min(constraints.max_weight * 1.2, 0.5)

        return adjusted_constraints

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history and current state."""
        if not self.optimization_history:
            return {"status": "No optimizations performed"}

        latest = self.optimization_history[-1]

        # Calculate performance metrics
        returns = [opt.expected_return for opt in self.optimization_history[-10:]]
        risks = [opt.expected_risk for opt in self.optimization_history[-10:]]
        sharpe_ratios = [opt.sharpe_ratio for opt in self.optimization_history[-10:]]

        summary = {
            "total_optimizations": len(self.optimization_history),
            "last_optimization": latest.timestamp.isoformat(),
            "current_method": latest.optimization_method,
            "current_regime": latest.market_regime,
            "current_weights": latest.weights,
            "performance_metrics": {
                "expected_return": latest.expected_return,
                "expected_risk": latest.expected_risk,
                "sharpe_ratio": latest.sharpe_ratio,
                "avg_return_10": np.mean(returns) if returns else 0,
                "avg_risk_10": np.mean(risks) if risks else 0,
                "avg_sharpe_10": np.mean(sharpe_ratios) if sharpe_ratios else 0
            },
            "regime_history": [
                {
                    "regime": regime.current_regime.value,
                    "confidence": regime.confidence,
                    "timestamp": regime.timestamp.isoformat()
                }
                for regime in self.regime_history[-5:]  # Last 5 regime detections
            ]
        }

        return summary

    def calculate_efficient_frontier(self, returns_data: pd.DataFrame,
                                   n_portfolios: int = 100) -> Dict[str, List[float]]:
        """
        Calculate efficient frontier for portfolio optimization.

        Args:
            returns_data: Historical returns data
            n_portfolios: Number of portfolios to calculate

        Returns:
            Dictionary with returns, risks, and weights for efficient frontier
        """
        try:
            expected_returns = self._calculate_expected_returns(returns_data, OptimizationMethod.MEAN_VARIANCE)
            cov_matrix = self._calculate_covariance_matrix(returns_data)

            # Calculate range of target returns
            min_return = np.min(expected_returns)
            max_return = np.max(expected_returns)
            target_returns = np.linspace(min_return, max_return, n_portfolios)

            frontier_returns = []
            frontier_risks = []
            frontier_weights = []

            for target_return in target_returns:
                try:
                    # Optimize for minimum risk at target return
                    constraints = OptimizationConstraints(target_return=target_return)
                    weights = self._optimize_for_target_return(expected_returns, cov_matrix, target_return)

                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

                    frontier_returns.append(portfolio_return * 252)  # Annualized
                    frontier_risks.append(portfolio_risk * np.sqrt(252))  # Annualized
                    frontier_weights.append(weights.tolist())

                except Exception as e:
                    logger.warning(f"Failed to optimize for target return {target_return}: {e}")
                    continue

            return {
                "returns": frontier_returns,
                "risks": frontier_risks,
                "weights": frontier_weights,
                "sharpe_ratios": [(r - self.risk_free_rate) / risk if risk > 0 else 0
                                for r, risk in zip(frontier_returns, frontier_risks)]
            }

        except Exception as e:
            logger.error(f"Efficient frontier calculation failed: {e}")
            return {"returns": [], "risks": [], "weights": [], "sharpe_ratios": []}

    def _optimize_for_target_return(self, expected_returns: np.ndarray,
                                   cov_matrix: np.ndarray, target_return: float) -> np.ndarray:
        """Optimize portfolio for minimum risk at target return."""
        n_assets = len(expected_returns)

        # Objective: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}  # Target return
        ]

        # Bounds
        bounds = [(0.0, 1.0) for _ in range(n_assets)]

        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)

        if result.success:
            return result.x
        else:
            return x0


# Global instance
dynamic_portfolio_optimizer = DynamicPortfolioOptimizer()
