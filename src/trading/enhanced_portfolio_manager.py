"""
Enhanced Portfolio Management for the Solana Memecoin Trading Bot.
Provides comprehensive portfolio tracking, risk management, and performance analytics.
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.trading.position_manager import position_manager
from src.trading.risk_manager import risk_manager
from src.trading.portfolio_analytics import portfolio_analytics
from src.wallet.wallet import wallet_manager

# Get logger for this module
logger = get_logger(__name__)


class PortfolioAllocationStrategy(Enum):
    """Portfolio allocation strategies."""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    CUSTOM = "custom"


@dataclass
class PortfolioTarget:
    """Represents a portfolio allocation target."""
    token_mint: str
    token_symbol: str
    target_percentage: float
    current_percentage: float
    deviation: float
    rebalance_needed: bool
    risk_level: str


@dataclass
class PortfolioConstraints:
    """Portfolio constraints and limits."""
    max_positions: int = 20
    max_position_size_percent: float = 10.0
    max_sector_allocation_percent: float = 30.0
    min_liquidity_sol: float = 1.0
    max_correlation: float = 0.8
    rebalance_threshold_percent: float = 5.0


class EnhancedPortfolioManager:
    """Enhanced portfolio management with advanced features."""

    def __init__(self):
        """Initialize the enhanced portfolio manager."""
        self.enabled = get_config_value("enhanced_portfolio_enabled", False)

        # Portfolio configuration
        self.allocation_strategy = PortfolioAllocationStrategy(
            get_config_value("portfolio_allocation_strategy", "equal_weight")
        )
        self.constraints = PortfolioConstraints(
            max_positions=int(get_config_value("portfolio_max_positions", "20")),
            max_position_size_percent=float(get_config_value("portfolio_max_position_size", "10.0")),
            max_sector_allocation_percent=float(get_config_value("portfolio_max_sector_allocation", "30.0")),
            min_liquidity_sol=float(get_config_value("portfolio_min_liquidity", "1.0")),
            max_correlation=float(get_config_value("portfolio_max_correlation", "0.8")),
            rebalance_threshold_percent=float(get_config_value("portfolio_rebalance_threshold", "5.0"))
        )

        # Portfolio targets and allocations
        self.target_allocations = get_config_value("portfolio_target_allocations", {})
        self.sector_allocations = get_config_value("portfolio_sector_allocations", {})

        # Rebalancing settings
        self.auto_rebalance_enabled = get_config_value("portfolio_auto_rebalance", False)
        self.rebalance_frequency_hours = int(get_config_value("portfolio_rebalance_frequency", "24"))
        self.last_rebalance_time = get_config_value("portfolio_last_rebalance", 0)

        # Performance tracking
        self.performance_history = get_config_value("portfolio_performance_history", [])
        self.benchmark_returns = get_config_value("portfolio_benchmark_returns", [])

        # Risk management integration
        self.risk_budget_allocation = get_config_value("portfolio_risk_budget", {
            "low_risk": 40.0,
            "medium_risk": 35.0,
            "high_risk": 20.0,
            "very_high_risk": 5.0
        })

        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = False

        logger.info("Enhanced portfolio manager initialized")

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable enhanced portfolio management.

        Args:
            enabled: Whether to enable enhanced portfolio management
        """
        self.enabled = enabled
        update_config("enhanced_portfolio_enabled", enabled)

        if enabled:
            self.start_monitoring()
        else:
            self.stop_monitoring()

        logger.info(f"Enhanced portfolio management {'enabled' if enabled else 'disabled'}")

    def set_allocation_strategy(self, strategy: PortfolioAllocationStrategy) -> None:
        """
        Set the portfolio allocation strategy.

        Args:
            strategy: The allocation strategy to use
        """
        self.allocation_strategy = strategy
        update_config("portfolio_allocation_strategy", strategy.value)
        logger.info(f"Portfolio allocation strategy set to: {strategy.value}")

    def add_target_allocation(self, token_mint: str, token_symbol: str,
                            target_percentage: float) -> None:
        """
        Add or update a target allocation for a token.

        Args:
            token_mint: Token mint address
            token_symbol: Token symbol
            target_percentage: Target allocation percentage (0-100)
        """
        if target_percentage < 0 or target_percentage > 100:
            raise ValueError("Target percentage must be between 0 and 100")

        self.target_allocations[token_mint] = {
            "symbol": token_symbol,
            "target_percentage": target_percentage,
            "added_at": datetime.now().isoformat()
        }

        update_config("portfolio_target_allocations", self.target_allocations)
        logger.info(f"Added target allocation: {token_symbol} = {target_percentage}%")

    def remove_target_allocation(self, token_mint: str) -> bool:
        """
        Remove a target allocation.

        Args:
            token_mint: Token mint address to remove

        Returns:
            True if removed, False if not found
        """
        if token_mint in self.target_allocations:
            symbol = self.target_allocations[token_mint]["symbol"]
            del self.target_allocations[token_mint]
            update_config("portfolio_target_allocations", self.target_allocations)
            logger.info(f"Removed target allocation for {symbol}")
            return True
        return False

    def calculate_current_allocations(self) -> Dict[str, PortfolioTarget]:
        """
        Calculate current portfolio allocations and compare to targets.

        Returns:
            Dictionary of portfolio targets with current vs target allocations
        """
        targets = {}

        # Get current portfolio value
        total_value = self.get_total_portfolio_value()
        if total_value <= 0:
            return targets

        # Get current positions
        positions = position_manager.get_all_positions()

        # Calculate current allocations
        current_allocations = {}
        for token_mint, position in positions.items():
            current_value = position.amount * position.current_price
            current_percentage = (current_value / total_value) * 100
            current_allocations[token_mint] = current_percentage

        # Compare with targets
        for token_mint, target_data in self.target_allocations.items():
            current_percentage = current_allocations.get(token_mint, 0.0)
            target_percentage = target_data["target_percentage"]
            deviation = abs(current_percentage - target_percentage)
            rebalance_needed = deviation > self.constraints.rebalance_threshold_percent

            # Get risk level
            risk_level = "unknown"
            if token_mint in positions:
                risk_assessment = risk_manager.assess_token_risk(token_mint)
                risk_level = risk_assessment.value if hasattr(risk_assessment, 'value') else str(risk_assessment)

            targets[token_mint] = PortfolioTarget(
                token_mint=token_mint,
                token_symbol=target_data["symbol"],
                target_percentage=target_percentage,
                current_percentage=current_percentage,
                deviation=deviation,
                rebalance_needed=rebalance_needed,
                risk_level=risk_level
            )

        return targets

    def get_total_portfolio_value(self) -> float:
        """
        Get the total portfolio value in SOL.

        Returns:
            Total portfolio value in SOL
        """
        total_value = 0.0

        # Add SOL balance
        try:
            sol_balance = wallet_manager.get_sol_balance()
            total_value += sol_balance
        except Exception as e:
            logger.warning(f"Error getting SOL balance: {e}")

        # Add position values
        positions = position_manager.get_all_positions()
        for position in positions.values():
            position_value = position.amount * position.current_price
            total_value += position_value

        return total_value

    def calculate_rebalancing_trades(self) -> List[Dict[str, Any]]:
        """
        Calculate the trades needed to rebalance the portfolio.

        Returns:
            List of trade recommendations
        """
        trades = []
        targets = self.calculate_current_allocations()
        total_value = self.get_total_portfolio_value()

        if total_value <= 0:
            return trades

        for target in targets.values():
            if not target.rebalance_needed:
                continue

            current_value = (target.current_percentage / 100) * total_value
            target_value = (target.target_percentage / 100) * total_value
            trade_value = target_value - current_value

            if abs(trade_value) < 0.01:  # Minimum trade size
                continue

            trade = {
                "token_mint": target.token_mint,
                "token_symbol": target.token_symbol,
                "action": "buy" if trade_value > 0 else "sell",
                "amount_sol": abs(trade_value),
                "current_percentage": target.current_percentage,
                "target_percentage": target.target_percentage,
                "deviation": target.deviation,
                "risk_level": target.risk_level
            }

            trades.append(trade)

        return trades

    def execute_rebalancing(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute portfolio rebalancing.

        Args:
            dry_run: If True, only simulate the rebalancing

        Returns:
            Rebalancing results
        """
        trades = self.calculate_rebalancing_trades()

        if not trades:
            return {
                "success": True,
                "message": "No rebalancing needed",
                "trades_executed": 0,
                "total_value_traded": 0.0
            }

        executed_trades = []
        total_value_traded = 0.0

        for trade in trades:
            try:
                if dry_run:
                    logger.info(f"DRY RUN: Would {trade['action']} {trade['amount_sol']} SOL of {trade['token_symbol']}")
                    executed_trades.append({**trade, "status": "simulated"})
                else:
                    # Execute the actual trade
                    if trade["action"] == "buy":
                        # Execute buy order
                        from src.trading.jupiter_api import jupiter_api
                        tx_signature = jupiter_api.execute_buy(
                            token_mint=trade["token_mint"],
                            amount_sol=trade["amount_sol"],
                            wallet=wallet_manager.current_keypair
                        )
                        executed_trades.append({**trade, "status": "executed", "tx_signature": tx_signature})
                    else:
                        # Execute sell order
                        position = position_manager.get_position(trade["token_mint"])
                        if position:
                            # Calculate amount to sell based on SOL value
                            amount_to_sell = trade["amount_sol"] / position.current_price
                            tx_signature = position_manager.execute_partial_sell(
                                trade["token_mint"], amount_to_sell
                            )
                            executed_trades.append({**trade, "status": "executed", "tx_signature": tx_signature})
                        else:
                            executed_trades.append({**trade, "status": "failed", "error": "Position not found"})

                total_value_traded += trade["amount_sol"]

            except Exception as e:
                logger.error(f"Error executing rebalancing trade for {trade['token_symbol']}: {e}")
                executed_trades.append({**trade, "status": "failed", "error": str(e)})

        # Update last rebalance time
        if not dry_run:
            self.last_rebalance_time = time.time()
            update_config("portfolio_last_rebalance", self.last_rebalance_time)

        return {
            "success": True,
            "trades_executed": len([t for t in executed_trades if t["status"] == "executed"]),
            "trades_failed": len([t for t in executed_trades if t["status"] == "failed"]),
            "total_value_traded": total_value_traded,
            "executed_trades": executed_trades,
            "dry_run": dry_run
        }

    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio metrics.

        Returns:
            Portfolio metrics and statistics
        """
        metrics = {}

        # Basic portfolio info
        total_value = self.get_total_portfolio_value()
        positions = position_manager.get_all_positions()

        metrics["total_value_sol"] = total_value
        metrics["position_count"] = len(positions)
        metrics["timestamp"] = datetime.now().isoformat()

        # Allocation metrics
        allocations = self.calculate_current_allocations()
        metrics["allocations"] = {mint: asdict(target) for mint, target in allocations.items()}

        # Risk metrics
        risk_distribution = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
        for target in allocations.values():
            risk_level = target.risk_level.lower()
            if risk_level in risk_distribution:
                risk_distribution[risk_level] += target.current_percentage

        metrics["risk_distribution"] = risk_distribution

        # Concentration metrics
        if allocations:
            largest_position = max(allocations.values(), key=lambda x: x.current_percentage)
            metrics["largest_position_percent"] = largest_position.current_percentage
            metrics["concentration_risk"] = sum(t.current_percentage**2 for t in allocations.values()) / 100
        else:
            metrics["largest_position_percent"] = 0
            metrics["concentration_risk"] = 0

        # Rebalancing metrics
        rebalancing_needed = any(t.rebalance_needed for t in allocations.values())
        avg_deviation = sum(t.deviation for t in allocations.values()) / len(allocations) if allocations else 0

        metrics["rebalancing_needed"] = rebalancing_needed
        metrics["average_deviation"] = avg_deviation
        metrics["time_since_last_rebalance"] = time.time() - self.last_rebalance_time

        return metrics

    def start_monitoring(self) -> None:
        """Start the portfolio monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Portfolio monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the portfolio monitoring thread."""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Portfolio monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring:
            try:
                if self.enabled:
                    # Check if rebalancing is needed
                    if self.auto_rebalance_enabled:
                        time_since_rebalance = time.time() - self.last_rebalance_time
                        if time_since_rebalance > (self.rebalance_frequency_hours * 3600):
                            logger.info("Auto-rebalancing triggered by time interval")
                            result = self.execute_rebalancing(dry_run=False)
                            logger.info(f"Auto-rebalancing result: {result}")

                    # Update performance history
                    metrics = self.calculate_portfolio_metrics()
                    self.performance_history.append(metrics)

                    # Keep only last 1000 entries
                    if len(self.performance_history) > 1000:
                        self.performance_history = self.performance_history[-1000:]

                    update_config("portfolio_performance_history", self.performance_history)

                # Sleep for monitoring interval
                time.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Error in portfolio monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


# Create global instance
enhanced_portfolio_manager = EnhancedPortfolioManager()