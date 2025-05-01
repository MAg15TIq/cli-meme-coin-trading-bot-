"""
Gas optimization module for the Solana Memecoin Trading Bot.
Provides advanced fee estimation and optimization for Solana transactions.
"""

import json
import logging
import time
import statistics
import threading
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import numpy as np

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.solana.solana_interact import solana_client

# Get logger for this module
logger = get_logger(__name__)


class TransactionPriority(Enum):
    """Transaction priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TransactionType(Enum):
    """Transaction types for fee optimization."""
    DEFAULT = "default"
    BUY = "buy"
    SELL = "sell"
    SNIPE = "snipe"
    SWAP = "swap"
    LIMIT_ORDER = "limit_order"
    WITHDRAW = "withdraw"


class GasOptimizer:
    """Manager for gas fee optimization."""

    def __init__(self):
        """Initialize the gas optimizer."""
        self.enabled = get_config_value("fee_optimization_enabled", True)

        # Fee history
        self.fee_history = []
        self.fee_history_max_entries = int(get_config_value("fee_history_max_entries", "1000"))
        self.fee_history_file = Path(get_config_value("data_dir", str(Path.home() / ".solana-trading-bot"))) / "fee_history.json"

        # Fee percentiles
        self.fee_percentiles = {
            TransactionPriority.LOW.value: int(get_config_value("low_priority_percentile", "25")),
            TransactionPriority.MEDIUM.value: int(get_config_value("medium_priority_percentile", "50")),
            TransactionPriority.HIGH.value: int(get_config_value("high_priority_percentile", "75")),
            TransactionPriority.URGENT.value: int(get_config_value("urgent_priority_percentile", "90"))
        }

        # Transaction type multipliers
        self.tx_type_multipliers = {
            TransactionType.DEFAULT.value: float(get_config_value("default_fee_multiplier", "1.0")),
            TransactionType.BUY.value: float(get_config_value("buy_fee_multiplier", "1.0")),
            TransactionType.SELL.value: float(get_config_value("sell_fee_multiplier", "1.0")),
            TransactionType.SNIPE.value: float(get_config_value("snipe_fee_multiplier", "2.0")),
            TransactionType.SWAP.value: float(get_config_value("swap_fee_multiplier", "1.2")),
            TransactionType.LIMIT_ORDER.value: float(get_config_value("limit_order_fee_multiplier", "0.8")),
            TransactionType.WITHDRAW.value: float(get_config_value("withdraw_fee_multiplier", "1.5"))
        }

        # Minimum fee
        self.min_fee = int(get_config_value("min_priority_fee", "1000"))

        # Network congestion tracking
        self.congestion_history = []
        self.congestion_history_max_entries = 100
        self.last_congestion_check = 0
        self.congestion_check_interval = int(get_config_value("congestion_check_interval", "60"))  # seconds

        # Load fee history
        self._load_fee_history()

        # Start background thread for fee data collection
        if self.enabled:
            self._start_fee_collection_thread()

        logger.info(f"Initialized gas optimizer (enabled: {self.enabled})")

    def _load_fee_history(self) -> None:
        """Load fee history from file."""
        try:
            if self.fee_history_file.exists():
                with open(self.fee_history_file, 'r') as f:
                    data = json.load(f)
                    self.fee_history = data.get("fee_history", [])
                    logger.info(f"Loaded {len(self.fee_history)} fee history entries")
        except Exception as e:
            logger.error(f"Error loading fee history: {e}")
            self.fee_history = []

    def _save_fee_history(self) -> None:
        """Save fee history to file."""
        try:
            # Create directory if it doesn't exist
            self.fee_history_file.parent.mkdir(parents=True, exist_ok=True)

            # Trim history if needed
            if len(self.fee_history) > self.fee_history_max_entries:
                self.fee_history = self.fee_history[-self.fee_history_max_entries:]

            # Save to file
            with open(self.fee_history_file, 'w') as f:
                json.dump({"fee_history": self.fee_history}, f)
        except Exception as e:
            logger.error(f"Error saving fee history: {e}")

    def _start_fee_collection_thread(self) -> None:
        """Start background thread for fee data collection."""
        def collect_fees():
            while True:
                try:
                    # Collect recent fees
                    self.collect_recent_fees()

                    # Sleep for 5 minutes
                    time.sleep(300)
                except Exception as e:
                    logger.error(f"Error in fee collection thread: {e}")
                    time.sleep(60)  # Sleep for 1 minute on error

        # Start thread
        thread = threading.Thread(target=collect_fees, daemon=True)
        thread.start()
        logger.info("Started fee collection thread")

    def collect_recent_fees(self) -> Dict[str, Any]:
        """
        Collect recent priority fees from the network.

        Returns:
            Dictionary with fee data
        """
        try:
            # Get recent priority fees
            fees_by_percentile = solana_client.get_recent_priority_fee()

            if not fees_by_percentile:
                logger.warning("No recent priority fees found")
                return {}

            # Get network congestion
            congestion = self._get_network_congestion()

            # Record fee data
            timestamp = datetime.now().timestamp()
            fee_data = {
                "timestamp": timestamp,
                "fees": fees_by_percentile,
                "congestion": congestion
            }

            # Add to history
            self.fee_history.append(fee_data)

            # Save periodically
            if len(self.fee_history) % 10 == 0:
                self._save_fee_history()

            return fee_data
        except Exception as e:
            logger.error(f"Error collecting recent fees: {e}")
            return {}

    def get_priority_fee(self, priority: Union[str, TransactionPriority] = None,
                        tx_type: Union[str, TransactionType] = None) -> int:
        """
        Get the optimal priority fee for a transaction.

        Args:
            priority: The priority level (low, medium, high, urgent)
            tx_type: The transaction type

        Returns:
            The priority fee in micro-lamports
        """
        if not self.enabled:
            logger.warning("Fee optimization is disabled")
            return self.min_fee

        try:
            # Set defaults
            if priority is None:
                priority = TransactionPriority.MEDIUM.value
            elif isinstance(priority, TransactionPriority):
                priority = priority.value

            if tx_type is None:
                tx_type = TransactionType.DEFAULT.value
            elif isinstance(tx_type, TransactionType):
                tx_type = tx_type.value

            # Get percentile for this priority
            percentile = self.fee_percentiles.get(priority, 50)

            # Get multiplier for this transaction type
            multiplier = self.tx_type_multipliers.get(tx_type, 1.0)

            # Get recent fees
            fees_by_percentile = self._get_recent_fees()

            # If no fees found, use minimum fee
            if not fees_by_percentile:
                logger.warning("No recent fees found, using minimum fee")
                return max(int(self.min_fee * multiplier), self.min_fee)

            # Get base fee at the specified percentile
            base_fee = fees_by_percentile.get(str(percentile), self.min_fee)

            # Get network congestion multiplier
            congestion_multiplier = self._get_congestion_multiplier()

            # Apply multipliers and ensure minimum fee
            final_fee = max(int(base_fee * multiplier * congestion_multiplier), self.min_fee)

            # Apply time-based adjustment if enabled
            if get_config_value("time_based_fee_adjustment", False):
                time_multiplier = self._get_time_based_multiplier()
                if time_multiplier != 1.0:
                    final_fee = int(final_fee * time_multiplier)
                    logger.info(f"Applied time-based fee multiplier: {time_multiplier}x")

            # Apply transaction-specific adjustments
            if tx_type == TransactionType.SNIPE.value:
                # For sniping, we want to be extra competitive
                final_fee = max(final_fee, int(fees_by_percentile.get("90", final_fee) * 1.1))
                logger.info(f"Adjusted snipe fee to be more competitive: {final_fee}")
            elif tx_type == TransactionType.SELL.value and get_config_value("urgent_sell_fee_boost", False):
                # For urgent sells, boost the fee
                final_fee = int(final_fee * 1.3)
                logger.info(f"Applied urgent sell fee boost: {final_fee}")

            logger.info(f"Priority fee for {tx_type} ({priority}): {final_fee} micro-lamports " +
                       f"(base: {base_fee}, multiplier: {multiplier}, congestion: {congestion_multiplier:.2f})")

            return final_fee
        except Exception as e:
            logger.error(f"Error calculating priority fee: {e}")
            return self.min_fee

    def _get_recent_fees(self) -> Dict[str, int]:
        """
        Get recent fees from history or fetch new ones.

        Returns:
            Dictionary of fees by percentile
        """
        try:
            # Check if we have recent fee data (last 10 minutes)
            current_time = datetime.now().timestamp()
            recent_entries = [entry for entry in self.fee_history
                             if current_time - entry["timestamp"] < 600]

            if recent_entries:
                # Use the most recent entry
                return recent_entries[-1]["fees"]

            # If no recent data, fetch new data
            fee_data = self.collect_recent_fees()
            if fee_data and "fees" in fee_data:
                return fee_data["fees"]

            # If still no data, use solana_client directly
            return solana_client.get_recent_priority_fee()
        except Exception as e:
            logger.error(f"Error getting recent fees: {e}")
            return {}

    def _get_network_congestion(self) -> float:
        """
        Get the current network congestion level.

        Returns:
            Congestion level from 0.0 (low) to 1.0 (high)
        """
        current_time = time.time()

        # Only check congestion at most once per minute
        if current_time - self.last_congestion_check < self.congestion_check_interval:
            # Return the most recent congestion value if available
            if self.congestion_history:
                return self.congestion_history[-1]["value"]
            return 0.5  # Default to medium congestion

        self.last_congestion_check = current_time

        try:
            # Get recent performance samples
            response = solana_client.rpc_manager.execute_with_failover("getRecentPerformanceSamples", 5)

            if "result" not in response or not response["result"]:
                return 0.5  # Default to medium congestion if no data

            # Get the most recent samples
            samples = response["result"]

            if not samples:
                return 0.5

            # Calculate average TPS and max TPS
            avg_tps = sum(sample["numTransactions"] / sample["samplePeriodSecs"] for sample in samples) / len(samples)

            # Solana's theoretical max TPS is around 50,000, but practical is lower
            # We'll use 20,000 as a reference point
            max_practical_tps = 20000

            # Calculate congestion as inverse of TPS ratio (lower TPS = higher congestion)
            congestion = 1.0 - min(avg_tps / max_practical_tps, 1.0)

            # Add to history
            self.congestion_history.append({
                "timestamp": current_time,
                "value": congestion
            })

            # Trim history if needed
            if len(self.congestion_history) > self.congestion_history_max_entries:
                self.congestion_history = self.congestion_history[-self.congestion_history_max_entries:]

            return congestion
        except Exception as e:
            logger.warning(f"Error calculating network congestion: {e}")
            return 0.5  # Default to medium congestion on error

    def _get_congestion_multiplier(self) -> float:
        """
        Get a fee multiplier based on network congestion.

        Returns:
            Fee multiplier based on congestion
        """
        try:
            congestion = self._get_network_congestion()

            if congestion > 0.8:  # High congestion
                return 1.5
            elif congestion > 0.5:  # Medium congestion
                return 1.2
            elif congestion > 0.3:  # Low-medium congestion
                return 1.1
            else:  # Low congestion
                return 1.0
        except Exception as e:
            logger.warning(f"Error calculating congestion multiplier: {e}")
            return 1.0  # No adjustment on error

    def _get_time_based_multiplier(self) -> float:
        """
        Get a fee multiplier based on time of day.

        Returns:
            Fee multiplier based on time of day
        """
        try:
            # Get current hour (UTC)
            current_hour = datetime.now(datetime.timezone.utc).hour

            # Define peak hours (typically US and Asian market hours)
            us_peak_hours = range(13, 21)  # 13:00-21:00 UTC (9am-5pm EST)
            asia_peak_hours = range(0, 8)  # 00:00-08:00 UTC (8am-4pm Asia)

            if current_hour in us_peak_hours:
                return 1.2  # 20% increase during US peak
            elif current_hour in asia_peak_hours:
                return 1.1  # 10% increase during Asia peak
            else:
                return 1.0  # No adjustment during off-peak
        except Exception as e:
            logger.warning(f"Error calculating time-based fee multiplier: {e}")
            return 1.0  # No adjustment on error

    def get_compute_limit(self, tx_type: Union[str, TransactionType] = None) -> int:
        """
        Get the optimal compute unit limit for a transaction.

        Args:
            tx_type: The transaction type

        Returns:
            The compute unit limit
        """
        try:
            # Default compute limit
            default_limit = int(get_config_value("compute_unit_limit", "200000"))

            if not self.enabled or tx_type is None:
                return default_limit

            # Convert enum to string if needed
            if isinstance(tx_type, TransactionType):
                tx_type = tx_type.value

            # Transaction-specific compute limits
            compute_limits = {
                TransactionType.DEFAULT.value: default_limit,
                TransactionType.BUY.value: int(get_config_value("buy_compute_limit", str(default_limit))),
                TransactionType.SELL.value: int(get_config_value("sell_compute_limit", str(default_limit))),
                TransactionType.SNIPE.value: int(get_config_value("snipe_compute_limit", str(default_limit * 1.2))),
                TransactionType.SWAP.value: int(get_config_value("swap_compute_limit", str(default_limit * 1.1))),
                TransactionType.LIMIT_ORDER.value: int(get_config_value("limit_order_compute_limit", str(default_limit))),
                TransactionType.WITHDRAW.value: int(get_config_value("withdraw_compute_limit", str(default_limit)))
            }

            return compute_limits.get(tx_type, default_limit)
        except Exception as e:
            logger.error(f"Error calculating compute limit: {e}")
            return int(get_config_value("compute_unit_limit", "200000"))

    def get_fee_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get statistics about fees over a time period.

        Args:
            hours: Number of hours to analyze

        Returns:
            Dictionary with fee statistics
        """
        try:
            # Get entries from the specified time period
            current_time = datetime.now().timestamp()
            cutoff_time = current_time - (hours * 3600)

            entries = [entry for entry in self.fee_history if entry["timestamp"] >= cutoff_time]

            if not entries:
                return {
                    "error": "No fee data available for the specified time period"
                }

            # Extract fees for different percentiles
            percentiles = ["10", "25", "50", "75", "90", "99"]
            fees_by_percentile = {p: [] for p in percentiles}

            for entry in entries:
                for p in percentiles:
                    if p in entry["fees"]:
                        fees_by_percentile[p].append(entry["fees"][p])

            # Calculate statistics
            stats = {}
            for p, fees in fees_by_percentile.items():
                if fees:
                    stats[p] = {
                        "min": min(fees),
                        "max": max(fees),
                        "avg": sum(fees) / len(fees),
                        "median": statistics.median(fees) if len(fees) > 0 else 0,
                        "std_dev": statistics.stdev(fees) if len(fees) > 1 else 0
                    }

            # Calculate congestion statistics
            congestion_values = [entry.get("congestion", 0.5) for entry in entries]
            congestion_stats = {
                "min": min(congestion_values),
                "max": max(congestion_values),
                "avg": sum(congestion_values) / len(congestion_values),
                "median": statistics.median(congestion_values),
                "std_dev": statistics.stdev(congestion_values) if len(congestion_values) > 1 else 0
            }

            return {
                "time_period_hours": hours,
                "num_entries": len(entries),
                "percentiles": stats,
                "congestion": congestion_stats
            }
        except Exception as e:
            logger.error(f"Error calculating fee statistics: {e}")
            return {
                "error": str(e)
            }

    def optimize_transaction_timing(self, urgency: str = "normal") -> Dict[str, Any]:
        """
        Optimize transaction timing based on network conditions.

        Args:
            urgency: Transaction urgency (low, normal, high, immediate)

        Returns:
            Recommendation for transaction timing
        """
        try:
            # Get current congestion
            congestion = self._get_network_congestion()

            # Get current hour (UTC)
            current_hour = datetime.now(datetime.timezone.utc).hour

            # Define peak hours
            us_peak_hours = range(13, 21)  # 13:00-21:00 UTC (9am-5pm EST)
            asia_peak_hours = range(0, 8)  # 00:00-08:00 UTC (8am-4pm Asia)

            # Determine if we're in peak hours
            in_peak_hours = current_hour in us_peak_hours or current_hour in asia_peak_hours

            # Calculate optimal wait time based on congestion and urgency
            wait_time_minutes = 0

            if urgency == "immediate":
                wait_time_minutes = 0
            elif urgency == "high":
                if congestion > 0.8:
                    wait_time_minutes = 5
                else:
                    wait_time_minutes = 0
            elif urgency == "normal":
                if congestion > 0.8:
                    wait_time_minutes = 15
                elif congestion > 0.5:
                    wait_time_minutes = 5
                else:
                    wait_time_minutes = 0
            elif urgency == "low":
                if congestion > 0.8:
                    wait_time_minutes = 30
                elif congestion > 0.5:
                    wait_time_minutes = 15
                elif congestion > 0.3:
                    wait_time_minutes = 5
                else:
                    wait_time_minutes = 0

            # Calculate optimal time to execute
            optimal_time = datetime.now() + timedelta(minutes=wait_time_minutes)

            # Calculate fee savings
            fee_savings_percent = 0

            if congestion > 0.8:
                fee_savings_percent = 30
            elif congestion > 0.5:
                fee_savings_percent = 20
            elif congestion > 0.3:
                fee_savings_percent = 10

            # Adjust based on peak hours
            if in_peak_hours and wait_time_minutes == 0:
                fee_savings_percent += 10

                # For low urgency transactions, suggest waiting until off-peak
                if urgency == "low":
                    # Find next off-peak hour
                    next_off_peak_hour = 0
                    for hour in range(current_hour + 1, current_hour + 24):
                        hour_mod = hour % 24
                        if hour_mod not in us_peak_hours and hour_mod not in asia_peak_hours:
                            next_off_peak_hour = hour_mod
                            break

                    # Calculate time until next off-peak hour
                    hours_until_off_peak = (next_off_peak_hour - current_hour) % 24
                    optimal_time = datetime.now() + timedelta(hours=hours_until_off_peak)
                    wait_time_minutes = hours_until_off_peak * 60
                    fee_savings_percent += 10

            return {
                "current_congestion": congestion,
                "in_peak_hours": in_peak_hours,
                "urgency": urgency,
                "recommended_wait_minutes": wait_time_minutes,
                "optimal_execution_time": optimal_time.isoformat(),
                "estimated_fee_savings_percent": fee_savings_percent
            }
        except Exception as e:
            logger.error(f"Error optimizing transaction timing: {e}")
            return {
                "error": str(e),
                "recommended_wait_minutes": 0
            }

    def refine_fee_parameters(self, tx_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Refine fee parameters based on transaction history.

        Args:
            tx_history: List of transaction data with success/failure info
                Each entry should have:
                - signature: Transaction signature
                - success: Whether the transaction succeeded
                - priority_fee: The priority fee used
                - tx_type: The transaction type
                - congestion: Network congestion at the time
                - time_of_day: Hour of day (UTC)

        Returns:
            Dictionary with refinement results
        """
        if not self.enabled:
            logger.warning("Fee optimization is disabled, cannot refine parameters")
            return {"success": False, "reason": "Fee optimization is disabled"}

        try:
            logger.info("Refining fee parameters based on transaction history")

            if not tx_history:
                logger.warning("No transaction history provided for fee parameter refinement")
                return {"success": False, "reason": "No transaction history provided"}

            # Group transactions by type
            tx_by_type = {}
            for tx in tx_history:
                tx_type = tx.get("tx_type", TransactionType.DEFAULT.value)
                if tx_type not in tx_by_type:
                    tx_by_type[tx_type] = []
                tx_by_type[tx_type].append(tx)

            # Store original multipliers for reporting changes
            original_multipliers = self.tx_type_multipliers.copy()

            # Analyze and adjust multipliers for each transaction type
            adjustments = {}
            for tx_type, transactions in tx_by_type.items():
                if not transactions:
                    continue

                # Calculate success rate
                success_count = sum(1 for tx in transactions if tx.get("success", False))
                success_rate = success_count / len(transactions) if transactions else 0

                # Group by congestion level
                high_congestion_txs = [tx for tx in transactions if tx.get("congestion", 0.5) > 0.7]
                medium_congestion_txs = [tx for tx in transactions if 0.3 <= tx.get("congestion", 0.5) <= 0.7]
                low_congestion_txs = [tx for tx in transactions if tx.get("congestion", 0.5) < 0.3]

                # Calculate success rates by congestion level
                high_success_rate = sum(1 for tx in high_congestion_txs if tx.get("success", False)) / len(high_congestion_txs) if high_congestion_txs else 0
                medium_success_rate = sum(1 for tx in medium_congestion_txs if tx.get("success", False)) / len(medium_congestion_txs) if medium_congestion_txs else 0
                low_success_rate = sum(1 for tx in low_congestion_txs if tx.get("success", False)) / len(low_congestion_txs) if low_congestion_txs else 0

                # Calculate adjustment factor based on success rates
                adjustment_factor = 1.0

                # If success rate is too low in high congestion, increase multiplier
                if high_congestion_txs and high_success_rate < 0.8:
                    adjustment_factor = 1.2
                # If success rate is too low in medium congestion, slightly increase multiplier
                elif medium_congestion_txs and medium_success_rate < 0.9:
                    adjustment_factor = 1.1
                # If success rate is high across all congestion levels, slightly decrease multiplier
                elif success_rate > 0.95 and (not high_congestion_txs or high_success_rate > 0.9):
                    adjustment_factor = 0.95

                # Apply adjustment if needed
                if adjustment_factor != 1.0:
                    current_multiplier = self.tx_type_multipliers.get(tx_type, 1.0)
                    new_multiplier = current_multiplier * adjustment_factor

                    # Ensure multiplier stays within reasonable bounds
                    if tx_type == TransactionType.SNIPE.value:
                        new_multiplier = max(min(new_multiplier, 3.0), 1.2)
                    elif tx_type == TransactionType.WITHDRAW.value:
                        new_multiplier = max(min(new_multiplier, 2.0), 1.0)
                    else:
                        new_multiplier = max(min(new_multiplier, 2.0), 0.8)

                    # Update multiplier
                    self.tx_type_multipliers[tx_type] = new_multiplier

                    # Record adjustment
                    adjustments[tx_type] = {
                        "original": current_multiplier,
                        "new": new_multiplier,
                        "change_percent": ((new_multiplier - current_multiplier) / current_multiplier * 100) if current_multiplier > 0 else 0.0,
                        "success_rate": success_rate,
                        "high_congestion_success_rate": high_success_rate,
                        "medium_congestion_success_rate": medium_success_rate,
                        "low_congestion_success_rate": low_success_rate
                    }

            # Analyze time-of-day patterns
            time_success_rates = {}
            for hour in range(24):
                hour_txs = [tx for tx in tx_history if tx.get("time_of_day") == hour]
                if hour_txs:
                    success_count = sum(1 for tx in hour_txs if tx.get("success", False))
                    time_success_rates[hour] = success_count / len(hour_txs)

            # Identify problematic hours
            problematic_hours = [hour for hour, rate in time_success_rates.items() if rate < 0.85]

            # Update peak hour definitions if needed
            time_adjustments = {}
            if problematic_hours:
                # Check if we need to update US peak hours
                us_peak_problematic = [h for h in problematic_hours if 13 <= h < 21]
                if us_peak_problematic:
                    time_adjustments["us_peak_hours"] = {
                        "problematic_hours": us_peak_problematic,
                        "recommendation": "Increase fee multiplier during these hours"
                    }

                # Check if we need to update Asia peak hours
                asia_peak_problematic = [h for h in problematic_hours if 0 <= h < 8]
                if asia_peak_problematic:
                    time_adjustments["asia_peak_hours"] = {
                        "problematic_hours": asia_peak_problematic,
                        "recommendation": "Increase fee multiplier during these hours"
                    }

            # Save updated configuration
            for tx_type, multiplier in self.tx_type_multipliers.items():
                config_key = f"{tx_type}_fee_multiplier"
                update_config(config_key, str(multiplier))

            logger.info(f"Fee parameters refined based on {len(tx_history)} transactions")

            return {
                "success": True,
                "multiplier_adjustments": adjustments,
                "time_adjustments": time_adjustments,
                "transaction_count": len(tx_history)
            }
        except Exception as e:
            logger.error(f"Error refining fee parameters: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Create a singleton instance
gas_optimizer = GasOptimizer()
