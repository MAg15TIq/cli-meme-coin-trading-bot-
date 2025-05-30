"""
Advanced Alert System for the Solana Memecoin Trading Bot.
Provides comprehensive alerting for price movements, trading opportunities, and system events.
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.trading.position_manager import position_manager
from src.trading.jupiter_api import jupiter_api
from src.notifications.notification_service import notification_service

# Get logger for this module
logger = get_logger(__name__)


class AlertType(Enum):
    """Types of alerts."""
    PRICE_MOVEMENT = "price_movement"
    VOLUME_SPIKE = "volume_spike"
    NEW_POOL = "new_pool"
    WHALE_MOVEMENT = "whale_movement"
    TECHNICAL_SIGNAL = "technical_signal"
    PORTFOLIO_ALERT = "portfolio_alert"
    RISK_ALERT = "risk_alert"
    SYSTEM_ALERT = "system_alert"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertCondition:
    """Represents an alert condition."""
    id: str
    alert_type: AlertType
    priority: AlertPriority
    token_mint: Optional[str]
    token_symbol: Optional[str]
    condition: Dict[str, Any]
    enabled: bool
    created_at: datetime
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class Alert:
    """Represents a triggered alert."""
    id: str
    condition_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    token_mint: Optional[str]
    token_symbol: Optional[str]
    data: Dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False


class AdvancedAlertSystem:
    """Advanced alert system with comprehensive monitoring."""

    def __init__(self):
        """Initialize the advanced alert system."""
        self.enabled = get_config_value("advanced_alerts_enabled", False)

        # Alert conditions and history
        self.alert_conditions = {}  # condition_id -> AlertCondition
        self.alert_history = []  # List of triggered alerts
        self.max_alert_history = int(get_config_value("max_alert_history", "1000"))

        # Monitoring settings
        self.monitoring_interval = int(get_config_value("alert_monitoring_interval", "30"))  # seconds
        self.price_check_interval = int(get_config_value("alert_price_check_interval", "10"))  # seconds

        # Notification settings
        self.notification_enabled = get_config_value("alert_notifications_enabled", True)
        self.notification_channels = get_config_value("alert_notification_channels", ["console", "log"])

        # Rate limiting
        self.rate_limit_window = int(get_config_value("alert_rate_limit_window", "300"))  # 5 minutes
        self.max_alerts_per_window = int(get_config_value("max_alerts_per_window", "10"))
        self.alert_timestamps = []

        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = False

        # Load existing conditions
        self._load_alert_conditions()

        logger.info("Advanced alert system initialized")

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the alert system.

        Args:
            enabled: Whether to enable alerts
        """
        self.enabled = enabled
        update_config("advanced_alerts_enabled", enabled)

        if enabled:
            self.start_monitoring()
        else:
            self.stop_monitoring()

        logger.info(f"Advanced alert system {'enabled' if enabled else 'disabled'}")

    def add_price_alert(self, token_mint: str, token_symbol: str,
                       condition_type: str, threshold: float,
                       priority: AlertPriority = AlertPriority.MEDIUM) -> str:
        """
        Add a price-based alert condition.

        Args:
            token_mint: Token mint address
            token_symbol: Token symbol
            condition_type: Type of condition ('above', 'below', 'change_percent')
            threshold: Threshold value
            priority: Alert priority

        Returns:
            Alert condition ID
        """
        condition_id = f"price_{token_mint}_{int(time.time())}"

        condition = AlertCondition(
            id=condition_id,
            alert_type=AlertType.PRICE_MOVEMENT,
            priority=priority,
            token_mint=token_mint,
            token_symbol=token_symbol,
            condition={
                "type": condition_type,
                "threshold": threshold,
                "baseline_price": None  # Will be set when monitoring starts
            },
            enabled=True,
            created_at=datetime.now()
        )

        self.alert_conditions[condition_id] = condition
        self._save_alert_conditions()

        logger.info(f"Added price alert: {token_symbol} {condition_type} {threshold}")
        return condition_id

    def add_volume_alert(self, token_mint: str, token_symbol: str,
                        volume_multiplier: float,
                        priority: AlertPriority = AlertPriority.MEDIUM) -> str:
        """
        Add a volume spike alert condition.

        Args:
            token_mint: Token mint address
            token_symbol: Token symbol
            volume_multiplier: Volume multiplier threshold (e.g., 5.0 for 5x normal volume)
            priority: Alert priority

        Returns:
            Alert condition ID
        """
        condition_id = f"volume_{token_mint}_{int(time.time())}"

        condition = AlertCondition(
            id=condition_id,
            alert_type=AlertType.VOLUME_SPIKE,
            priority=priority,
            token_mint=token_mint,
            token_symbol=token_symbol,
            condition={
                "volume_multiplier": volume_multiplier,
                "baseline_volume": None  # Will be calculated from historical data
            },
            enabled=True,
            created_at=datetime.now()
        )

        self.alert_conditions[condition_id] = condition
        self._save_alert_conditions()

        logger.info(f"Added volume alert: {token_symbol} volume spike {volume_multiplier}x")
        return condition_id

    def add_whale_alert(self, token_mint: str, token_symbol: str,
                       min_transaction_size: float,
                       priority: AlertPriority = AlertPriority.HIGH) -> str:
        """
        Add a whale movement alert condition.

        Args:
            token_mint: Token mint address
            token_symbol: Token symbol
            min_transaction_size: Minimum transaction size in SOL to trigger alert
            priority: Alert priority

        Returns:
            Alert condition ID
        """
        condition_id = f"whale_{token_mint}_{int(time.time())}"

        condition = AlertCondition(
            id=condition_id,
            alert_type=AlertType.WHALE_MOVEMENT,
            priority=priority,
            token_mint=token_mint,
            token_symbol=token_symbol,
            condition={
                "min_transaction_size": min_transaction_size
            },
            enabled=True,
            created_at=datetime.now()
        )

        self.alert_conditions[condition_id] = condition
        self._save_alert_conditions()

        logger.info(f"Added whale alert: {token_symbol} transactions >= {min_transaction_size} SOL")
        return condition_id

    def add_portfolio_alert(self, alert_type: str, threshold: float,
                           priority: AlertPriority = AlertPriority.MEDIUM) -> str:
        """
        Add a portfolio-based alert condition.

        Args:
            alert_type: Type of portfolio alert ('total_loss', 'position_loss', 'concentration')
            threshold: Threshold value
            priority: Alert priority

        Returns:
            Alert condition ID
        """
        condition_id = f"portfolio_{alert_type}_{int(time.time())}"

        condition = AlertCondition(
            id=condition_id,
            alert_type=AlertType.PORTFOLIO_ALERT,
            priority=priority,
            token_mint=None,
            token_symbol=None,
            condition={
                "type": alert_type,
                "threshold": threshold
            },
            enabled=True,
            created_at=datetime.now()
        )

        self.alert_conditions[condition_id] = condition
        self._save_alert_conditions()

        logger.info(f"Added portfolio alert: {alert_type} threshold {threshold}")
        return condition_id

    def remove_alert_condition(self, condition_id: str) -> bool:
        """
        Remove an alert condition.

        Args:
            condition_id: Alert condition ID to remove

        Returns:
            True if removed, False if not found
        """
        if condition_id in self.alert_conditions:
            condition = self.alert_conditions[condition_id]
            del self.alert_conditions[condition_id]
            self._save_alert_conditions()
            logger.info(f"Removed alert condition: {condition_id}")
            return True
        return False

    def enable_alert_condition(self, condition_id: str, enabled: bool) -> bool:
        """
        Enable or disable an alert condition.

        Args:
            condition_id: Alert condition ID
            enabled: Whether to enable the condition

        Returns:
            True if updated, False if not found
        """
        if condition_id in self.alert_conditions:
            self.alert_conditions[condition_id].enabled = enabled
            self._save_alert_conditions()
            logger.info(f"Alert condition {condition_id} {'enabled' if enabled else 'disabled'}")
            return True
        return False

    def trigger_alert(self, condition_id: str, title: str, message: str,
                     data: Dict[str, Any] = None) -> None:
        """
        Trigger an alert.

        Args:
            condition_id: Alert condition ID
            title: Alert title
            message: Alert message
            data: Additional alert data
        """
        if not self.enabled:
            return

        # Check rate limiting
        if not self._check_rate_limit():
            logger.warning("Alert rate limit exceeded, skipping alert")
            return

        condition = self.alert_conditions.get(condition_id)
        if not condition or not condition.enabled:
            return

        # Create alert
        alert_id = f"alert_{int(time.time())}_{len(self.alert_history)}"
        alert = Alert(
            id=alert_id,
            condition_id=condition_id,
            alert_type=condition.alert_type,
            priority=condition.priority,
            title=title,
            message=message,
            token_mint=condition.token_mint,
            token_symbol=condition.token_symbol,
            data=data or {},
            timestamp=datetime.now()
        )

        # Add to history
        self.alert_history.append(alert)

        # Keep history within limits
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history = self.alert_history[-self.max_alert_history:]

        # Update condition
        condition.last_triggered = datetime.now()
        condition.trigger_count += 1

        # Send notification
        if self.notification_enabled:
            self._send_notification(alert)

        logger.info(f"Alert triggered: {title}")

    def start_monitoring(self) -> None:
        """Start the alert monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Alert monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the alert monitoring thread."""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Alert monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring:
            try:
                if self.enabled:
                    # Check all alert conditions
                    for condition in self.alert_conditions.values():
                        if condition.enabled:
                            self._check_condition(condition)

                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def _check_condition(self, condition: AlertCondition) -> None:
        """
        Check if an alert condition is met.

        Args:
            condition: Alert condition to check
        """
        try:
            if condition.alert_type == AlertType.PRICE_MOVEMENT:
                self._check_price_condition(condition)
            elif condition.alert_type == AlertType.VOLUME_SPIKE:
                self._check_volume_condition(condition)
            elif condition.alert_type == AlertType.WHALE_MOVEMENT:
                self._check_whale_condition(condition)
            elif condition.alert_type == AlertType.PORTFOLIO_ALERT:
                self._check_portfolio_condition(condition)
        except Exception as e:
            logger.error(f"Error checking condition {condition.id}: {e}")

    def _check_price_condition(self, condition: AlertCondition) -> None:
        """Check price-based alert condition."""
        if not condition.token_mint:
            return

        try:
            # Get current price
            current_price = jupiter_api.get_token_price(condition.token_mint)
            if current_price is None:
                return

            condition_data = condition.condition
            condition_type = condition_data.get("type")
            threshold = condition_data.get("threshold")

            # Set baseline price if not set
            if condition_data.get("baseline_price") is None:
                condition_data["baseline_price"] = current_price
                return

            baseline_price = condition_data["baseline_price"]

            # Check condition
            triggered = False
            message = ""

            if condition_type == "above" and current_price > threshold:
                triggered = True
                message = f"{condition.token_symbol} price ${current_price:.6f} is above threshold ${threshold:.6f}"
            elif condition_type == "below" and current_price < threshold:
                triggered = True
                message = f"{condition.token_symbol} price ${current_price:.6f} is below threshold ${threshold:.6f}"
            elif condition_type == "change_percent":
                change_percent = ((current_price - baseline_price) / baseline_price) * 100
                if abs(change_percent) >= threshold:
                    triggered = True
                    direction = "increased" if change_percent > 0 else "decreased"
                    message = f"{condition.token_symbol} price {direction} by {abs(change_percent):.2f}% (threshold: {threshold}%)"

            if triggered:
                self.trigger_alert(
                    condition.id,
                    f"Price Alert: {condition.token_symbol}",
                    message,
                    {
                        "current_price": current_price,
                        "baseline_price": baseline_price,
                        "threshold": threshold,
                        "condition_type": condition_type
                    }
                )
        except Exception as e:
            logger.error(f"Error checking price condition for {condition.token_symbol}: {e}")

    def _check_rate_limit(self) -> bool:
        """
        Check if alert rate limit is exceeded.

        Returns:
            True if within rate limit, False otherwise
        """
        current_time = time.time()

        # Remove old timestamps
        self.alert_timestamps = [
            ts for ts in self.alert_timestamps
            if current_time - ts < self.rate_limit_window
        ]

        # Check if we're within the limit
        if len(self.alert_timestamps) >= self.max_alerts_per_window:
            return False

        # Add current timestamp
        self.alert_timestamps.append(current_time)
        return True

    def _send_notification(self, alert: Alert) -> None:
        """
        Send notification for an alert.

        Args:
            alert: Alert to send notification for
        """
        try:
            # Format notification message
            notification_message = f"[{alert.priority.value.upper()}] {alert.title}: {alert.message}"

            # Send to configured channels
            for channel in self.notification_channels:
                if channel == "console":
                    print(f"\n🚨 ALERT: {notification_message}")
                elif channel == "log":
                    if alert.priority == AlertPriority.CRITICAL:
                        logger.critical(notification_message)
                    elif alert.priority == AlertPriority.HIGH:
                        logger.error(notification_message)
                    elif alert.priority == AlertPriority.MEDIUM:
                        logger.warning(notification_message)
                    else:
                        logger.info(notification_message)
        except Exception as e:
            logger.error(f"Error sending notification: {e}")


# Create global instance
advanced_alert_system = AdvancedAlertSystem()