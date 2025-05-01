"""
Price alerts module for the Solana Memecoin Trading Bot.
Allows setting and managing price alerts for tokens.
"""

import json
import logging
import threading
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.trading.jupiter_api import jupiter_api
from src.notifications.notification_service import notification_service, NotificationPriority

# Get logger for this module
logger = get_logger(__name__)


class AlertCondition:
    """Enum-like class for alert conditions."""
    ABOVE = "above"
    BELOW = "below"
    PERCENT_INCREASE = "percent_increase"
    PERCENT_DECREASE = "percent_decrease"


class PriceAlertManager:
    """Manager for token price alerts."""
    
    def __init__(self):
        """Initialize the price alert manager."""
        self.enabled = get_config_value("price_alerts_enabled", False)
        self.alerts: Dict[str, Dict[str, Any]] = {}
        self.triggered_alerts: List[Dict[str, Any]] = []
        self.last_prices: Dict[str, float] = {}
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.monitoring_interval = int(get_config_value("price_alert_interval_seconds", "60"))  # Default: 60 seconds
        
        # Load alerts
        self._load_alerts()
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable price alerts.
        
        Args:
            enabled: Whether price alerts should be enabled
        """
        self.enabled = enabled
        update_config("price_alerts_enabled", enabled)
        logger.info(f"Price alerts {'enabled' if enabled else 'disabled'}")
        
        if enabled and not self.monitoring_thread:
            self.start_monitoring_thread()
        elif not enabled and self.monitoring_thread:
            self.stop_monitoring_thread()
    
    def _load_alerts(self) -> None:
        """Load alerts from config."""
        self.alerts = get_config_value("price_alerts", {})
        self.triggered_alerts = get_config_value("triggered_price_alerts", [])
        logger.info(f"Loaded {len(self.alerts)} price alerts")
    
    def _save_alerts(self) -> None:
        """Save alerts to config."""
        update_config("price_alerts", self.alerts)
        update_config("triggered_price_alerts", self.triggered_alerts)
    
    def start_monitoring_thread(self) -> None:
        """Start the monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Price alert monitoring thread already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Price alert monitoring thread started")
    
    def stop_monitoring_thread(self) -> None:
        """Stop the monitoring thread."""
        if not self.monitoring_thread:
            logger.warning("Price alert monitoring thread not running")
            return
        
        self.stop_monitoring.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.monitoring_thread = None
        logger.info("Price alert monitoring thread stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Check all active alerts
                self._check_alerts()
            except Exception as e:
                logger.error(f"Error in price alert monitoring loop: {e}")
            
            # Sleep for the monitoring interval
            self.stop_monitoring.wait(self.monitoring_interval)
    
    def _check_alerts(self) -> None:
        """Check all active alerts."""
        if not self.alerts:
            return
        
        # Get unique token mints from alerts
        token_mints = set()
        for alert_id, alert in self.alerts.items():
            if alert["active"]:
                token_mints.add(alert["token_mint"])
        
        # Get current prices for all tokens
        current_prices = {}
        for token_mint in token_mints:
            try:
                price = jupiter_api.get_token_price(token_mint)
                if price is not None:
                    current_prices[token_mint] = price
                    self.last_prices[token_mint] = price
            except Exception as e:
                logger.error(f"Error getting price for {token_mint}: {e}")
        
        # Check each alert
        for alert_id, alert in list(self.alerts.items()):
            if not alert["active"]:
                continue
            
            token_mint = alert["token_mint"]
            if token_mint not in current_prices:
                continue
            
            current_price = current_prices[token_mint]
            condition = alert["condition"]
            target_price = alert["target_price"]
            
            # Check if alert is triggered
            triggered = False
            if condition == AlertCondition.ABOVE and current_price >= target_price:
                triggered = True
            elif condition == AlertCondition.BELOW and current_price <= target_price:
                triggered = True
            elif condition == AlertCondition.PERCENT_INCREASE:
                # Get reference price
                reference_price = alert.get("reference_price")
                if reference_price is None:
                    # Use price when alert was created
                    reference_price = alert.get("price_at_creation", 0)
                    if reference_price == 0:
                        # Use last known price
                        reference_price = self.last_prices.get(token_mint, current_price)
                
                # Calculate percent change
                if reference_price > 0:
                    percent_change = ((current_price - reference_price) / reference_price) * 100
                    if percent_change >= target_price:
                        triggered = True
            
            elif condition == AlertCondition.PERCENT_DECREASE:
                # Get reference price
                reference_price = alert.get("reference_price")
                if reference_price is None:
                    # Use price when alert was created
                    reference_price = alert.get("price_at_creation", 0)
                    if reference_price == 0:
                        # Use last known price
                        reference_price = self.last_prices.get(token_mint, current_price)
                
                # Calculate percent change
                if reference_price > 0:
                    percent_change = ((reference_price - current_price) / reference_price) * 100
                    if percent_change >= target_price:
                        triggered = True
            
            if triggered:
                # Mark alert as triggered
                alert["active"] = False
                alert["triggered_at"] = datetime.now().isoformat()
                alert["price_at_trigger"] = current_price
                
                # Add to triggered alerts
                self.triggered_alerts.append({
                    "alert_id": alert_id,
                    "token_mint": token_mint,
                    "token_symbol": alert["token_symbol"],
                    "condition": condition,
                    "target_price": target_price,
                    "price_at_trigger": current_price,
                    "triggered_at": datetime.now().isoformat()
                })
                
                # Save alerts
                self._save_alerts()
                
                # Send notification
                self._send_alert_notification(alert, current_price)
                
                logger.info(f"Price alert triggered: {alert['token_symbol']} {condition} {target_price}")
    
    def _send_alert_notification(self, alert: Dict[str, Any], current_price: float) -> None:
        """
        Send notification for triggered alert.
        
        Args:
            alert: The triggered alert
            current_price: Current price of the token
        """
        token_symbol = alert["token_symbol"]
        condition = alert["condition"]
        target_price = alert["target_price"]
        
        # Format message based on condition
        if condition == AlertCondition.ABOVE:
            message = f"{token_symbol} price is now above {target_price:.6f}: {current_price:.6f}"
        elif condition == AlertCondition.BELOW:
            message = f"{token_symbol} price is now below {target_price:.6f}: {current_price:.6f}"
        elif condition == AlertCondition.PERCENT_INCREASE:
            message = f"{token_symbol} price increased by {target_price:.2f}%: {current_price:.6f}"
        elif condition == AlertCondition.PERCENT_DECREASE:
            message = f"{token_symbol} price decreased by {target_price:.2f}%: {current_price:.6f}"
        else:
            message = f"{token_symbol} price alert triggered: {current_price:.6f}"
        
        # Send notification
        notification_service.send_price_alert(
            token_symbol=token_symbol,
            message=message,
            price=current_price,
            alert_id=alert.get("id", ""),
            priority=NotificationPriority.HIGH.value
        )
    
    def add_alert(self, token_mint: str, token_symbol: str, condition: str, 
                 target_price: float, reference_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Add a new price alert.
        
        Args:
            token_mint: Token mint address
            token_symbol: Token symbol
            condition: Alert condition (above, below, percent_increase, percent_decrease)
            target_price: Target price or percentage
            reference_price: Reference price for percentage alerts (optional)
            
        Returns:
            The created alert
        """
        # Validate condition
        if condition not in [AlertCondition.ABOVE, AlertCondition.BELOW, 
                            AlertCondition.PERCENT_INCREASE, AlertCondition.PERCENT_DECREASE]:
            raise ValueError(f"Invalid condition: {condition}")
        
        # Validate target price
        if target_price <= 0:
            raise ValueError("Target price must be greater than 0")
        
        # Get current price if needed
        current_price = None
        if reference_price is None:
            current_price = jupiter_api.get_token_price(token_mint)
            reference_price = current_price
        
        # Generate alert ID
        alert_id = str(uuid.uuid4())
        
        # Create alert
        alert = {
            "id": alert_id,
            "token_mint": token_mint,
            "token_symbol": token_symbol,
            "condition": condition,
            "target_price": target_price,
            "reference_price": reference_price,
            "price_at_creation": current_price or reference_price,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        # Add to alerts
        self.alerts[alert_id] = alert
        
        # Save alerts
        self._save_alerts()
        
        logger.info(f"Added price alert: {token_symbol} {condition} {target_price}")
        return alert
    
    def remove_alert(self, alert_id: str) -> bool:
        """
        Remove a price alert.
        
        Args:
            alert_id: ID of the alert to remove
            
        Returns:
            True if alert was removed, False otherwise
        """
        if alert_id not in self.alerts:
            logger.warning(f"Alert not found: {alert_id}")
            return False
        
        # Remove alert
        del self.alerts[alert_id]
        
        # Save alerts
        self._save_alerts()
        
        logger.info(f"Removed price alert: {alert_id}")
        return True
    
    def get_alerts(self, active_only: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get all price alerts.
        
        Args:
            active_only: Whether to return only active alerts
            
        Returns:
            Dictionary of alerts
        """
        if active_only:
            return {alert_id: alert for alert_id, alert in self.alerts.items() if alert["active"]}
        return self.alerts
    
    def get_triggered_alerts(self) -> List[Dict[str, Any]]:
        """
        Get triggered alerts.
        
        Returns:
            List of triggered alerts
        """
        return self.triggered_alerts
    
    def clear_triggered_alerts(self) -> None:
        """Clear triggered alerts."""
        self.triggered_alerts = []
        self._save_alerts()
        logger.info("Cleared triggered alerts")
    
    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific alert.
        
        Args:
            alert_id: ID of the alert
            
        Returns:
            The alert, or None if not found
        """
        return self.alerts.get(alert_id)


# Create singleton instance
price_alert_manager = PriceAlertManager()
