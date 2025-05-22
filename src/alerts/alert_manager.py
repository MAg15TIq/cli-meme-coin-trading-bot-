"""
Alert Manager for Solana Memecoin Trading Bot.
Handles customizable alerts and notifications for various trading events.
"""

import os
import json
import logging
import smtplib
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from src.utils.logging_utils import get_logger
from src.trading.position_manager import position_manager
from src.trading.token_analytics import token_analytics
from src.trading.technical_analysis import technical_analyzer
from src.trading.sentiment_analysis import sentiment_analyzer
from src.utils.performance_tracker import performance_tracker
from src.security.security_manager import security_manager

logger = get_logger(__name__)

@dataclass
class AlertConfig:
    """Alert configuration settings."""
    data_dir: str = "data/alerts"
    email_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    webhook_enabled: bool = False
    webhook_url: str = ""
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    alert_cooldown: int = 300  # seconds
    max_alerts_per_hour: int = 10

@dataclass
class Alert:
    """Alert configuration."""
    id: str
    name: str
    type: str  # price, volume, technical, sentiment, portfolio, custom
    condition: str  # above, below, crosses, percentage_change
    value: float
    token_address: Optional[str] = None
    enabled: bool = True
    notification_channels: List[str] = None  # email, webhook, telegram
    cooldown: int = 300  # seconds
    last_triggered: Optional[datetime] = None

class AlertManager:
    def __init__(self):
        self.config = AlertConfig()
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[str, Callable] = {
            'price': self._check_price_alert,
            'volume': self._check_volume_alert,
            'technical': self._check_technical_alert,
            'sentiment': self._check_sentiment_alert,
            'portfolio': self._check_portfolio_alert,
            'custom': self._check_custom_alert
        }
        
        # Ensure data directory exists
        os.makedirs(self.config.data_dir, exist_ok=True)
        
        # Load saved alerts
        self._load_alerts()
        
    def _load_alerts(self) -> None:
        """Load saved alerts from file."""
        try:
            file_path = os.path.join(self.config.data_dir, 'alerts.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for alert_data in data:
                        alert = Alert(**alert_data)
                        self.alerts[alert.id] = alert
                logger.info(f"Loaded {len(self.alerts)} alerts")
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")
            
    def _save_alerts(self) -> None:
        """Save alerts to file."""
        try:
            file_path = os.path.join(self.config.data_dir, 'alerts.json')
            with open(file_path, 'w') as f:
                json.dump([vars(alert) for alert in self.alerts.values()], f)
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")
            
    def create_alert(self, alert: Alert) -> str:
        """Create a new alert."""
        try:
            # Validate alert
            if alert.type not in self.alert_handlers:
                raise ValueError(f"Invalid alert type: {alert.type}")
                
            # Generate unique ID
            alert.id = f"{alert.type}_{datetime.now().timestamp()}"
            
            # Save alert
            self.alerts[alert.id] = alert
            self._save_alerts()
            
            logger.info(f"Created alert: {alert.name}")
            return alert.id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            raise
            
    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing alert."""
        try:
            if alert_id not in self.alerts:
                raise ValueError(f"Alert not found: {alert_id}")
                
            alert = self.alerts[alert_id]
            for key, value in updates.items():
                if hasattr(alert, key):
                    setattr(alert, key, value)
                    
            self._save_alerts()
            logger.info(f"Updated alert: {alert.name}")
            
        except Exception as e:
            logger.error(f"Error updating alert: {e}")
            raise
            
    def delete_alert(self, alert_id: str) -> None:
        """Delete an alert."""
        try:
            if alert_id not in self.alerts:
                raise ValueError(f"Alert not found: {alert_id}")
                
            alert = self.alerts.pop(alert_id)
            self._save_alerts()
            logger.info(f"Deleted alert: {alert.name}")
            
        except Exception as e:
            logger.error(f"Error deleting alert: {e}")
            raise
            
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self.alerts.get(alert_id)
        
    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts."""
        return list(self.alerts.values())
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.alerts.values() if alert.enabled]
        
    def check_alerts(self) -> None:
        """Check all active alerts."""
        try:
            for alert in self.get_active_alerts():
                # Check cooldown
                if alert.last_triggered and (
                    datetime.now() - alert.last_triggered
                ).total_seconds() < alert.cooldown:
                    continue
                    
                # Check alert condition
                handler = self.alert_handlers[alert.type]
                if handler(alert):
                    self._trigger_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            
    def _check_price_alert(self, alert: Alert) -> bool:
        """Check price alert condition."""
        try:
            current_price = token_analytics.get_token_price(alert.token_address)
            
            if alert.condition == 'above':
                return current_price > alert.value
            elif alert.condition == 'below':
                return current_price < alert.value
            elif alert.condition == 'crosses':
                # Get previous price from cache or API
                previous_price = self._get_cached_price(alert.token_address)
                return (previous_price < alert.value and current_price > alert.value) or \
                       (previous_price > alert.value and current_price < alert.value)
            elif alert.condition == 'percentage_change':
                previous_price = self._get_cached_price(alert.token_address)
                change = (current_price - previous_price) / previous_price * 100
                return abs(change) >= alert.value
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking price alert: {e}")
            return False
            
    def _check_volume_alert(self, alert: Alert) -> bool:
        """Check volume alert condition."""
        try:
            volume_24h = token_analytics.get_token_volume(alert.token_address)
            
            if alert.condition == 'above':
                return volume_24h > alert.value
            elif alert.condition == 'below':
                return volume_24h < alert.value
            elif alert.condition == 'percentage_change':
                previous_volume = self._get_cached_volume(alert.token_address)
                change = (volume_24h - previous_volume) / previous_volume * 100
                return abs(change) >= alert.value
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking volume alert: {e}")
            return False
            
    def _check_technical_alert(self, alert: Alert) -> bool:
        """Check technical indicator alert condition."""
        try:
            analysis = technical_analyzer.get_token_analysis(alert.token_address)
            
            if alert.condition == 'rsi':
                return analysis['rsi'] > alert.value if alert.value > 50 else analysis['rsi'] < alert.value
            elif alert.condition == 'macd':
                return analysis['macd']['signal'] > alert.value
            elif alert.condition == 'bollinger':
                return analysis['bollinger']['upper'] > alert.value or \
                       analysis['bollinger']['lower'] < alert.value
                       
            return False
            
        except Exception as e:
            logger.error(f"Error checking technical alert: {e}")
            return False
            
    def _check_sentiment_alert(self, alert: Alert) -> bool:
        """Check sentiment alert condition."""
        try:
            sentiment = sentiment_analyzer.get_token_sentiment(alert.token_address)
            
            if alert.condition == 'score':
                return sentiment['score'] > alert.value
            elif alert.condition == 'change':
                previous_sentiment = self._get_cached_sentiment(alert.token_address)
                change = sentiment['score'] - previous_sentiment['score']
                return abs(change) >= alert.value
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking sentiment alert: {e}")
            return False
            
    def _check_portfolio_alert(self, alert: Alert) -> bool:
        """Check portfolio alert condition."""
        try:
            metrics = performance_tracker.get_performance_metrics()
            
            if alert.condition == 'total_value':
                return metrics['total_value'] > alert.value
            elif alert.condition == 'daily_pnl':
                return metrics['daily_pnl'] > alert.value
            elif alert.condition == 'drawdown':
                return metrics['max_drawdown'] > alert.value
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking portfolio alert: {e}")
            return False
            
    def _check_custom_alert(self, alert: Alert) -> bool:
        """Check custom alert condition."""
        try:
            # Evaluate custom condition
            # This is a placeholder for custom alert conditions
            # Users can implement their own logic here
            return False
            
        except Exception as e:
            logger.error(f"Error checking custom alert: {e}")
            return False
            
    def _trigger_alert(self, alert: Alert) -> None:
        """Trigger alert notifications."""
        try:
            # Update last triggered time
            alert.last_triggered = datetime.now()
            self._save_alerts()
            
            # Prepare message
            message = self._format_alert_message(alert)
            
            # Send notifications
            if 'email' in alert.notification_channels and self.config.email_enabled:
                self._send_email_alert(message)
                
            if 'webhook' in alert.notification_channels and self.config.webhook_enabled:
                self._send_webhook_alert(message)
                
            if 'telegram' in alert.notification_channels and self.config.telegram_enabled:
                self._send_telegram_alert(message)
                
            logger.info(f"Triggered alert: {alert.name}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
            
    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert message."""
        try:
            if alert.type == 'price':
                current_price = token_analytics.get_token_price(alert.token_address)
                return f"Price Alert: {alert.name}\n" \
                       f"Token: {alert.token_address}\n" \
                       f"Current Price: ${current_price:.6f}\n" \
                       f"Condition: {alert.condition} {alert.value}"
                       
            elif alert.type == 'volume':
                volume_24h = token_analytics.get_token_volume(alert.token_address)
                return f"Volume Alert: {alert.name}\n" \
                       f"Token: {alert.token_address}\n" \
                       f"24h Volume: ${volume_24h:.2f}\n" \
                       f"Condition: {alert.condition} {alert.value}"
                       
            elif alert.type == 'technical':
                return f"Technical Alert: {alert.name}\n" \
                       f"Token: {alert.token_address}\n" \
                       f"Indicator: {alert.condition}\n" \
                       f"Value: {alert.value}"
                       
            elif alert.type == 'sentiment':
                sentiment = sentiment_analyzer.get_token_sentiment(alert.token_address)
                return f"Sentiment Alert: {alert.name}\n" \
                       f"Token: {alert.token_address}\n" \
                       f"Current Score: {sentiment['score']:.2f}\n" \
                       f"Condition: {alert.condition} {alert.value}"
                       
            elif alert.type == 'portfolio':
                metrics = performance_tracker.get_performance_metrics()
                return f"Portfolio Alert: {alert.name}\n" \
                       f"Total Value: ${metrics['total_value']:.2f}\n" \
                       f"Daily P/L: ${metrics['daily_pnl']:.2f}\n" \
                       f"Condition: {alert.condition} {alert.value}"
                       
            else:
                return f"Alert: {alert.name}\n" \
                       f"Type: {alert.type}\n" \
                       f"Condition: {alert.condition} {alert.value}"
                       
        except Exception as e:
            logger.error(f"Error formatting alert message: {e}")
            return f"Alert: {alert.name}"
            
    def _send_email_alert(self, message: str) -> None:
        """Send alert via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = self.config.smtp_username
            msg['Subject'] = "Trading Bot Alert"
            
            msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            
    def _send_webhook_alert(self, message: str) -> None:
        """Send alert via webhook."""
        try:
            payload = {
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=5
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
            
    def _send_telegram_alert(self, message: str) -> None:
        """Send alert via Telegram."""
        try:
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.config.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
            
    def _get_cached_price(self, token_address: str) -> float:
        """Get cached token price."""
        # Implement price caching logic
        return token_analytics.get_token_price(token_address)
        
    def _get_cached_volume(self, token_address: str) -> float:
        """Get cached token volume."""
        # Implement volume caching logic
        return token_analytics.get_token_volume(token_address)
        
    def _get_cached_sentiment(self, token_address: str) -> Dict[str, float]:
        """Get cached token sentiment."""
        # Implement sentiment caching logic
        return sentiment_analyzer.get_token_sentiment(token_address)

# Global instance
alert_manager = AlertManager() 