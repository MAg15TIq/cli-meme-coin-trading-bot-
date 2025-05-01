"""
Notification service for the Solana Memecoin Trading Bot.
Handles sending notifications to various channels including mobile devices.
"""

import json
import logging
import time
import threading
import requests
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
from pathlib import Path

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class NotificationPriority(Enum):
    """Priority levels for notifications."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationType(Enum):
    """Types of notifications."""
    TRADE = "trade"
    POSITION = "position"
    PRICE_ALERT = "price_alert"
    SYSTEM = "system"
    SECURITY = "security"
    SENTIMENT = "sentiment"
    STRATEGY = "strategy"
    WALLET = "wallet"


class NotificationChannel(Enum):
    """Channels for sending notifications."""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"


class NotificationService:
    """Service for sending notifications to various channels."""
    
    def __init__(self):
        """Initialize the notification service."""
        self.enabled = get_config_value("notifications_enabled", False)
        self.channels = get_config_value("notification_channels", {})
        self.notification_settings = get_config_value("notification_settings", {})
        
        # Queue for pending notifications
        self.notification_queue: List[Dict[str, Any]] = []
        
        # Thread for processing notifications
        self.notification_thread = None
        self.stop_notification_thread = threading.Event()
        
        # Load notification templates
        self.templates = self._load_templates()
        
        # Initialize notification channels
        self._init_channels()
    
    def _load_templates(self) -> Dict[str, str]:
        """
        Load notification templates.
        
        Returns:
            Dictionary of templates
        """
        templates = {
            "trade_executed": "Trade executed: {action} {amount} {token} at {price}",
            "position_opened": "Position opened: {amount} {token} at {price}",
            "position_closed": "Position closed: {amount} {token} at {price} ({profit_loss})",
            "stop_loss_triggered": "Stop loss triggered for {token} at {price} ({profit_loss})",
            "take_profit_triggered": "Take profit triggered for {token} at {price} ({profit_loss})",
            "price_alert": "Price alert: {token} is now {price} ({change})",
            "system_error": "System error: {message}",
            "system_warning": "System warning: {message}",
            "system_info": "System info: {message}",
            "security_alert": "Security alert: {message}",
            "sentiment_alert": "Sentiment alert for {token}: {message}",
            "strategy_alert": "Strategy alert: {message}",
            "wallet_alert": "Wallet alert: {message}"
        }
        
        # Override with custom templates if available
        custom_templates = get_config_value("notification_templates", {})
        templates.update(custom_templates)
        
        return templates
    
    def _init_channels(self) -> None:
        """Initialize notification channels."""
        # Initialize Telegram
        if "telegram" in self.channels:
            self._init_telegram()
        
        # Initialize Discord
        if "discord" in self.channels:
            self._init_discord()
        
        # Initialize email
        if "email" in self.channels:
            self._init_email()
        
        # Initialize push notifications
        if "push" in self.channels:
            self._init_push()
        
        # Initialize SMS
        if "sms" in self.channels:
            self._init_sms()
    
    def _init_telegram(self) -> None:
        """Initialize Telegram bot."""
        telegram_config = self.channels.get("telegram", {})
        self.telegram_bot_token = telegram_config.get("bot_token", "")
        self.telegram_chat_id = telegram_config.get("chat_id", "")
        
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram bot token or chat ID not configured")
            return
        
        logger.info("Telegram notifications initialized")
    
    def _init_discord(self) -> None:
        """Initialize Discord webhook."""
        discord_config = self.channels.get("discord", {})
        self.discord_webhook_url = discord_config.get("webhook_url", "")
        
        if not self.discord_webhook_url:
            logger.warning("Discord webhook URL not configured")
            return
        
        logger.info("Discord notifications initialized")
    
    def _init_email(self) -> None:
        """Initialize email notifications."""
        email_config = self.channels.get("email", {})
        self.email_sender = email_config.get("sender", "")
        self.email_recipient = email_config.get("recipient", "")
        self.email_smtp_server = email_config.get("smtp_server", "")
        self.email_smtp_port = email_config.get("smtp_port", 587)
        self.email_smtp_username = email_config.get("smtp_username", "")
        self.email_smtp_password = email_config.get("smtp_password", "")
        
        if not all([self.email_sender, self.email_recipient, self.email_smtp_server, 
                   self.email_smtp_username, self.email_smtp_password]):
            logger.warning("Email configuration incomplete")
            return
        
        logger.info("Email notifications initialized")
    
    def _init_push(self) -> None:
        """Initialize push notifications."""
        push_config = self.channels.get("push", {})
        self.push_service = push_config.get("service", "firebase")
        self.push_api_key = push_config.get("api_key", "")
        self.push_device_tokens = push_config.get("device_tokens", [])
        
        if not self.push_api_key or not self.push_device_tokens:
            logger.warning("Push notification configuration incomplete")
            return
        
        logger.info(f"Push notifications initialized using {self.push_service}")
    
    def _init_sms(self) -> None:
        """Initialize SMS notifications."""
        sms_config = self.channels.get("sms", {})
        self.sms_service = sms_config.get("service", "twilio")
        self.sms_account_sid = sms_config.get("account_sid", "")
        self.sms_auth_token = sms_config.get("auth_token", "")
        self.sms_from_number = sms_config.get("from_number", "")
        self.sms_to_numbers = sms_config.get("to_numbers", [])
        
        if not all([self.sms_account_sid, self.sms_auth_token, self.sms_from_number]) or not self.sms_to_numbers:
            logger.warning("SMS configuration incomplete")
            return
        
        logger.info(f"SMS notifications initialized using {self.sms_service}")
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable notifications.
        
        Args:
            enabled: Whether notifications should be enabled
        """
        self.enabled = enabled
        update_config("notifications_enabled", enabled)
        logger.info(f"Notifications {'enabled' if enabled else 'disabled'}")
        
        if enabled:
            self.start_notification_thread()
        else:
            self.stop_notification_thread_event()
    
    def start_notification_thread(self) -> None:
        """Start the notification processing thread."""
        if self.notification_thread and self.notification_thread.is_alive():
            logger.info("Notification thread already running")
            return
        
        self.stop_notification_thread.clear()
        self.notification_thread = threading.Thread(target=self._process_notifications)
        self.notification_thread.daemon = True
        self.notification_thread.start()
        
        logger.info("Started notification processing thread")
    
    def stop_notification_thread_event(self) -> None:
        """Stop the notification processing thread."""
        if not self.notification_thread or not self.notification_thread.is_alive():
            logger.info("Notification thread not running")
            return
        
        self.stop_notification_thread.set()
        self.notification_thread.join(timeout=5)
        
        logger.info("Stopped notification processing thread")
    
    def _process_notifications(self) -> None:
        """Process notifications in the queue."""
        while not self.stop_notification_thread.is_set():
            try:
                # Process all notifications in the queue
                while self.notification_queue:
                    notification = self.notification_queue.pop(0)
                    self._send_notification(notification)
            except Exception as e:
                logger.error(f"Error processing notifications: {e}")
            
            # Sleep for a short time
            time.sleep(1)
    
    def _send_notification(self, notification: Dict[str, Any]) -> None:
        """
        Send a notification to all configured channels.
        
        Args:
            notification: Notification data
        """
        notification_type = notification.get("type")
        priority = notification.get("priority", NotificationPriority.NORMAL.value)
        
        # Check if this type of notification is enabled
        if not self._is_notification_enabled(notification_type, priority):
            logger.debug(f"Notification of type {notification_type} with priority {priority} is disabled")
            return
        
        # Get channels for this notification type
        channels = self._get_channels_for_notification(notification_type, priority)
        
        # Send to each channel
        for channel in channels:
            try:
                if channel == NotificationChannel.TELEGRAM.value:
                    self._send_telegram(notification)
                elif channel == NotificationChannel.DISCORD.value:
                    self._send_discord(notification)
                elif channel == NotificationChannel.EMAIL.value:
                    self._send_email(notification)
                elif channel == NotificationChannel.PUSH.value:
                    self._send_push(notification)
                elif channel == NotificationChannel.SMS.value:
                    self._send_sms(notification)
            except Exception as e:
                logger.error(f"Error sending {notification_type} notification to {channel}: {e}")
    
    def _is_notification_enabled(self, notification_type: str, priority: str) -> bool:
        """
        Check if a notification type is enabled.
        
        Args:
            notification_type: Type of notification
            priority: Priority level
            
        Returns:
            True if enabled, False otherwise
        """
        # Check global enabled status
        if not self.enabled:
            return False
        
        # Check type-specific settings
        type_settings = self.notification_settings.get(notification_type, {})
        enabled = type_settings.get("enabled", True)
        
        # Check priority threshold
        priority_threshold = type_settings.get("min_priority", NotificationPriority.NORMAL.value)
        priority_levels = {
            NotificationPriority.LOW.value: 0,
            NotificationPriority.NORMAL.value: 1,
            NotificationPriority.HIGH.value: 2,
            NotificationPriority.CRITICAL.value: 3
        }
        
        priority_met = priority_levels.get(priority, 1) >= priority_levels.get(priority_threshold, 1)
        
        return enabled and priority_met
    
    def _get_channels_for_notification(self, notification_type: str, priority: str) -> List[str]:
        """
        Get channels for a notification type.
        
        Args:
            notification_type: Type of notification
            priority: Priority level
            
        Returns:
            List of channel names
        """
        # Get type-specific channels
        type_settings = self.notification_settings.get(notification_type, {})
        type_channels = type_settings.get("channels", [])
        
        # If no type-specific channels, use all configured channels
        if not type_channels:
            type_channels = list(self.channels.keys())
        
        # For critical notifications, use all available channels
        if priority == NotificationPriority.CRITICAL.value:
            return list(self.channels.keys())
        
        return type_channels
    
    def _format_message(self, notification: Dict[str, Any]) -> str:
        """
        Format a notification message using templates.
        
        Args:
            notification: Notification data
            
        Returns:
            Formatted message
        """
        notification_type = notification.get("type")
        template_key = notification.get("template", notification_type)
        template = self.templates.get(template_key, "{message}")
        
        # Format the template with notification data
        try:
            message = template.format(**notification.get("data", {}))
        except KeyError as e:
            logger.error(f"Error formatting notification template: {e}")
            message = notification.get("data", {}).get("message", "Notification")
        
        # Add timestamp
        timestamp = notification.get("timestamp", datetime.now().isoformat())
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        
        # Format: [PRIORITY] Message (Timestamp)
        priority = notification.get("priority", NotificationPriority.NORMAL.value).upper()
        return f"[{priority}] {message} ({timestamp})"
    
    def _send_telegram(self, notification: Dict[str, Any]) -> None:
        """
        Send a notification via Telegram.
        
        Args:
            notification: Notification data
        """
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram not configured, skipping notification")
            return
        
        message = self._format_message(notification)
        
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        data = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, data=data)
        
        if response.status_code != 200:
            logger.error(f"Error sending Telegram notification: {response.text}")
        else:
            logger.info(f"Sent Telegram notification: {message[:50]}...")
    
    def _send_discord(self, notification: Dict[str, Any]) -> None:
        """
        Send a notification via Discord.
        
        Args:
            notification: Notification data
        """
        if not self.discord_webhook_url:
            logger.warning("Discord not configured, skipping notification")
            return
        
        message = self._format_message(notification)
        
        data = {
            "content": message
        }
        
        response = requests.post(self.discord_webhook_url, json=data)
        
        if response.status_code not in [200, 204]:
            logger.error(f"Error sending Discord notification: {response.text}")
        else:
            logger.info(f"Sent Discord notification: {message[:50]}...")
    
    def _send_email(self, notification: Dict[str, Any]) -> None:
        """
        Send a notification via email.
        
        Args:
            notification: Notification data
        """
        if not all([self.email_sender, self.email_recipient, self.email_smtp_server, 
                   self.email_smtp_username, self.email_smtp_password]):
            logger.warning("Email not configured, skipping notification")
            return
        
        message = self._format_message(notification)
        
        # This is a simplified implementation
        # In a real implementation, we would use smtplib to send the email
        logger.info(f"Would send email notification: {message[:50]}...")
    
    def _send_push(self, notification: Dict[str, Any]) -> None:
        """
        Send a push notification.
        
        Args:
            notification: Notification data
        """
        if not self.push_api_key or not self.push_device_tokens:
            logger.warning("Push notifications not configured, skipping notification")
            return
        
        message = self._format_message(notification)
        
        # This is a simplified implementation
        # In a real implementation, we would use a push notification service
        logger.info(f"Would send push notification: {message[:50]}...")
    
    def _send_sms(self, notification: Dict[str, Any]) -> None:
        """
        Send an SMS notification.
        
        Args:
            notification: Notification data
        """
        if not all([self.sms_account_sid, self.sms_auth_token, self.sms_from_number]) or not self.sms_to_numbers:
            logger.warning("SMS not configured, skipping notification")
            return
        
        message = self._format_message(notification)
        
        # This is a simplified implementation
        # In a real implementation, we would use a service like Twilio
        logger.info(f"Would send SMS notification: {message[:50]}...")
    
    def send_notification(self, notification_type: str, data: Dict[str, Any], 
                         priority: str = NotificationPriority.NORMAL.value,
                         template: Optional[str] = None) -> None:
        """
        Queue a notification for sending.
        
        Args:
            notification_type: Type of notification
            data: Notification data
            priority: Priority level
            template: Template name (optional)
        """
        if not self.enabled:
            logger.debug(f"Notifications disabled, skipping {notification_type} notification")
            return
        
        notification = {
            "type": notification_type,
            "data": data,
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        }
        
        if template:
            notification["template"] = template
        
        # Add to queue
        self.notification_queue.append(notification)
        
        logger.debug(f"Queued {notification_type} notification with priority {priority}")
    
    def send_trade_notification(self, action: str, token: str, amount: float, price: float, 
                              priority: str = NotificationPriority.NORMAL.value) -> None:
        """
        Send a trade notification.
        
        Args:
            action: Trade action (buy, sell)
            token: Token name or symbol
            amount: Amount of tokens
            price: Price per token
            priority: Priority level
        """
        data = {
            "action": action,
            "token": token,
            "amount": amount,
            "price": price
        }
        
        self.send_notification(
            notification_type=NotificationType.TRADE.value,
            data=data,
            priority=priority,
            template="trade_executed"
        )
    
    def send_position_notification(self, action: str, token: str, amount: float, price: float, 
                                 profit_loss: Optional[str] = None,
                                 priority: str = NotificationPriority.NORMAL.value) -> None:
        """
        Send a position notification.
        
        Args:
            action: Position action (opened, closed, stop_loss, take_profit)
            token: Token name or symbol
            amount: Amount of tokens
            price: Price per token
            profit_loss: Profit/loss string (optional)
            priority: Priority level
        """
        data = {
            "action": action,
            "token": token,
            "amount": amount,
            "price": price,
            "profit_loss": profit_loss or "N/A"
        }
        
        template = f"position_{action}"
        if action == "stop_loss":
            template = "stop_loss_triggered"
        elif action == "take_profit":
            template = "take_profit_triggered"
        
        self.send_notification(
            notification_type=NotificationType.POSITION.value,
            data=data,
            priority=priority,
            template=template
        )
    
    def send_price_alert(self, token: str, price: float, change: str,
                       priority: str = NotificationPriority.NORMAL.value) -> None:
        """
        Send a price alert notification.
        
        Args:
            token: Token name or symbol
            price: Current price
            change: Price change string
            priority: Priority level
        """
        data = {
            "token": token,
            "price": price,
            "change": change
        }
        
        self.send_notification(
            notification_type=NotificationType.PRICE_ALERT.value,
            data=data,
            priority=priority,
            template="price_alert"
        )
    
    def send_system_notification(self, message: str, level: str = "info",
                               priority: str = NotificationPriority.NORMAL.value) -> None:
        """
        Send a system notification.
        
        Args:
            message: System message
            level: Message level (info, warning, error)
            priority: Priority level
        """
        data = {
            "message": message,
            "level": level
        }
        
        template = f"system_{level}"
        
        self.send_notification(
            notification_type=NotificationType.SYSTEM.value,
            data=data,
            priority=priority,
            template=template
        )
    
    def send_security_alert(self, message: str, priority: str = NotificationPriority.HIGH.value) -> None:
        """
        Send a security alert notification.
        
        Args:
            message: Alert message
            priority: Priority level
        """
        data = {
            "message": message
        }
        
        self.send_notification(
            notification_type=NotificationType.SECURITY.value,
            data=data,
            priority=priority,
            template="security_alert"
        )
    
    def send_sentiment_alert(self, token: str, message: str,
                           priority: str = NotificationPriority.NORMAL.value) -> None:
        """
        Send a sentiment alert notification.
        
        Args:
            token: Token name or symbol
            message: Alert message
            priority: Priority level
        """
        data = {
            "token": token,
            "message": message
        }
        
        self.send_notification(
            notification_type=NotificationType.SENTIMENT.value,
            data=data,
            priority=priority,
            template="sentiment_alert"
        )
    
    def send_strategy_alert(self, message: str, priority: str = NotificationPriority.NORMAL.value) -> None:
        """
        Send a strategy alert notification.
        
        Args:
            message: Alert message
            priority: Priority level
        """
        data = {
            "message": message
        }
        
        self.send_notification(
            notification_type=NotificationType.STRATEGY.value,
            data=data,
            priority=priority,
            template="strategy_alert"
        )
    
    def send_wallet_alert(self, message: str, priority: str = NotificationPriority.NORMAL.value) -> None:
        """
        Send a wallet alert notification.
        
        Args:
            message: Alert message
            priority: Priority level
        """
        data = {
            "message": message
        }
        
        self.send_notification(
            notification_type=NotificationType.WALLET.value,
            data=data,
            priority=priority,
            template="wallet_alert"
        )


# Create a singleton instance
notification_service = NotificationService()
