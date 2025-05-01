"""
Mobile companion app integration for the Solana Memecoin Trading Bot.
Provides functionality for mobile notifications and remote monitoring.
"""

import json
import logging
import time
import base64
import qrcode
import io
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import uuid

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.notifications.notification_service import notification_service, NotificationPriority

# Get logger for this module
logger = get_logger(__name__)


class MobileAppManager:
    """Manager for mobile app integration."""

    def __init__(self):
        """Initialize the mobile app manager."""
        self.enabled = get_config_value("mobile_app_enabled", False)
        self.api_key = get_config_value("mobile_app_api_key", "")
        self.device_tokens = get_config_value("mobile_app_device_tokens", [])
        self.pairing_code = get_config_value("mobile_app_pairing_code", "")
        self.pairing_expiry = get_config_value("mobile_app_pairing_expiry", 0)
        
        # Path for storing mobile app data
        self.data_path = Path(get_config_value("mobile_app_data_path", 
                                             str(Path.home() / ".solana-trading-bot" / "mobile_app")))
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # QR code path
        self.qr_code_path = self.data_path / "pairing_qr.png"
        
        # Notification settings
        self.notification_settings = get_config_value("mobile_app_notification_settings", {
            "trade_executed": True,
            "price_alert": True,
            "wallet_connected": True,
            "position_closed": True,
            "error": True,
            "security_alert": True
        })
        
        # Generate API key if not exists
        if not self.api_key:
            self._generate_api_key()
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable mobile app integration.
        
        Args:
            enabled: Whether mobile app integration should be enabled
        """
        self.enabled = enabled
        update_config("mobile_app_enabled", enabled)
        logger.info(f"Mobile app integration {'enabled' if enabled else 'disabled'}")
    
    def _generate_api_key(self) -> None:
        """Generate a new API key."""
        import secrets
        self.api_key = secrets.token_hex(16)
        update_config("mobile_app_api_key", self.api_key)
        logger.info("Generated new mobile app API key")
    
    def generate_pairing_code(self) -> str:
        """
        Generate a new pairing code for mobile app connection.
        
        Returns:
            The generated pairing code
        """
        import random
        import string
        
        # Generate a random 8-character code
        code_chars = string.ascii_uppercase + string.digits
        code = ''.join(random.choice(code_chars) for _ in range(8))
        
        # Set expiry time (10 minutes from now)
        expiry = int(time.time()) + 600
        
        # Save to config
        self.pairing_code = code
        self.pairing_expiry = expiry
        update_config("mobile_app_pairing_code", code)
        update_config("mobile_app_pairing_expiry", expiry)
        
        logger.info(f"Generated new pairing code: {code} (expires in 10 minutes)")
        return code
    
    def generate_pairing_qr(self) -> Optional[str]:
        """
        Generate a QR code for mobile app pairing.
        
        Returns:
            Path to the generated QR code image, or None if failed
        """
        try:
            # Generate pairing code if not exists or expired
            current_time = int(time.time())
            if not self.pairing_code or current_time >= self.pairing_expiry:
                self.generate_pairing_code()
            
            # Create pairing data
            pairing_data = {
                "type": "solana_trading_bot_pairing",
                "code": self.pairing_code,
                "api_key": self.api_key,
                "expiry": self.pairing_expiry,
                "bot_id": get_config_value("bot_id", str(uuid.uuid4()))
            }
            
            # Convert to JSON
            pairing_json = json.dumps(pairing_data)
            
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(pairing_json)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Save image
            img.save(self.qr_code_path)
            
            logger.info(f"Generated pairing QR code: {self.qr_code_path}")
            return str(self.qr_code_path)
        except Exception as e:
            logger.error(f"Error generating pairing QR code: {e}")
            return None
    
    def verify_pairing_code(self, code: str, device_token: str) -> bool:
        """
        Verify a pairing code and register a device.
        
        Args:
            code: The pairing code to verify
            device_token: The device token to register
            
        Returns:
            True if pairing successful, False otherwise
        """
        current_time = int(time.time())
        
        # Check if code is valid and not expired
        if code != self.pairing_code or current_time >= self.pairing_expiry:
            logger.warning(f"Invalid or expired pairing code: {code}")
            return False
        
        # Register device token
        if device_token not in self.device_tokens:
            self.device_tokens.append(device_token)
            update_config("mobile_app_device_tokens", self.device_tokens)
            logger.info(f"Registered new device: {device_token[:8]}...")
        
        # Clear pairing code
        self.pairing_code = ""
        self.pairing_expiry = 0
        update_config("mobile_app_pairing_code", "")
        update_config("mobile_app_pairing_expiry", 0)
        
        # Send test notification
        self.send_notification(
            title="Pairing Successful",
            message="Your mobile device has been paired with Solana Trading Bot",
            priority="normal",
            data={"type": "pairing_success"}
        )
        
        return True
    
    def send_notification(self, title: str, message: str, priority: str = "normal", 
                         data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a notification to all registered mobile devices.
        
        Args:
            title: Notification title
            message: Notification message
            priority: Notification priority (normal, high)
            data: Additional data to include
            
        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Mobile app integration is disabled")
            return False
        
        if not self.device_tokens:
            logger.warning("No registered devices")
            return False
        
        try:
            # This is a simplified implementation
            # In a real implementation, we would use Firebase Cloud Messaging or similar
            
            # Create notification payload
            payload = {
                "notification": {
                    "title": title,
                    "body": message,
                    "sound": "default",
                    "badge": 1,
                    "priority": priority
                },
                "data": data or {},
                "tokens": self.device_tokens
            }
            
            # Log notification
            logger.info(f"Would send mobile notification: {title} - {message}")
            logger.debug(f"Notification payload: {payload}")
            
            # In a real implementation, we would send the notification to FCM
            # For now, just simulate success
            return True
        except Exception as e:
            logger.error(f"Error sending mobile notification: {e}")
            return False
    
    def update_notification_settings(self, settings: Dict[str, bool]) -> None:
        """
        Update notification settings.
        
        Args:
            settings: Dictionary of notification settings
        """
        # Update only provided settings
        for key, value in settings.items():
            if key in self.notification_settings:
                self.notification_settings[key] = value
        
        # Save to config
        update_config("mobile_app_notification_settings", self.notification_settings)
        logger.info(f"Updated mobile notification settings: {settings}")
    
    def get_notification_settings(self) -> Dict[str, bool]:
        """
        Get current notification settings.
        
        Returns:
            Dictionary of notification settings
        """
        return self.notification_settings
    
    def remove_device(self, device_token: str) -> bool:
        """
        Remove a registered device.
        
        Args:
            device_token: The device token to remove
            
        Returns:
            True if device removed, False otherwise
        """
        if device_token in self.device_tokens:
            self.device_tokens.remove(device_token)
            update_config("mobile_app_device_tokens", self.device_tokens)
            logger.info(f"Removed device: {device_token[:8]}...")
            return True
        
        logger.warning(f"Device not found: {device_token[:8]}...")
        return False
    
    def get_registered_devices(self) -> List[str]:
        """
        Get list of registered devices.
        
        Returns:
            List of device tokens
        """
        return self.device_tokens


# Create singleton instance
mobile_app_manager = MobileAppManager()
