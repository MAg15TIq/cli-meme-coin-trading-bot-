"""
Security manager for the Solana Memecoin Trading Bot.
Implements multi-factor authentication, transaction signing verification,
honeypot detection, and rate limiting.
"""

import time
import logging
import hashlib
import hmac
import base64
import pyotp
import qrcode
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json
import requests
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.utils.performance_optimizer import cache_result
from src.trading.token_analytics import token_analytics
from src.wallet.wallet import wallet_manager

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    mfa_enabled: bool
    mfa_secret: str
    rate_limit_requests: int
    rate_limit_period: int
    max_daily_trades: int
    max_position_size: float
    require_2fa_for_trades: bool
    require_2fa_for_withdrawals: bool
    ip_whitelist: List[str]
    allowed_tokens: List[str]

class SecurityManager:
    def __init__(self):
        self.config: Optional[SecurityConfig] = None
        self.rate_limits: Dict[str, List[float]] = {}
        self.daily_trades: Dict[str, int] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.locked_until: Dict[str, datetime] = {}
        self.encryption_key: Optional[bytes] = None
        
        # Load configuration
        self._load_config()
        
        # Initialize encryption
        self._init_encryption()
        
    def _load_config(self) -> None:
        """Load security configuration."""
        config_path = Path("config/security.json")
        if not config_path.exists():
            # Create default configuration
            self.config = SecurityConfig(
                mfa_enabled=False,
                mfa_secret="",
                rate_limit_requests=100,
                rate_limit_period=60,
                max_daily_trades=50,
                max_position_size=1000.0,
                require_2fa_for_trades=True,
                require_2fa_for_withdrawals=True,
                ip_whitelist=[],
                allowed_tokens=[]
            )
            self._save_config()
        else:
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                self.config = SecurityConfig(**config_data)
            except Exception as e:
                logger.error(f"Error loading security config: {e}")
                self.config = SecurityConfig(
                    mfa_enabled=False,
                    mfa_secret="",
                    rate_limit_requests=100,
                    rate_limit_period=60,
                    max_daily_trades=50,
                    max_position_size=1000.0,
                    require_2fa_for_trades=True,
                    require_2fa_for_withdrawals=True,
                    ip_whitelist=[],
                    allowed_tokens=[]
                )
    
    def _save_config(self) -> None:
        """Save security configuration."""
        if not self.config:
            return
            
        config_path = Path("config/security.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving security config: {e}")
    
    def _init_encryption(self) -> None:
        """Initialize encryption key."""
        try:
            key_path = Path("config/encryption.key")
            if key_path.exists():
                with open(key_path, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()
                key_path.parent.mkdir(parents=True, exist_ok=True)
                with open(key_path, 'wb') as f:
                    f.write(self.encryption_key)
        except Exception as e:
            logger.error(f"Error initializing encryption: {e}")
            self.encryption_key = None
    
    def setup_mfa(self, user_id: str) -> Tuple[bool, str]:
        """
        Set up multi-factor authentication for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (success, qr_code_data)
        """
        try:
            # Generate secret
            secret = pyotp.random_base32()
            
            # Create TOTP object
            totp = pyotp.TOTP(secret)
            
            # Generate provisioning URI
            provisioning_uri = totp.provisioning_uri(
                user_id,
                issuer_name="Solana Memecoin Trading Bot"
            )
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            # Save secret
            self.config.mfa_secret = secret
            self.config.mfa_enabled = True
            self._save_config()
            
            return True, provisioning_uri
            
        except Exception as e:
            logger.error(f"Error setting up MFA: {e}")
            return False, ""
    
    def verify_mfa(self, code: str) -> bool:
        """
        Verify MFA code.
        
        Args:
            code: MFA code to verify
            
        Returns:
            True if code is valid
        """
        try:
            if not self.config.mfa_enabled or not self.config.mfa_secret:
                return False
                
            totp = pyotp.TOTP(self.config.mfa_secret)
            return totp.verify(code)
            
        except Exception as e:
            logger.error(f"Error verifying MFA code: {e}")
            return False
    
    def check_rate_limit(self, user_id: str) -> bool:
        """
        Check if user has exceeded rate limits.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if within limits
        """
        try:
            if user_id not in self.rate_limits:
                self.rate_limits[user_id] = []
            
            # Remove old timestamps
            current_time = time.time()
            self.rate_limits[user_id] = [
                t for t in self.rate_limits[user_id]
                if current_time - t < self.config.rate_limit_period
            ]
            
            # Check if limit exceeded
            if len(self.rate_limits[user_id]) >= self.config.rate_limit_requests:
                return False
            
            # Add new timestamp
            self.rate_limits[user_id].append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    def check_daily_trades(self, user_id: str) -> bool:
        """
        Check if user has exceeded daily trade limit.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if within limits
        """
        try:
            # Reset counter if new day
            today = datetime.now().date()
            if user_id not in self.daily_trades:
                self.daily_trades[user_id] = {'date': today, 'count': 0}
            elif self.daily_trades[user_id]['date'] != today:
                self.daily_trades[user_id] = {'date': today, 'count': 0}
            
            # Check limit
            return self.daily_trades[user_id]['count'] < self.config.max_daily_trades
            
        except Exception as e:
            logger.error(f"Error checking daily trades: {e}")
            return False
    
    def increment_daily_trades(self, user_id: str) -> None:
        """Increment daily trade counter for user."""
        if user_id in self.daily_trades:
            self.daily_trades[user_id]['count'] += 1
    
    def verify_transaction(self, transaction: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify transaction signature and parameters.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check if transaction is signed
            if not transaction.get('signature'):
                return False, "Transaction not signed"
            
            # Verify signature
            if not wallet_manager.verify_signature(
                transaction['signature'],
                transaction['message']
            ):
                return False, "Invalid signature"
            
            # Check amount limits
            if transaction.get('amount', 0) > self.config.max_position_size:
                return False, "Amount exceeds maximum position size"
            
            # Check if token is allowed
            if (self.config.allowed_tokens and
                transaction.get('token_address') not in self.config.allowed_tokens):
                return False, "Token not in allowed list"
            
            return True, "Transaction verified"
            
        except Exception as e:
            logger.error(f"Error verifying transaction: {e}")
            return False, str(e)
    
    @cache_result(ttl=300)
    def check_honeypot(self, token_address: str) -> Tuple[bool, str]:
        """
        Check if token is a honeypot.
        
        Args:
            token_address: Token address to check
            
        Returns:
            Tuple of (is_honeypot, reason)
        """
        try:
            # Get token analytics
            analytics = token_analytics.get_token_analytics(token_address)
            
            # Check for honeypot indicators
            if analytics.get('is_honeypot', False):
                return True, "Token identified as honeypot"
            
            # Check liquidity
            if analytics.get('liquidity', 0) < 1000:  # Less than 1000 SOL liquidity
                return True, "Suspiciously low liquidity"
            
            # Check holder distribution
            holders = analytics.get('holders', {})
            if len(holders) < 10:  # Less than 10 holders
                return True, "Suspiciously few holders"
            
            # Check for large holder concentration
            if holders:
                top_holder_percentage = max(holders.values())
                if top_holder_percentage > 0.5:  # Single holder owns more than 50%
                    return True, "High holder concentration"
            
            return False, "Token appears safe"
            
        except Exception as e:
            logger.error(f"Error checking honeypot: {e}")
            return True, f"Error during check: {str(e)}"
    
    def encrypt_sensitive_data(self, data: str) -> Optional[str]:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        try:
            if not self.encryption_key:
                return None
                
            f = Fernet(self.encryption_key)
            return f.encrypt(data.encode()).decode()
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            return None
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Optional[str]:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: Data to decrypt
            
        Returns:
            Decrypted data
        """
        try:
            if not self.encryption_key:
                return None
                
            f = Fernet(self.encryption_key)
            return f.decrypt(encrypted_data.encode()).decode()
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return None
    
    def check_ip_whitelist(self, ip_address: str) -> bool:
        """
        Check if IP address is whitelisted.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            True if IP is whitelisted
        """
        return not self.config.ip_whitelist or ip_address in self.config.ip_whitelist
    
    def handle_failed_attempt(self, user_id: str) -> None:
        """
        Handle failed authentication attempt.
        
        Args:
            user_id: User identifier
        """
        try:
            if user_id not in self.failed_attempts:
                self.failed_attempts[user_id] = 0
            
            self.failed_attempts[user_id] += 1
            
            # Lock account after 5 failed attempts
            if self.failed_attempts[user_id] >= 5:
                self.locked_until[user_id] = datetime.now() + timedelta(minutes=30)
                logger.warning(f"Account locked for {user_id} due to multiple failed attempts")
                
        except Exception as e:
            logger.error(f"Error handling failed attempt: {e}")
    
    def is_account_locked(self, user_id: str) -> bool:
        """
        Check if account is locked.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if account is locked
        """
        try:
            if user_id not in self.locked_until:
                return False
                
            if datetime.now() > self.locked_until[user_id]:
                # Reset lock
                del self.locked_until[user_id]
                self.failed_attempts[user_id] = 0
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking account lock: {e}")
            return True

# Global instance
security_manager = SecurityManager() 