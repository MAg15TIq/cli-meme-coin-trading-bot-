"""
Limit orders module for the Solana Memecoin Trading Bot.
Allows setting and managing limit orders for buying and selling tokens.
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
from src.trading.position_manager import position_manager
from src.notifications.notification_service import notification_service, NotificationPriority
from src.wallet.wallet import wallet_manager

# Get logger for this module
logger = get_logger(__name__)


class OrderType:
    """Enum-like class for order types."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus:
    """Enum-like class for order statuses."""
    ACTIVE = "active"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


class LimitOrderManager:
    """Manager for limit orders."""
    
    def __init__(self):
        """Initialize the limit order manager."""
        self.enabled = get_config_value("limit_orders_enabled", False)
        self.orders: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.monitoring_interval = int(get_config_value("limit_order_interval_seconds", "30"))  # Default: 30 seconds
        
        # Load orders
        self._load_orders()
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable limit orders.
        
        Args:
            enabled: Whether limit orders should be enabled
        """
        self.enabled = enabled
        update_config("limit_orders_enabled", enabled)
        logger.info(f"Limit orders {'enabled' if enabled else 'disabled'}")
        
        if enabled and not self.monitoring_thread:
            self.start_monitoring_thread()
        elif not enabled and self.monitoring_thread:
            self.stop_monitoring_thread()
    
    def _load_orders(self) -> None:
        """Load orders from config."""
        self.orders = get_config_value("limit_orders", {})
        logger.info(f"Loaded {len(self.orders)} limit orders")
    
    def _save_orders(self) -> None:
        """Save orders to config."""
        update_config("limit_orders", self.orders)
    
    def start_monitoring_thread(self) -> None:
        """Start the monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Limit order monitoring thread already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Limit order monitoring thread started")
    
    def stop_monitoring_thread(self) -> None:
        """Stop the monitoring thread."""
        if not self.monitoring_thread:
            logger.warning("Limit order monitoring thread not running")
            return
        
        self.stop_monitoring.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.monitoring_thread = None
        logger.info("Limit order monitoring thread stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Check all active orders
                self._check_orders()
            except Exception as e:
                logger.error(f"Error in limit order monitoring loop: {e}")
            
            # Sleep for the monitoring interval
            self.stop_monitoring.wait(self.monitoring_interval)
    
    def _check_orders(self) -> None:
        """Check all active orders."""
        if not self.orders:
            return
        
        # Get unique token mints from orders
        token_mints = set()
        for order_id, order in self.orders.items():
            if order["status"] == OrderStatus.ACTIVE:
                token_mints.add(order["token_mint"])
        
        # Get current prices for all tokens
        current_prices = {}
        for token_mint in token_mints:
            try:
                price = jupiter_api.get_token_price(token_mint)
                if price is not None:
                    current_prices[token_mint] = price
            except Exception as e:
                logger.error(f"Error getting price for {token_mint}: {e}")
        
        # Check each order
        for order_id, order in list(self.orders.items()):
            if order["status"] != OrderStatus.ACTIVE:
                continue
            
            token_mint = order["token_mint"]
            if token_mint not in current_prices:
                continue
            
            current_price = current_prices[token_mint]
            order_type = order["type"]
            target_price = order["target_price"]
            
            # Check if order should be executed
            should_execute = False
            if order_type == OrderType.BUY and current_price <= target_price:
                should_execute = True
            elif order_type == OrderType.SELL and current_price >= target_price:
                should_execute = True
            
            if should_execute:
                # Execute order
                self._execute_order(order_id, order, current_price)
            
            # Check if order has expired
            if "expiry" in order and order["expiry"]:
                expiry_time = datetime.fromisoformat(order["expiry"])
                if datetime.now() > expiry_time:
                    # Mark order as expired
                    order["status"] = OrderStatus.EXPIRED
                    order["expired_at"] = datetime.now().isoformat()
                    self._save_orders()
                    
                    # Send notification
                    notification_service.send_order_alert(
                        message=f"Limit order expired: {order['token_symbol']} {order_type} at {target_price}",
                        priority=NotificationPriority.NORMAL.value
                    )
                    
                    logger.info(f"Limit order expired: {order_id}")
    
    def _execute_order(self, order_id: str, order: Dict[str, Any], current_price: float) -> None:
        """
        Execute a limit order.
        
        Args:
            order_id: ID of the order
            order: Order data
            current_price: Current price of the token
        """
        try:
            # Check if wallet is connected
            if not wallet_manager.current_keypair:
                logger.warning("Cannot execute order: wallet not connected")
                return
            
            token_mint = order["token_mint"]
            token_symbol = order["token_symbol"]
            order_type = order["type"]
            amount = order["amount"]
            
            # Execute order
            if order_type == OrderType.BUY:
                # Buy token
                result = position_manager.buy_token(
                    token_mint=token_mint,
                    amount_sol=amount,
                    token_symbol=token_symbol
                )
                
                if result["success"]:
                    # Mark order as executed
                    order["status"] = OrderStatus.EXECUTED
                    order["executed_at"] = datetime.now().isoformat()
                    order["execution_price"] = current_price
                    order["transaction_signature"] = result["signature"]
                    self._save_orders()
                    
                    # Send notification
                    notification_service.send_order_alert(
                        message=f"Limit buy order executed: {token_symbol} at {current_price}",
                        priority=NotificationPriority.HIGH.value
                    )
                    
                    logger.info(f"Limit buy order executed: {order_id}")
                else:
                    # Mark order as failed
                    order["status"] = OrderStatus.FAILED
                    order["failed_at"] = datetime.now().isoformat()
                    order["error"] = result.get("error", "Unknown error")
                    self._save_orders()
                    
                    # Send notification
                    notification_service.send_order_alert(
                        message=f"Limit buy order failed: {token_symbol} at {current_price}",
                        priority=NotificationPriority.HIGH.value
                    )
                    
                    logger.error(f"Limit buy order failed: {order_id} - {result.get('error', 'Unknown error')}")
            
            elif order_type == OrderType.SELL:
                # Sell token
                result = position_manager.sell_token(
                    token_mint=token_mint,
                    percentage=100.0  # Sell all
                )
                
                if result["success"]:
                    # Mark order as executed
                    order["status"] = OrderStatus.EXECUTED
                    order["executed_at"] = datetime.now().isoformat()
                    order["execution_price"] = current_price
                    order["transaction_signature"] = result["signature"]
                    self._save_orders()
                    
                    # Send notification
                    notification_service.send_order_alert(
                        message=f"Limit sell order executed: {token_symbol} at {current_price}",
                        priority=NotificationPriority.HIGH.value
                    )
                    
                    logger.info(f"Limit sell order executed: {order_id}")
                else:
                    # Mark order as failed
                    order["status"] = OrderStatus.FAILED
                    order["failed_at"] = datetime.now().isoformat()
                    order["error"] = result.get("error", "Unknown error")
                    self._save_orders()
                    
                    # Send notification
                    notification_service.send_order_alert(
                        message=f"Limit sell order failed: {token_symbol} at {current_price}",
                        priority=NotificationPriority.HIGH.value
                    )
                    
                    logger.error(f"Limit sell order failed: {order_id} - {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error executing limit order {order_id}: {e}")
            
            # Mark order as failed
            order["status"] = OrderStatus.FAILED
            order["failed_at"] = datetime.now().isoformat()
            order["error"] = str(e)
            self._save_orders()
            
            # Send notification
            notification_service.send_order_alert(
                message=f"Error executing limit order: {order['token_symbol']} - {str(e)}",
                priority=NotificationPriority.HIGH.value
            )
    
    def create_limit_order(self, token_mint: str, token_symbol: str, order_type: str, 
                          target_price: float, amount: float, 
                          expiry: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Create a new limit order.
        
        Args:
            token_mint: Token mint address
            token_symbol: Token symbol
            order_type: Order type (buy, sell)
            target_price: Target price
            amount: Amount in SOL (for buy) or percentage (for sell)
            expiry: Optional expiry time
            
        Returns:
            The created order
        """
        # Validate order type
        if order_type not in [OrderType.BUY, OrderType.SELL]:
            raise ValueError(f"Invalid order type: {order_type}")
        
        # Validate target price
        if target_price <= 0:
            raise ValueError("Target price must be greater than 0")
        
        # Validate amount
        if amount <= 0:
            raise ValueError("Amount must be greater than 0")
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Create order
        order = {
            "id": order_id,
            "token_mint": token_mint,
            "token_symbol": token_symbol,
            "type": order_type,
            "target_price": target_price,
            "amount": amount,
            "created_at": datetime.now().isoformat(),
            "status": OrderStatus.ACTIVE,
            "expiry": expiry.isoformat() if expiry else None
        }
        
        # Add to orders
        self.orders[order_id] = order
        
        # Save orders
        self._save_orders()
        
        logger.info(f"Created limit order: {token_symbol} {order_type} at {target_price}")
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a limit order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if order was cancelled, False otherwise
        """
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        # Get order
        order = self.orders[order_id]
        
        # Check if order is active
        if order["status"] != OrderStatus.ACTIVE:
            logger.warning(f"Order is not active: {order_id}")
            return False
        
        # Mark order as cancelled
        order["status"] = OrderStatus.CANCELLED
        order["cancelled_at"] = datetime.now().isoformat()
        
        # Save orders
        self._save_orders()
        
        logger.info(f"Cancelled limit order: {order_id}")
        return True
    
    def get_orders(self, status: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get all limit orders.
        
        Args:
            status: Optional status filter
            
        Returns:
            Dictionary of orders
        """
        if status:
            return {order_id: order for order_id, order in self.orders.items() if order["status"] == status}
        return self.orders
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific order.
        
        Args:
            order_id: ID of the order
            
        Returns:
            The order, or None if not found
        """
        return self.orders.get(order_id)
    
    def clear_inactive_orders(self) -> int:
        """
        Clear all inactive orders (executed, cancelled, expired, failed).
        
        Returns:
            Number of orders cleared
        """
        # Get inactive orders
        inactive_orders = [order_id for order_id, order in self.orders.items() 
                          if order["status"] != OrderStatus.ACTIVE]
        
        # Remove inactive orders
        for order_id in inactive_orders:
            del self.orders[order_id]
        
        # Save orders
        self._save_orders()
        
        logger.info(f"Cleared {len(inactive_orders)} inactive orders")
        return len(inactive_orders)


# Create singleton instance
limit_order_manager = LimitOrderManager()
