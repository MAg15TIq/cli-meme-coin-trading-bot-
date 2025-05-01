"""
DCA (Dollar Cost Averaging) orders module for the Solana Memecoin Trading Bot.
Allows setting up automatic recurring buys or sells at regular intervals.
"""

import json
import logging
import threading
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.trading.jupiter_api import jupiter_api
from src.trading.position_manager import position_manager
from src.notifications.notification_service import notification_service, NotificationPriority
from src.wallet.wallet import wallet_manager

# Get logger for this module
logger = get_logger(__name__)


class DCAOrderType:
    """Enum-like class for DCA order types."""
    BUY = "buy"
    SELL = "sell"


class DCAOrderStatus:
    """Enum-like class for DCA order statuses."""
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    FAILED = "failed"


class DCAManager:
    """Manager for DCA (Dollar Cost Averaging) orders."""
    
    def __init__(self):
        """Initialize the DCA manager."""
        self.enabled = get_config_value("dca_enabled", False)
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.executions: Dict[str, List[Dict[str, Any]]] = {}
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.monitoring_interval = int(get_config_value("dca_interval_seconds", "60"))  # Default: 60 seconds
        
        # Load orders
        self._load_orders()
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable DCA orders.
        
        Args:
            enabled: Whether DCA orders should be enabled
        """
        self.enabled = enabled
        update_config("dca_enabled", enabled)
        logger.info(f"DCA orders {'enabled' if enabled else 'disabled'}")
        
        if enabled and not self.monitoring_thread:
            self.start_monitoring_thread()
        elif not enabled and self.monitoring_thread:
            self.stop_monitoring_thread()
    
    def _load_orders(self) -> None:
        """Load orders from config."""
        self.orders = get_config_value("dca_orders", {})
        self.executions = get_config_value("dca_executions", {})
        logger.info(f"Loaded {len(self.orders)} DCA orders")
    
    def _save_orders(self) -> None:
        """Save orders to config."""
        update_config("dca_orders", self.orders)
        update_config("dca_executions", self.executions)
    
    def start_monitoring_thread(self) -> None:
        """Start the monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("DCA monitoring thread already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("DCA monitoring thread started")
    
    def stop_monitoring_thread(self) -> None:
        """Stop the monitoring thread."""
        if not self.monitoring_thread:
            logger.warning("DCA monitoring thread not running")
            return
        
        self.stop_monitoring.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.monitoring_thread = None
        logger.info("DCA monitoring thread stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Check all active orders
                self._check_orders()
            except Exception as e:
                logger.error(f"Error in DCA monitoring loop: {e}")
            
            # Sleep for the monitoring interval
            self.stop_monitoring.wait(self.monitoring_interval)
    
    def _check_orders(self) -> None:
        """Check all active orders."""
        if not self.orders:
            return
        
        current_time = datetime.now()
        
        for order_id, order in list(self.orders.items()):
            if order["status"] != DCAOrderStatus.ACTIVE:
                continue
            
            # Check if order has completed all executions
            if order["max_executions"] and order["execution_count"] >= order["max_executions"]:
                # Mark order as completed
                order["status"] = DCAOrderStatus.COMPLETED
                order["completed_at"] = current_time.isoformat()
                self._save_orders()
                
                # Send notification
                notification_service.send_order_alert(
                    message=f"DCA order completed: {order['token_symbol']} {order['type']}",
                    priority=NotificationPriority.NORMAL.value
                )
                
                logger.info(f"DCA order completed: {order_id}")
                continue
            
            # Check if order has end date and it has passed
            if order["end_date"] and datetime.fromisoformat(order["end_date"]) <= current_time:
                # Mark order as completed
                order["status"] = DCAOrderStatus.COMPLETED
                order["completed_at"] = current_time.isoformat()
                self._save_orders()
                
                # Send notification
                notification_service.send_order_alert(
                    message=f"DCA order completed (end date reached): {order['token_symbol']} {order['type']}",
                    priority=NotificationPriority.NORMAL.value
                )
                
                logger.info(f"DCA order completed (end date reached): {order_id}")
                continue
            
            # Check if it's time for the next execution
            next_execution_time = self._get_next_execution_time(order)
            
            if next_execution_time and current_time >= next_execution_time:
                # Execute order
                self._execute_order(order_id, order)
    
    def _get_next_execution_time(self, order: Dict[str, Any]) -> Optional[datetime]:
        """
        Get the next execution time for an order.
        
        Args:
            order: Order data
            
        Returns:
            Next execution time, or None if no more executions
        """
        # Get interval in seconds
        interval_seconds = order["interval_seconds"]
        
        # Get last execution time
        last_execution_time = None
        if order["last_execution"]:
            last_execution_time = datetime.fromisoformat(order["last_execution"])
        else:
            # If no executions yet, use start date
            last_execution_time = datetime.fromisoformat(order["start_date"])
            
            # If start date is in the future, return it
            if last_execution_time > datetime.now():
                return last_execution_time
        
        # Calculate next execution time
        next_execution_time = last_execution_time + timedelta(seconds=interval_seconds)
        
        # Check if next execution is after end date
        if order["end_date"] and next_execution_time > datetime.fromisoformat(order["end_date"]):
            return None
        
        # Check if max executions reached
        if order["max_executions"] and order["execution_count"] >= order["max_executions"]:
            return None
        
        return next_execution_time
    
    def _execute_order(self, order_id: str, order: Dict[str, Any]) -> None:
        """
        Execute a DCA order.
        
        Args:
            order_id: ID of the order
            order: Order data
        """
        try:
            # Check if wallet is connected
            if not wallet_manager.current_keypair:
                logger.warning("Cannot execute DCA order: wallet not connected")
                return
            
            token_mint = order["token_mint"]
            token_symbol = order["token_symbol"]
            order_type = order["type"]
            amount_per_execution = order["amount_per_execution"]
            
            # Get current price
            current_price = jupiter_api.get_token_price(token_mint)
            
            if current_price is None:
                logger.warning(f"Cannot execute DCA order: failed to get price for {token_symbol}")
                return
            
            # Execute order
            if order_type == DCAOrderType.BUY:
                # Buy token
                result = position_manager.buy_token(
                    token_mint=token_mint,
                    amount_sol=amount_per_execution,
                    token_symbol=token_symbol
                )
                
                if result["success"]:
                    # Record execution
                    execution = {
                        "timestamp": datetime.now().isoformat(),
                        "price": current_price,
                        "amount_sol": amount_per_execution,
                        "transaction_signature": result["signature"],
                        "status": "success"
                    }
                    
                    # Update order
                    order["last_execution"] = datetime.now().isoformat()
                    order["execution_count"] += 1
                    
                    # Add execution to history
                    if order_id not in self.executions:
                        self.executions[order_id] = []
                    self.executions[order_id].append(execution)
                    
                    # Save orders
                    self._save_orders()
                    
                    # Send notification
                    notification_service.send_order_alert(
                        message=f"DCA buy executed: {token_symbol} at {current_price} ({order['execution_count']}/{order['max_executions'] or 'unlimited'})",
                        priority=NotificationPriority.NORMAL.value
                    )
                    
                    logger.info(f"DCA buy executed: {order_id} - {token_symbol} at {current_price}")
                else:
                    # Record failed execution
                    execution = {
                        "timestamp": datetime.now().isoformat(),
                        "price": current_price,
                        "amount_sol": amount_per_execution,
                        "error": result.get("error", "Unknown error"),
                        "status": "failed"
                    }
                    
                    # Add execution to history
                    if order_id not in self.executions:
                        self.executions[order_id] = []
                    self.executions[order_id].append(execution)
                    
                    # Save orders
                    self._save_orders()
                    
                    # Send notification
                    notification_service.send_order_alert(
                        message=f"DCA buy failed: {token_symbol} - {result.get('error', 'Unknown error')}",
                        priority=NotificationPriority.HIGH.value
                    )
                    
                    logger.error(f"DCA buy failed: {order_id} - {result.get('error', 'Unknown error')}")
            
            elif order_type == DCAOrderType.SELL:
                # Calculate percentage to sell
                percentage = amount_per_execution
                
                # Sell token
                result = position_manager.sell_token(
                    token_mint=token_mint,
                    percentage=percentage
                )
                
                if result["success"]:
                    # Record execution
                    execution = {
                        "timestamp": datetime.now().isoformat(),
                        "price": current_price,
                        "percentage": percentage,
                        "transaction_signature": result["signature"],
                        "status": "success"
                    }
                    
                    # Update order
                    order["last_execution"] = datetime.now().isoformat()
                    order["execution_count"] += 1
                    
                    # Add execution to history
                    if order_id not in self.executions:
                        self.executions[order_id] = []
                    self.executions[order_id].append(execution)
                    
                    # Save orders
                    self._save_orders()
                    
                    # Send notification
                    notification_service.send_order_alert(
                        message=f"DCA sell executed: {token_symbol} at {current_price} ({order['execution_count']}/{order['max_executions'] or 'unlimited'})",
                        priority=NotificationPriority.NORMAL.value
                    )
                    
                    logger.info(f"DCA sell executed: {order_id} - {token_symbol} at {current_price}")
                else:
                    # Record failed execution
                    execution = {
                        "timestamp": datetime.now().isoformat(),
                        "price": current_price,
                        "percentage": percentage,
                        "error": result.get("error", "Unknown error"),
                        "status": "failed"
                    }
                    
                    # Add execution to history
                    if order_id not in self.executions:
                        self.executions[order_id] = []
                    self.executions[order_id].append(execution)
                    
                    # Save orders
                    self._save_orders()
                    
                    # Send notification
                    notification_service.send_order_alert(
                        message=f"DCA sell failed: {token_symbol} - {result.get('error', 'Unknown error')}",
                        priority=NotificationPriority.HIGH.value
                    )
                    
                    logger.error(f"DCA sell failed: {order_id} - {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error executing DCA order {order_id}: {e}")
            
            # Record failed execution
            execution = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "failed"
            }
            
            # Add execution to history
            if order_id not in self.executions:
                self.executions[order_id] = []
            self.executions[order_id].append(execution)
            
            # Save orders
            self._save_orders()
            
            # Send notification
            notification_service.send_order_alert(
                message=f"Error executing DCA order: {order['token_symbol']} - {str(e)}",
                priority=NotificationPriority.HIGH.value
            )
    
    def create_dca_order(self, token_mint: str, token_symbol: str, order_type: str, 
                        amount_per_execution: float, interval_seconds: int,
                        start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None,
                        max_executions: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a new DCA order.
        
        Args:
            token_mint: Token mint address
            token_symbol: Token symbol
            order_type: Order type (buy, sell)
            amount_per_execution: Amount in SOL (for buy) or percentage (for sell) per execution
            interval_seconds: Interval between executions in seconds
            start_date: Optional start date (default: now)
            end_date: Optional end date
            max_executions: Optional maximum number of executions
            
        Returns:
            The created order
        """
        # Validate order type
        if order_type not in [DCAOrderType.BUY, DCAOrderType.SELL]:
            raise ValueError(f"Invalid order type: {order_type}")
        
        # Validate amount
        if amount_per_execution <= 0:
            raise ValueError("Amount per execution must be greater than 0")
        
        # Validate interval
        if interval_seconds <= 0:
            raise ValueError("Interval must be greater than 0")
        
        # Set default start date
        if start_date is None:
            start_date = datetime.now()
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Create order
        order = {
            "id": order_id,
            "token_mint": token_mint,
            "token_symbol": token_symbol,
            "type": order_type,
            "amount_per_execution": amount_per_execution,
            "interval_seconds": interval_seconds,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat() if end_date else None,
            "max_executions": max_executions,
            "execution_count": 0,
            "last_execution": None,
            "created_at": datetime.now().isoformat(),
            "status": DCAOrderStatus.ACTIVE
        }
        
        # Add to orders
        self.orders[order_id] = order
        
        # Initialize executions
        self.executions[order_id] = []
        
        # Save orders
        self._save_orders()
        
        logger.info(f"Created DCA order: {token_symbol} {order_type} {amount_per_execution} every {interval_seconds} seconds")
        return order
    
    def pause_order(self, order_id: str) -> bool:
        """
        Pause a DCA order.
        
        Args:
            order_id: ID of the order to pause
            
        Returns:
            True if order was paused, False otherwise
        """
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        # Get order
        order = self.orders[order_id]
        
        # Check if order is active
        if order["status"] != DCAOrderStatus.ACTIVE:
            logger.warning(f"Order is not active: {order_id}")
            return False
        
        # Mark order as paused
        order["status"] = DCAOrderStatus.PAUSED
        order["paused_at"] = datetime.now().isoformat()
        
        # Save orders
        self._save_orders()
        
        logger.info(f"Paused DCA order: {order_id}")
        return True
    
    def resume_order(self, order_id: str) -> bool:
        """
        Resume a paused DCA order.
        
        Args:
            order_id: ID of the order to resume
            
        Returns:
            True if order was resumed, False otherwise
        """
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        # Get order
        order = self.orders[order_id]
        
        # Check if order is paused
        if order["status"] != DCAOrderStatus.PAUSED:
            logger.warning(f"Order is not paused: {order_id}")
            return False
        
        # Mark order as active
        order["status"] = DCAOrderStatus.ACTIVE
        order["resumed_at"] = datetime.now().isoformat()
        
        # Save orders
        self._save_orders()
        
        logger.info(f"Resumed DCA order: {order_id}")
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a DCA order.
        
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
        
        # Check if order is active or paused
        if order["status"] not in [DCAOrderStatus.ACTIVE, DCAOrderStatus.PAUSED]:
            logger.warning(f"Order is not active or paused: {order_id}")
            return False
        
        # Mark order as cancelled
        order["status"] = DCAOrderStatus.CANCELLED
        order["cancelled_at"] = datetime.now().isoformat()
        
        # Save orders
        self._save_orders()
        
        logger.info(f"Cancelled DCA order: {order_id}")
        return True
    
    def get_orders(self, status: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get all DCA orders.
        
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
    
    def get_executions(self, order_id: str) -> List[Dict[str, Any]]:
        """
        Get executions for a specific order.
        
        Args:
            order_id: ID of the order
            
        Returns:
            List of executions
        """
        return self.executions.get(order_id, [])
    
    def clear_completed_orders(self) -> int:
        """
        Clear all completed, cancelled, and failed orders.
        
        Returns:
            Number of orders cleared
        """
        # Get completed orders
        completed_orders = [order_id for order_id, order in self.orders.items() 
                           if order["status"] in [DCAOrderStatus.COMPLETED, DCAOrderStatus.CANCELLED, DCAOrderStatus.FAILED]]
        
        # Remove completed orders
        for order_id in completed_orders:
            del self.orders[order_id]
            if order_id in self.executions:
                del self.executions[order_id]
        
        # Save orders
        self._save_orders()
        
        logger.info(f"Cleared {len(completed_orders)} completed orders")
        return len(completed_orders)


# Create singleton instance
dca_manager = DCAManager()
