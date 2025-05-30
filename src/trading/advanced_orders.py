"""
Advanced order types for the Solana Memecoin Trading Bot.
Implements sophisticated order management including TWAP, VWAP, and Iceberg orders.
"""

import json
import logging
import time
import threading
import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.trading.jupiter_api import jupiter_api
from src.wallet.wallet import wallet_manager

# Get logger for this module
logger = get_logger(__name__)


class AdvancedOrderType(Enum):
    """Advanced order types."""
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"  # Large order split into smaller chunks
    CONDITIONAL = "conditional"  # Order based on multiple conditions


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class AdvancedOrder:
    """Represents an advanced order."""
    id: str
    order_type: AdvancedOrderType
    token_mint: str
    token_name: str
    side: str  # "buy" or "sell"
    total_amount: float
    filled_amount: float = 0.0
    remaining_amount: float = 0.0
    target_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    expires_at: Optional[datetime] = None

    # TWAP specific
    duration_minutes: Optional[int] = None
    interval_minutes: Optional[int] = None

    # VWAP specific
    volume_target: Optional[float] = None
    volume_participation_rate: Optional[float] = None

    # Iceberg specific
    slice_size: Optional[float] = None
    min_slice_size: Optional[float] = None

    # Conditional specific
    conditions: Optional[List[Dict[str, Any]]] = None

    # Execution tracking
    child_orders: List[str] = None
    execution_log: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.remaining_amount == 0.0:
            self.remaining_amount = self.total_amount
        if self.child_orders is None:
            self.child_orders = []
        if self.execution_log is None:
            self.execution_log = []


class AdvancedOrderManager:
    """Manager for advanced order types."""

    def __init__(self):
        """Initialize the advanced order manager."""
        self.enabled = get_config_value("advanced_orders_enabled", False)
        self.orders = {}  # order_id -> AdvancedOrder
        self.active_orders = set()

        # Execution thread
        self._execution_thread = None
        self._stop_execution = False
        self._execution_interval = 30  # seconds

        # Market data cache
        self._price_cache = {}
        self._volume_cache = {}
        self._cache_ttl = 60  # seconds

        logger.info("Advanced order manager initialized")

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable advanced orders.

        Args:
            enabled: Whether advanced orders should be enabled
        """
        self.enabled = enabled
        update_config("advanced_orders_enabled", enabled)

        if enabled and not self._execution_thread:
            self.start_execution_thread()
        elif not enabled and self._execution_thread:
            self.stop_execution_thread()

        logger.info(f"Advanced orders {'enabled' if enabled else 'disabled'}")

    def start_execution_thread(self) -> None:
        """Start the order execution thread."""
        if self._execution_thread and self._execution_thread.is_alive():
            return

        self._stop_execution = False
        self._execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self._execution_thread.start()
        logger.info("Advanced order execution thread started")

    def stop_execution_thread(self) -> None:
        """Stop the order execution thread."""
        self._stop_execution = True
        if self._execution_thread and self._execution_thread.is_alive():
            self._execution_thread.join(timeout=5)
        logger.info("Advanced order execution thread stopped")

    def create_twap_order(self, token_mint: str, token_name: str, side: str,
                         total_amount: float, duration_minutes: int,
                         interval_minutes: int = 5, target_price: Optional[float] = None) -> str:
        """
        Create a Time-Weighted Average Price (TWAP) order.

        Args:
            token_mint: Token mint address
            token_name: Token name/symbol
            side: "buy" or "sell"
            total_amount: Total amount to trade
            duration_minutes: Total duration for the order
            interval_minutes: Interval between executions
            target_price: Optional target price limit

        Returns:
            Order ID
        """
        order_id = f"twap_{int(time.time())}_{len(self.orders)}"

        order = AdvancedOrder(
            id=order_id,
            order_type=AdvancedOrderType.TWAP,
            token_mint=token_mint,
            token_name=token_name,
            side=side,
            total_amount=total_amount,
            target_price=target_price,
            duration_minutes=duration_minutes,
            interval_minutes=interval_minutes,
            expires_at=datetime.now() + timedelta(minutes=duration_minutes)
        )

        self.orders[order_id] = order
        self.active_orders.add(order_id)

        logger.info(f"Created TWAP order {order_id}: {side} {total_amount} {token_name} "
                   f"over {duration_minutes} minutes")

        return order_id

    def create_vwap_order(self, token_mint: str, token_name: str, side: str,
                         total_amount: float, volume_participation_rate: float = 0.1,
                         target_price: Optional[float] = None) -> str:
        """
        Create a Volume-Weighted Average Price (VWAP) order.

        Args:
            token_mint: Token mint address
            token_name: Token name/symbol
            side: "buy" or "sell"
            total_amount: Total amount to trade
            volume_participation_rate: Percentage of market volume to participate (0.0-1.0)
            target_price: Optional target price limit

        Returns:
            Order ID
        """
        order_id = f"vwap_{int(time.time())}_{len(self.orders)}"

        order = AdvancedOrder(
            id=order_id,
            order_type=AdvancedOrderType.VWAP,
            token_mint=token_mint,
            token_name=token_name,
            side=side,
            total_amount=total_amount,
            target_price=target_price,
            volume_participation_rate=volume_participation_rate
        )

        self.orders[order_id] = order
        self.active_orders.add(order_id)

        logger.info(f"Created VWAP order {order_id}: {side} {total_amount} {token_name} "
                   f"with {volume_participation_rate:.1%} participation rate")

        return order_id

    def create_iceberg_order(self, token_mint: str, token_name: str, side: str,
                           total_amount: float, slice_size: float,
                           min_slice_size: Optional[float] = None,
                           target_price: Optional[float] = None) -> str:
        """
        Create an Iceberg order (large order split into smaller chunks).

        Args:
            token_mint: Token mint address
            token_name: Token name/symbol
            side: "buy" or "sell"
            total_amount: Total amount to trade
            slice_size: Size of each slice
            min_slice_size: Minimum slice size for the last slice
            target_price: Optional target price limit

        Returns:
            Order ID
        """
        order_id = f"iceberg_{int(time.time())}_{len(self.orders)}"

        if min_slice_size is None:
            min_slice_size = slice_size * 0.1  # 10% of slice size

        order = AdvancedOrder(
            id=order_id,
            order_type=AdvancedOrderType.ICEBERG,
            token_mint=token_mint,
            token_name=token_name,
            side=side,
            total_amount=total_amount,
            target_price=target_price,
            slice_size=slice_size,
            min_slice_size=min_slice_size
        )

        self.orders[order_id] = order
        self.active_orders.add(order_id)

        logger.info(f"Created Iceberg order {order_id}: {side} {total_amount} {token_name} "
                   f"in slices of {slice_size}")

        return order_id

    def create_conditional_order(self, token_mint: str, token_name: str, side: str,
                               total_amount: float, conditions: List[Dict[str, Any]],
                               target_price: Optional[float] = None) -> str:
        """
        Create a conditional order based on multiple criteria.

        Args:
            token_mint: Token mint address
            token_name: Token name/symbol
            side: "buy" or "sell"
            total_amount: Total amount to trade
            conditions: List of conditions to check
            target_price: Optional target price limit

        Returns:
            Order ID
        """
        order_id = f"conditional_{int(time.time())}_{len(self.orders)}"

        order = AdvancedOrder(
            id=order_id,
            order_type=AdvancedOrderType.CONDITIONAL,
            token_mint=token_mint,
            token_name=token_name,
            side=side,
            total_amount=total_amount,
            target_price=target_price,
            conditions=conditions
        )

        self.orders[order_id] = order
        self.active_orders.add(order_id)

        logger.info(f"Created conditional order {order_id}: {side} {total_amount} {token_name} "
                   f"with {len(conditions)} conditions")

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an advanced order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if order was cancelled successfully
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False

        order = self.orders[order_id]

        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
            logger.warning(f"Order {order_id} cannot be cancelled (status: {order.status.value})")
            return False

        order.status = OrderStatus.CANCELLED
        self.active_orders.discard(order_id)

        # Cancel any pending child orders
        # This would integrate with the regular order system

        logger.info(f"Cancelled order {order_id}")
        return True

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an advanced order.

        Args:
            order_id: Order ID

        Returns:
            Order status information or None if not found
        """
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]

        return {
            'id': order.id,
            'type': order.order_type.value,
            'token': order.token_name,
            'side': order.side,
            'total_amount': order.total_amount,
            'filled_amount': order.filled_amount,
            'remaining_amount': order.remaining_amount,
            'fill_percentage': (order.filled_amount / order.total_amount) * 100,
            'status': order.status.value,
            'created_at': order.created_at.isoformat(),
            'expires_at': order.expires_at.isoformat() if order.expires_at else None,
            'child_orders': len(order.child_orders),
            'execution_events': len(order.execution_log)
        }

    def list_active_orders(self) -> List[Dict[str, Any]]:
        """
        List all active advanced orders.

        Returns:
            List of active order information
        """
        active_orders = []

        for order_id in self.active_orders:
            if order_id in self.orders:
                order_info = self.get_order_status(order_id)
                if order_info:
                    active_orders.append(order_info)

        return active_orders

    def _execution_loop(self) -> None:
        """Main execution loop for processing advanced orders."""
        logger.info("Advanced order execution loop started")

        while not self._stop_execution:
            try:
                # Process all active orders
                for order_id in list(self.active_orders):
                    if order_id in self.orders:
                        self._process_order(order_id)

                # Clean up completed orders
                self._cleanup_completed_orders()

            except Exception as e:
                logger.error(f"Error in advanced order execution loop: {e}")

            # Wait before next iteration
            time.sleep(self._execution_interval)

        logger.info("Advanced order execution loop stopped")

    def _process_order(self, order_id: str) -> None:
        """
        Process a single advanced order.

        Args:
            order_id: Order ID to process
        """
        order = self.orders.get(order_id)
        if not order:
            return

        try:
            # Check if order has expired
            if order.expires_at and datetime.now() > order.expires_at:
                order.status = OrderStatus.EXPIRED
                self.active_orders.discard(order_id)
                logger.info(f"Order {order_id} expired")
                return

            # Process based on order type
            if order.order_type == AdvancedOrderType.TWAP:
                self._process_twap_order(order)
            elif order.order_type == AdvancedOrderType.VWAP:
                self._process_vwap_order(order)
            elif order.order_type == AdvancedOrderType.ICEBERG:
                self._process_iceberg_order(order)
            elif order.order_type == AdvancedOrderType.CONDITIONAL:
                self._process_conditional_order(order)

        except Exception as e:
            logger.error(f"Error processing order {order_id}: {e}")
            order.execution_log.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'error',
                'message': str(e)
            })

    def _process_twap_order(self, order: AdvancedOrder) -> None:
        """Process a TWAP order."""
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.ACTIVE

        # Calculate how much time has passed and how much we should have executed
        elapsed_minutes = (datetime.now() - order.created_at).total_seconds() / 60
        progress_ratio = min(elapsed_minutes / order.duration_minutes, 1.0)
        target_filled = order.total_amount * progress_ratio

        # Check if we need to execute more
        if order.filled_amount < target_filled:
            # Calculate slice size for this interval
            remaining_time_ratio = (order.duration_minutes - elapsed_minutes) / order.duration_minutes
            if remaining_time_ratio > 0:
                slice_size = min(
                    target_filled - order.filled_amount,
                    order.remaining_amount / max(remaining_time_ratio * (order.duration_minutes / order.interval_minutes), 1)
                )

                if slice_size > 0:
                    self._execute_slice(order, slice_size)

    def _process_vwap_order(self, order: AdvancedOrder) -> None:
        """Process a VWAP order."""
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.ACTIVE

        # Get current market volume (simplified - would need real volume data)
        current_volume = self._get_market_volume(order.token_mint)

        if current_volume > 0:
            # Calculate slice size based on volume participation rate
            slice_size = min(
                current_volume * order.volume_participation_rate,
                order.remaining_amount
            )

            if slice_size > 0:
                self._execute_slice(order, slice_size)

    def _process_iceberg_order(self, order: AdvancedOrder) -> None:
        """Process an Iceberg order."""
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.ACTIVE

        # Check if we need to execute the next slice
        if order.remaining_amount > 0:
            # Calculate slice size
            slice_size = min(order.slice_size, order.remaining_amount)

            # Don't execute if slice would be too small
            if slice_size < order.min_slice_size and order.remaining_amount > order.min_slice_size:
                return

            self._execute_slice(order, slice_size)

    def _process_conditional_order(self, order: AdvancedOrder) -> None:
        """Process a conditional order."""
        if order.status == OrderStatus.PENDING:
            # Check if all conditions are met
            if self._check_conditions(order.conditions, order.token_mint):
                order.status = OrderStatus.ACTIVE
                # Execute the entire order at once
                self._execute_slice(order, order.remaining_amount)

    def _execute_slice(self, order: AdvancedOrder, slice_size: float) -> None:
        """
        Execute a slice of an advanced order.

        Args:
            order: The advanced order
            slice_size: Size of the slice to execute
        """
        try:
            # Check wallet connection
            if not wallet_manager.current_keypair:
                logger.warning(f"Cannot execute slice for order {order.id}: wallet not connected")
                return

            # Get current price
            current_price = jupiter_api.get_token_price(order.token_mint)

            # Check price limit if specified
            if order.target_price:
                if order.side == "buy" and current_price > order.target_price:
                    logger.debug(f"Order {order.id}: current price {current_price} above target {order.target_price}")
                    return
                elif order.side == "sell" and current_price < order.target_price:
                    logger.debug(f"Order {order.id}: current price {current_price} below target {order.target_price}")
                    return

            # Execute the trade
            if order.side == "buy":
                # Convert slice size (in SOL) to buy amount
                tx_signature = jupiter_api.execute_buy(
                    token_mint=order.token_mint,
                    amount_sol=slice_size,
                    wallet=wallet_manager.current_keypair
                )
            else:
                # For sell orders, slice_size is in tokens
                # Get token decimals (simplified - would need real token info)
                decimals = 9  # Default for most tokens
                tx_signature = jupiter_api.execute_sell(
                    token_mint=order.token_mint,
                    amount_token=slice_size,
                    decimals=decimals,
                    wallet=wallet_manager.current_keypair
                )

            if tx_signature:
                # Update order state
                order.filled_amount += slice_size
                order.remaining_amount -= slice_size

                # Log execution
                order.execution_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'event': 'slice_executed',
                    'slice_size': slice_size,
                    'price': current_price,
                    'tx_signature': tx_signature
                })

                logger.info(f"Executed slice for order {order.id}: {slice_size} at {current_price}")

                # Check if order is complete
                if order.remaining_amount <= 0:
                    order.status = OrderStatus.FILLED
                    self.active_orders.discard(order.id)
                    logger.info(f"Order {order.id} completed")
                elif order.filled_amount > 0:
                    order.status = OrderStatus.PARTIALLY_FILLED

        except Exception as e:
            logger.error(f"Error executing slice for order {order.id}: {e}")
            order.execution_log.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'execution_error',
                'message': str(e)
            })

    def _check_conditions(self, conditions: List[Dict[str, Any]], token_mint: str) -> bool:
        """
        Check if all conditions for a conditional order are met.

        Args:
            conditions: List of conditions to check
            token_mint: Token mint address

        Returns:
            True if all conditions are met
        """
        try:
            current_price = jupiter_api.get_token_price(token_mint)

            for condition in conditions:
                condition_type = condition.get('type')

                if condition_type == 'price_above':
                    if current_price <= condition.get('value', 0):
                        return False
                elif condition_type == 'price_below':
                    if current_price >= condition.get('value', float('inf')):
                        return False
                elif condition_type == 'price_change':
                    # Would need historical price data
                    pass
                elif condition_type == 'volume_above':
                    volume = self._get_market_volume(token_mint)
                    if volume <= condition.get('value', 0):
                        return False
                # Add more condition types as needed

            return True

        except Exception as e:
            logger.error(f"Error checking conditions: {e}")
            return False

    def _get_market_volume(self, token_mint: str) -> float:
        """
        Get market volume for a token (simplified implementation).

        Args:
            token_mint: Token mint address

        Returns:
            Market volume
        """
        # This is a simplified implementation
        # In reality, you would fetch this from a market data provider
        cache_key = f"volume_{token_mint}"
        current_time = time.time()

        if cache_key in self._volume_cache:
            volume, timestamp = self._volume_cache[cache_key]
            if current_time - timestamp < self._cache_ttl:
                return volume

        # Simulate volume data (replace with real implementation)
        import random
        volume = random.uniform(1000, 10000)
        self._volume_cache[cache_key] = (volume, current_time)

        return volume

    def _cleanup_completed_orders(self) -> None:
        """Clean up completed orders from active set."""
        completed_orders = []

        for order_id in list(self.active_orders):
            order = self.orders.get(order_id)
            if order and order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.FAILED]:
                completed_orders.append(order_id)

        for order_id in completed_orders:
            self.active_orders.discard(order_id)


# Create a singleton instance
advanced_order_manager = AdvancedOrderManager()
