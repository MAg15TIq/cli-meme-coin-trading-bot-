"""
Rapid Trading Executor for high-frequency trading scenarios.
Optimized for minimal latency and maximum throughput.
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from queue import Queue, PriorityQueue
import json

from config import get_config_value
from src.utils.logging_utils import get_logger
from src.trading.jupiter_api import jupiter_api
from src.solana.solana_interact import solana_client
from src.solana.gas_optimizer import gas_optimizer, TransactionType, TransactionPriority

logger = get_logger(__name__)


@dataclass
class TradeOrder:
    """Represents a trade order with priority."""
    priority: int  # Lower number = higher priority
    order_id: str
    token_mint: str
    action: str  # 'buy' or 'sell'
    amount: float
    price_limit: Optional[float] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def __lt__(self, other):
        return self.priority < other.priority


class RapidExecutor:
    """High-performance trading executor for rapid order execution."""
    
    def __init__(self, max_workers: int = 5):
        """
        Initialize the rapid executor.
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.order_queue = PriorityQueue()
        self.active_orders = {}
        self.completed_orders = {}
        
        # Performance tracking
        self.execution_times = []
        self.success_rate = 0.0
        self.total_orders = 0
        self.successful_orders = 0
        
        # Worker thread control
        self.running = False
        self.worker_thread = None
        
        # Pre-computed transaction parameters
        self._precomputed_params = {}
        self._param_cache_ttl = 30  # 30 seconds
        
        logger.info(f"Rapid executor initialized with {max_workers} workers")
    
    def start(self):
        """Start the rapid executor worker thread."""
        if self.running:
            logger.warning("Rapid executor is already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Rapid executor started")
    
    def stop(self):
        """Stop the rapid executor."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("Rapid executor stopped")
    
    def submit_order(self, order: TradeOrder) -> str:
        """
        Submit a trade order for rapid execution.
        
        Args:
            order: The trade order to execute
            
        Returns:
            Order ID for tracking
        """
        self.order_queue.put(order)
        self.active_orders[order.order_id] = order
        logger.info(f"Order {order.order_id} submitted for {order.action} {order.amount} {order.token_mint}")
        return order.order_id
    
    def submit_rapid_buy(self, token_mint: str, amount_sol: float, 
                        priority: int = 1, price_limit: Optional[float] = None) -> str:
        """
        Submit a rapid buy order.
        
        Args:
            token_mint: Token to buy
            amount_sol: Amount in SOL to spend
            priority: Order priority (1=highest, 5=lowest)
            price_limit: Maximum price to pay
            
        Returns:
            Order ID
        """
        order_id = f"buy_{int(time.time() * 1000)}_{len(self.active_orders)}"
        order = TradeOrder(
            priority=priority,
            order_id=order_id,
            token_mint=token_mint,
            action="buy",
            amount=amount_sol,
            price_limit=price_limit
        )
        return self.submit_order(order)
    
    def submit_rapid_sell(self, token_mint: str, amount_tokens: float,
                         priority: int = 1, price_limit: Optional[float] = None) -> str:
        """
        Submit a rapid sell order.
        
        Args:
            token_mint: Token to sell
            amount_tokens: Amount of tokens to sell
            priority: Order priority (1=highest, 5=lowest)
            price_limit: Minimum price to accept
            
        Returns:
            Order ID
        """
        order_id = f"sell_{int(time.time() * 1000)}_{len(self.active_orders)}"
        order = TradeOrder(
            priority=priority,
            order_id=order_id,
            token_mint=token_mint,
            action="sell",
            amount=amount_tokens,
            price_limit=price_limit
        )
        return self.submit_order(order)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if cancelled, False if not found or already executing
        """
        if order_id in self.active_orders:
            # Remove from active orders
            del self.active_orders[order_id]
            logger.info(f"Order {order_id} cancelled")
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status information
        """
        if order_id in self.active_orders:
            return {"status": "pending", "order": self.active_orders[order_id]}
        elif order_id in self.completed_orders:
            return {"status": "completed", "result": self.completed_orders[order_id]}
        else:
            return {"status": "not_found"}
    
    def _worker_loop(self):
        """Main worker loop for processing orders."""
        logger.info("Rapid executor worker loop started")
        
        while self.running:
            try:
                # Get next order with timeout
                try:
                    order = self.order_queue.get(timeout=1.0)
                except:
                    continue
                
                # Execute order
                start_time = time.time()
                result = self._execute_order(order)
                execution_time = time.time() - start_time
                
                # Update statistics
                self.execution_times.append(execution_time)
                self.total_orders += 1
                if result.get("success", False):
                    self.successful_orders += 1
                
                self.success_rate = self.successful_orders / self.total_orders
                
                # Store result
                self.completed_orders[order.order_id] = result
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
                
                # Keep only last 100 execution times
                if len(self.execution_times) > 100:
                    self.execution_times = self.execution_times[-100:]
                
                logger.info(f"Order {order.order_id} executed in {execution_time:.3f}s, "
                           f"success: {result.get('success', False)}")
                
            except Exception as e:
                logger.error(f"Error in rapid executor worker loop: {e}")
    
    def _execute_order(self, order: TradeOrder) -> Dict[str, Any]:
        """
        Execute a single order.
        
        Args:
            order: Order to execute
            
        Returns:
            Execution result
        """
        try:
            # Get pre-computed parameters or compute them
            params = self._get_transaction_params(order)
            
            if order.action == "buy":
                return self._execute_buy_order(order, params)
            elif order.action == "sell":
                return self._execute_sell_order(order, params)
            else:
                return {"success": False, "error": f"Unknown action: {order.action}"}
                
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_buy_order(self, order: TradeOrder, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a buy order."""
        try:
            # Check price limit if specified
            if order.price_limit:
                current_price = jupiter_api.get_token_price(order.token_mint)
                if current_price > order.price_limit:
                    return {"success": False, "error": "Price above limit"}
            
            # Execute the buy
            tx_signature = jupiter_api.execute_buy(
                token_mint=order.token_mint,
                amount_sol=order.amount,
                wallet=params["wallet"],
                priority_fee=params["priority_fee"]
            )
            
            return {
                "success": True,
                "tx_signature": tx_signature,
                "order_id": order.order_id,
                "execution_time": time.time() - order.timestamp
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_sell_order(self, order: TradeOrder, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a sell order."""
        try:
            # Check price limit if specified
            if order.price_limit:
                current_price = jupiter_api.get_token_price(order.token_mint)
                if current_price < order.price_limit:
                    return {"success": False, "error": "Price below limit"}
            
            # For sell orders, we need token decimals
            # This should be cached or pre-computed for better performance
            decimals = params.get("decimals", 9)  # Default to 9 decimals
            
            # Execute the sell
            tx_signature = jupiter_api.execute_sell(
                token_mint=order.token_mint,
                amount_token=order.amount,
                decimals=decimals,
                wallet=params["wallet"],
                priority_fee=params["priority_fee"]
            )
            
            return {
                "success": True,
                "tx_signature": tx_signature,
                "order_id": order.order_id,
                "execution_time": time.time() - order.timestamp
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_transaction_params(self, order: TradeOrder) -> Dict[str, Any]:
        """Get or compute transaction parameters."""
        # This would need to be implemented with actual wallet management
        # For now, return placeholder parameters
        return {
            "wallet": None,  # Would need actual wallet instance
            "priority_fee": gas_optimizer.get_priority_fee(
                priority=TransactionPriority.HIGH,
                tx_type=TransactionType.BUY if order.action == "buy" else TransactionType.SELL
            ),
            "decimals": 9  # Default decimals
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_execution_time = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
        
        return {
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "success_rate": self.success_rate,
            "average_execution_time": avg_execution_time,
            "pending_orders": len(self.active_orders),
            "completed_orders": len(self.completed_orders)
        }


# Create singleton instance
rapid_executor = RapidExecutor()
