"""
Test suite for enhanced features of the Solana Memecoin Trading Bot.
Tests backtesting engine, advanced orders, and portfolio analytics.
"""

import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from src.trading.backtesting_engine import (
    BacktestingEngine, OrderType, OrderStatus, BacktestOrder, 
    BacktestPosition, BacktestMetrics
)
from src.trading.advanced_orders import (
    AdvancedOrderManager, AdvancedOrderType, AdvancedOrder
)
from src.trading.portfolio_analytics import (
    PortfolioAnalytics, PortfolioSnapshot, PerformanceMetrics, RiskMetrics
)
from src.trading.jupiter_api import jupiter_api


class TestBacktestingEngine(unittest.TestCase):
    """Test cases for the backtesting engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = BacktestingEngine()
        self.engine.reset_state(1000.0)  # Start with 1000 SOL
        
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)  # Random walk
        
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates))
        })
        
        self.token_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        self.engine.load_price_data(self.token_mint, self.sample_data)
    
    def test_initialization(self):
        """Test backtesting engine initialization."""
        self.assertEqual(self.engine.balance, 1000.0)
        self.assertEqual(self.engine.initial_balance, 1000.0)
        self.assertEqual(len(self.engine.positions), 0)
        self.assertEqual(len(self.engine.orders), 0)
    
    def test_load_price_data(self):
        """Test loading price data."""
        self.assertIn(self.token_mint, self.engine.price_data)
        data = self.engine.price_data[self.token_mint]
        self.assertEqual(len(data), len(self.sample_data))
        self.assertTrue(data.index.is_monotonic_increasing)
    
    def test_place_order(self):
        """Test placing orders."""
        self.engine.current_time = self.sample_data.iloc[0]['timestamp']
        
        order_id = self.engine.place_order(
            OrderType.BUY, 
            self.token_mint, 
            10.0  # 10 SOL
        )
        
        self.assertIn(order_id, self.engine.orders)
        order = self.engine.orders[order_id]
        self.assertEqual(order.order_type, OrderType.BUY)
        self.assertEqual(order.amount, 10.0)
    
    def test_simple_strategy(self):
        """Test a simple buy-and-hold strategy."""
        def simple_strategy(engine, current_time, price_data):
            # Buy on first day, sell on last day
            if current_time == engine.price_data[self.token_mint].index[0]:
                engine.place_order(OrderType.BUY, self.token_mint, 100.0)
            elif current_time == engine.price_data[self.token_mint].index[-1]:
                if self.token_mint in engine.positions:
                    pos = engine.positions[self.token_mint]
                    engine.place_order(OrderType.SELL, self.token_mint, pos.amount)
        
        self.engine.set_strategy(simple_strategy)
        
        start_date = self.sample_data.iloc[0]['timestamp']
        end_date = self.sample_data.iloc[-1]['timestamp']
        
        metrics = self.engine.run_backtest(start_date, end_date, timedelta(days=1))
        
        self.assertIsInstance(metrics, BacktestMetrics)
        self.assertGreaterEqual(metrics.total_trades, 0)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        # Simulate some trades
        self.engine.trade_history = [
            {'timestamp': datetime.now(), 'pnl': 10.0, 'fees': 1.0},
            {'timestamp': datetime.now(), 'pnl': -5.0, 'fees': 0.5},
            {'timestamp': datetime.now(), 'pnl': 15.0, 'fees': 1.5}
        ]
        
        self.engine.balance_history = [
            {'timestamp': datetime.now() - timedelta(days=1), 'total_value': 1000.0},
            {'timestamp': datetime.now(), 'total_value': 1020.0}
        ]
        
        metrics = self.engine._calculate_metrics()
        
        self.assertEqual(metrics.total_trades, 3)
        self.assertEqual(metrics.winning_trades, 2)
        self.assertEqual(metrics.losing_trades, 1)
        self.assertAlmostEqual(metrics.win_rate, 2/3)


class TestAdvancedOrders(unittest.TestCase):
    """Test cases for advanced order types."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = AdvancedOrderManager()
        self.token_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        self.token_name = "USDC"
    
    def test_initialization(self):
        """Test advanced order manager initialization."""
        self.assertFalse(self.manager.enabled)
        self.assertEqual(len(self.manager.orders), 0)
        self.assertEqual(len(self.manager.active_orders), 0)
    
    def test_create_twap_order(self):
        """Test creating TWAP orders."""
        order_id = self.manager.create_twap_order(
            token_mint=self.token_mint,
            token_name=self.token_name,
            side="buy",
            total_amount=100.0,
            duration_minutes=60,
            interval_minutes=5
        )
        
        self.assertIn(order_id, self.manager.orders)
        self.assertIn(order_id, self.manager.active_orders)
        
        order = self.manager.orders[order_id]
        self.assertEqual(order.order_type, AdvancedOrderType.TWAP)
        self.assertEqual(order.total_amount, 100.0)
        self.assertEqual(order.duration_minutes, 60)
    
    def test_create_vwap_order(self):
        """Test creating VWAP orders."""
        order_id = self.manager.create_vwap_order(
            token_mint=self.token_mint,
            token_name=self.token_name,
            side="sell",
            total_amount=50.0,
            volume_participation_rate=0.1
        )
        
        self.assertIn(order_id, self.manager.orders)
        order = self.manager.orders[order_id]
        self.assertEqual(order.order_type, AdvancedOrderType.VWAP)
        self.assertEqual(order.volume_participation_rate, 0.1)
    
    def test_create_iceberg_order(self):
        """Test creating Iceberg orders."""
        order_id = self.manager.create_iceberg_order(
            token_mint=self.token_mint,
            token_name=self.token_name,
            side="buy",
            total_amount=1000.0,
            slice_size=100.0
        )
        
        self.assertIn(order_id, self.manager.orders)
        order = self.manager.orders[order_id]
        self.assertEqual(order.order_type, AdvancedOrderType.ICEBERG)
        self.assertEqual(order.slice_size, 100.0)
    
    def test_create_conditional_order(self):
        """Test creating conditional orders."""
        conditions = [
            {'type': 'price_above', 'value': 1.0},
            {'type': 'volume_above', 'value': 10000.0}
        ]
        
        order_id = self.manager.create_conditional_order(
            token_mint=self.token_mint,
            token_name=self.token_name,
            side="buy",
            total_amount=200.0,
            conditions=conditions
        )
        
        self.assertIn(order_id, self.manager.orders)
        order = self.manager.orders[order_id]
        self.assertEqual(order.order_type, AdvancedOrderType.CONDITIONAL)
        self.assertEqual(len(order.conditions), 2)
    
    def test_cancel_order(self):
        """Test cancelling orders."""
        order_id = self.manager.create_twap_order(
            token_mint=self.token_mint,
            token_name=self.token_name,
            side="buy",
            total_amount=100.0,
            duration_minutes=60
        )
        
        success = self.manager.cancel_order(order_id)
        self.assertTrue(success)
        
        order = self.manager.orders[order_id]
        self.assertEqual(order.status, OrderStatus.CANCELLED)
        self.assertNotIn(order_id, self.manager.active_orders)
    
    def test_get_order_status(self):
        """Test getting order status."""
        order_id = self.manager.create_twap_order(
            token_mint=self.token_mint,
            token_name=self.token_name,
            side="buy",
            total_amount=100.0,
            duration_minutes=60
        )
        
        status = self.manager.get_order_status(order_id)
        self.assertIsNotNone(status)
        self.assertEqual(status['id'], order_id)
        self.assertEqual(status['type'], 'twap')
        self.assertEqual(status['total_amount'], 100.0)


class TestPortfolioAnalytics(unittest.TestCase):
    """Test cases for portfolio analytics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analytics = PortfolioAnalytics()
    
    def test_initialization(self):
        """Test portfolio analytics initialization."""
        self.assertFalse(self.analytics.enabled)
        self.assertEqual(len(self.analytics.snapshots), 0)
        self.assertEqual(len(self.analytics.price_history), 0)
    
    @patch('src.trading.portfolio_analytics.wallet_manager')
    @patch('src.trading.portfolio_analytics.position_manager')
    @patch('src.trading.portfolio_analytics.jupiter_api')
    def test_capture_snapshot(self, mock_jupiter, mock_position_manager, mock_wallet_manager):
        """Test capturing portfolio snapshots."""
        # Mock wallet balance
        mock_wallet_manager.get_sol_balance.return_value = 100.0
        mock_wallet_manager.current_keypair = True
        
        # Mock positions
        mock_position = Mock()
        mock_position.token_name = "TEST"
        mock_position.amount = 1000.0
        mock_position.entry_price = 1.0
        
        mock_position_manager.positions = {
            "test_token": mock_position
        }
        
        # Mock price
        mock_jupiter.get_token_price.return_value = 1.1
        
        snapshot = self.analytics.capture_snapshot()
        
        self.assertIsInstance(snapshot, PortfolioSnapshot)
        self.assertEqual(snapshot.sol_balance, 100.0)
        self.assertGreater(snapshot.total_value_sol, 100.0)  # Should include token value
        self.assertIn("test_token", snapshot.positions)
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create sample snapshots
        base_time = datetime.now()
        for i in range(10):
            snapshot = PortfolioSnapshot(
                timestamp=base_time + timedelta(days=i),
                total_value_sol=1000 + i * 10,  # Increasing value
                sol_balance=500,
                token_value_sol=500 + i * 10,
                positions={}
            )
            self.analytics.snapshots.append(snapshot)
        
        metrics = self.analytics.calculate_performance_metrics(days=10)
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.total_return, 0)  # Should be positive
    
    def test_get_current_allocation(self):
        """Test current allocation calculation."""
        # Create a snapshot with mixed positions
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value_sol=1000.0,
            sol_balance=400.0,
            token_value_sol=600.0,
            positions={
                "token1": {
                    'token_name': 'TOKEN1',
                    'value_sol': 300.0
                },
                "token2": {
                    'token_name': 'TOKEN2',
                    'value_sol': 300.0
                }
            }
        )
        
        self.analytics.snapshots = [snapshot]
        
        with patch.object(self.analytics, 'capture_snapshot', return_value=snapshot):
            allocation = self.analytics.get_current_allocation()
        
        self.assertAlmostEqual(allocation['SOL'], 40.0)  # 400/1000 * 100
        self.assertAlmostEqual(allocation['TOKEN1'], 30.0)  # 300/1000 * 100
        self.assertAlmostEqual(allocation['TOKEN2'], 30.0)  # 300/1000 * 100


class TestJupiterAPIEnhancements(unittest.TestCase):
    """Test cases for Jupiter API enhancements."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api = jupiter_api
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        metrics = self.api.get_performance_metrics()
        
        self.assertIn('total_requests', metrics)
        self.assertIn('cache_hits', metrics)
        self.assertIn('cache_hit_rate', metrics)
        self.assertIn('average_request_time', metrics)
    
    @patch('src.trading.jupiter_api.requests.Session.get')
    def test_caching(self, mock_get):
        """Test price caching functionality."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "test_token": {"price": "1.5"}
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        token_mint = "test_token"
        
        # First call should hit the API
        price1 = self.api.get_token_price(token_mint)
        self.assertEqual(price1, 1.5)
        self.assertEqual(mock_get.call_count, 1)
        
        # Second call should use cache
        price2 = self.api.get_token_price(token_mint)
        self.assertEqual(price2, 1.5)
        self.assertEqual(mock_get.call_count, 1)  # Should not increase
    
    def test_async_methods_exist(self):
        """Test that async methods are available."""
        self.assertTrue(hasattr(self.api, 'get_token_price_async'))
        self.assertTrue(hasattr(self.api, 'get_multiple_token_prices_async'))
        self.assertTrue(hasattr(self.api, 'get_multiple_token_prices'))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
