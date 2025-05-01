"""
Tests for the gas optimization module.
"""

import unittest
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
import json

from src.solana.gas_optimizer import GasOptimizer, TransactionType, TransactionPriority


class TestGasOptimizer(unittest.TestCase):
    """Test cases for the GasOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for gas optimizer files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Mock config values
        self.config_values = {
            "fee_optimization_enabled": True,
            "fee_history_max_entries": "1000",
            "data_dir": str(Path(self.temp_dir.name)),
            "min_priority_fee": "1000",
            "congestion_check_interval": "60",
            "compute_unit_limit": "200000",
            "time_based_fee_adjustment": False
        }
        
        # Patch config.get_config_value to return our mock values
        self.get_config_patcher = patch('src.solana.gas_optimizer.get_config_value')
        self.mock_get_config = self.get_config_patcher.start()
        self.mock_get_config.side_effect = lambda key, default=None: self.config_values.get(key, default)
        
        # Patch solana_client
        self.solana_client_patcher = patch('src.solana.gas_optimizer.solana_client')
        self.mock_solana_client = self.solana_client_patcher.start()
        
        # Mock recent priority fees
        self.mock_solana_client.get_recent_priority_fee.return_value = {
            "10": 1000,
            "25": 1500,
            "50": 2000,
            "75": 3000,
            "90": 5000
        }
        
        # Mock recent block production
        self.mock_solana_client.get_recent_performance.return_value = {
            "total_slots": 100,
            "slots_per_second": 0.5,
            "num_transactions": 1000,
            "slot_skipped_rate": 0.05
        }
        
        # Patch threading.Thread to prevent background thread from starting
        self.thread_patcher = patch('src.solana.gas_optimizer.threading.Thread')
        self.mock_thread = self.thread_patcher.start()
        
        # Create a gas optimizer
        self.gas_optimizer = GasOptimizer()
        
        # Create sample fee history
        self.gas_optimizer.fee_history = [
            {
                "timestamp": (datetime.now() - timedelta(hours=1)).timestamp(),
                "fees": {
                    "10": 800,
                    "25": 1200,
                    "50": 1800,
                    "75": 2500,
                    "90": 4000
                },
                "congestion": 0.3
            },
            {
                "timestamp": datetime.now().timestamp(),
                "fees": {
                    "10": 1000,
                    "25": 1500,
                    "50": 2000,
                    "75": 3000,
                    "90": 5000
                },
                "congestion": 0.5
            }
        ]
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patchers
        self.get_config_patcher.stop()
        self.solana_client_patcher.stop()
        self.thread_patcher.stop()
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_init(self):
        """Test initialization of GasOptimizer."""
        # Verify enabled state
        self.assertTrue(self.gas_optimizer.enabled)
        
        # Verify fee percentiles
        self.assertEqual(self.gas_optimizer.fee_percentiles[TransactionPriority.LOW.value], 25)
        self.assertEqual(self.gas_optimizer.fee_percentiles[TransactionPriority.MEDIUM.value], 50)
        self.assertEqual(self.gas_optimizer.fee_percentiles[TransactionPriority.HIGH.value], 75)
        self.assertEqual(self.gas_optimizer.fee_percentiles[TransactionPriority.URGENT.value], 90)
        
        # Verify transaction type multipliers
        self.assertEqual(self.gas_optimizer.tx_type_multipliers[TransactionType.DEFAULT.value], 1.0)
        self.assertEqual(self.gas_optimizer.tx_type_multipliers[TransactionType.BUY.value], 1.0)
        self.assertEqual(self.gas_optimizer.tx_type_multipliers[TransactionType.SELL.value], 1.0)
        self.assertEqual(self.gas_optimizer.tx_type_multipliers[TransactionType.SNIPE.value], 1.5)
    
    def test_get_priority_fee(self):
        """Test getting priority fee."""
        # Test with default parameters
        fee = self.gas_optimizer.get_priority_fee()
        
        # Verify fee is calculated correctly
        self.assertEqual(fee, 2000)  # Medium priority (50th percentile)
        
        # Test with high priority
        fee = self.gas_optimizer.get_priority_fee(priority=TransactionPriority.HIGH)
        self.assertEqual(fee, 3000)  # High priority (75th percentile)
        
        # Test with snipe transaction type
        fee = self.gas_optimizer.get_priority_fee(tx_type=TransactionType.SNIPE)
        self.assertEqual(fee, 3000)  # Medium priority with 1.5x multiplier
        
        # Test with high priority and snipe transaction type
        fee = self.gas_optimizer.get_priority_fee(
            priority=TransactionPriority.HIGH,
            tx_type=TransactionType.SNIPE
        )
        self.assertEqual(fee, 4500)  # High priority with 1.5x multiplier
        
        # Test with disabled optimizer
        self.gas_optimizer.enabled = False
        fee = self.gas_optimizer.get_priority_fee()
        self.assertEqual(fee, 1000)  # Minimum fee
    
    def test_get_compute_limit(self):
        """Test getting compute unit limit."""
        # Test with default parameters
        limit = self.gas_optimizer.get_compute_limit()
        
        # Verify limit is calculated correctly
        self.assertEqual(limit, 200000)  # Default limit
        
        # Test with specific transaction type
        limit = self.gas_optimizer.get_compute_limit(tx_type=TransactionType.SNIPE)
        self.assertEqual(limit, 300000)  # Snipe has higher compute limit
        
        # Test with disabled optimizer
        self.gas_optimizer.enabled = False
        limit = self.gas_optimizer.get_compute_limit()
        self.assertEqual(limit, 200000)  # Default limit
    
    def test_get_fee_stats(self):
        """Test getting fee statistics."""
        # Get fee stats for the last 24 hours
        stats = self.gas_optimizer.get_fee_stats(hours=24)
        
        # Verify stats
        self.assertEqual(stats["time_period_hours"], 24)
        self.assertEqual(stats["num_entries"], 2)
        self.assertIn("percentiles", stats)
        self.assertIn("congestion", stats)
        
        # Verify percentiles
        self.assertIn("10", stats["percentiles"])
        self.assertIn("25", stats["percentiles"])
        self.assertIn("50", stats["percentiles"])
        self.assertIn("75", stats["percentiles"])
        self.assertIn("90", stats["percentiles"])
        
        # Verify congestion stats
        self.assertIn("average", stats["congestion"])
        self.assertIn("max", stats["congestion"])
        self.assertIn("min", stats["congestion"])
    
    def test_optimize_transaction_timing(self):
        """Test optimizing transaction timing."""
        # Test with normal urgency
        timing = self.gas_optimizer.optimize_transaction_timing(urgency="normal")
        
        # Verify timing recommendation
        self.assertIn("current_congestion", timing)
        self.assertIn("in_peak_hours", timing)
        self.assertIn("urgency", timing)
        self.assertIn("recommended_wait_minutes", timing)
        self.assertIn("optimal_execution_time", timing)
        self.assertIn("estimated_fee_savings_percent", timing)
        
        # Test with immediate urgency
        timing = self.gas_optimizer.optimize_transaction_timing(urgency="immediate")
        self.assertEqual(timing["recommended_wait_minutes"], 0)
        
        # Test with low urgency
        timing = self.gas_optimizer.optimize_transaction_timing(urgency="low")
        # Low urgency might recommend waiting
        self.assertGreaterEqual(timing["recommended_wait_minutes"], 0)
    
    def test_collect_recent_fees(self):
        """Test collecting recent fees."""
        # Call collect_recent_fees
        fee_data = self.gas_optimizer._collect_recent_fees()
        
        # Verify fee data
        self.assertIn("timestamp", fee_data)
        self.assertIn("fees", fee_data)
        self.assertIn("congestion", fee_data)
        
        # Verify fees
        self.assertEqual(fee_data["fees"], self.mock_solana_client.get_recent_priority_fee.return_value)
        
        # Verify congestion
        self.assertGreaterEqual(fee_data["congestion"], 0.0)
        self.assertLessEqual(fee_data["congestion"], 1.0)
    
    def test_get_network_congestion(self):
        """Test getting network congestion."""
        # Set up mock
        self.mock_solana_client.get_recent_performance.return_value = {
            "total_slots": 100,
            "slots_per_second": 0.5,  # 50% of max throughput
            "num_transactions": 1000,
            "slot_skipped_rate": 0.1  # 10% skip rate
        }
        
        # Get network congestion
        congestion = self.gas_optimizer._get_network_congestion()
        
        # Verify congestion
        self.assertGreaterEqual(congestion, 0.0)
        self.assertLessEqual(congestion, 1.0)
        
        # Higher skip rate should increase congestion
        self.mock_solana_client.get_recent_performance.return_value["slot_skipped_rate"] = 0.2
        congestion_high = self.gas_optimizer._get_network_congestion()
        self.assertGreater(congestion_high, congestion)
    
    def test_get_congestion_multiplier(self):
        """Test getting congestion multiplier."""
        # Test with low congestion
        self.gas_optimizer.last_congestion = 0.1
        multiplier = self.gas_optimizer._get_congestion_multiplier()
        self.assertLessEqual(multiplier, 1.0)
        
        # Test with medium congestion
        self.gas_optimizer.last_congestion = 0.5
        multiplier = self.gas_optimizer._get_congestion_multiplier()
        self.assertGreaterEqual(multiplier, 1.0)
        
        # Test with high congestion
        self.gas_optimizer.last_congestion = 0.9
        multiplier = self.gas_optimizer._get_congestion_multiplier()
        self.assertGreater(multiplier, 1.5)
    
    def test_refine_fee_parameters(self):
        """Test refining fee parameters based on transaction history."""
        # Create mock transaction history
        tx_history = [
            {
                "signature": "sig1",
                "success": True,
                "priority_fee": 2000,
                "tx_type": TransactionType.BUY.value,
                "congestion": 0.3,
                "time_of_day": 14  # 2pm UTC
            },
            {
                "signature": "sig2",
                "success": False,
                "priority_fee": 1500,
                "tx_type": TransactionType.SELL.value,
                "congestion": 0.7,
                "time_of_day": 15  # 3pm UTC
            },
            {
                "signature": "sig3",
                "success": True,
                "priority_fee": 3000,
                "tx_type": TransactionType.SNIPE.value,
                "congestion": 0.8,
                "time_of_day": 16  # 4pm UTC
            }
        ]
        
        # Store original multipliers
        original_multipliers = self.gas_optimizer.tx_type_multipliers.copy()
        
        # Refine parameters
        self.gas_optimizer.refine_fee_parameters(tx_history)
        
        # Verify multipliers were adjusted
        # Failed transactions should increase multipliers
        self.assertGreater(
            self.gas_optimizer.tx_type_multipliers[TransactionType.SELL.value],
            original_multipliers[TransactionType.SELL.value]
        )
        
        # Successful transactions in high congestion should maintain or slightly increase multipliers
        self.assertGreaterEqual(
            self.gas_optimizer.tx_type_multipliers[TransactionType.SNIPE.value],
            original_multipliers[TransactionType.SNIPE.value]
        )


if __name__ == '__main__':
    unittest.main()
