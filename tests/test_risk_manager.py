"""
Tests for the risk management module.
"""

import unittest
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime
import json

from src.trading.risk_manager import RiskManager, RiskProfile
from src.trading.position_manager import RiskLevel


class TestRiskManager(unittest.TestCase):
    """Test cases for the RiskManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for risk manager files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Mock config values
        self.config_values = {
            "risk_management_enabled": True,
            "risk_profile": "moderate",
            "conservative_max_allocation_percent": "20.0",
            "conservative_max_position_percent": "2.0",
            "conservative_stop_loss_percent": "5.0",
            "conservative_max_drawdown_percent": "10.0",
            "moderate_max_allocation_percent": "40.0",
            "moderate_max_position_percent": "5.0",
            "moderate_stop_loss_percent": "10.0",
            "moderate_max_drawdown_percent": "20.0",
            "aggressive_max_allocation_percent": "60.0",
            "aggressive_max_position_percent": "10.0",
            "aggressive_stop_loss_percent": "15.0",
            "aggressive_max_drawdown_percent": "30.0",
            "max_low_risk_allocation_percent": "60.0",
            "max_medium_risk_allocation_percent": "30.0",
            "max_high_risk_allocation_percent": "10.0",
            "max_very_high_risk_allocation_percent": "5.0",
            "data_dir": str(Path(self.temp_dir.name))
        }
        
        # Patch config.get_config_value to return our mock values
        self.get_config_patcher = patch('src.trading.risk_manager.get_config_value')
        self.mock_get_config = self.get_config_patcher.start()
        self.mock_get_config.side_effect = lambda key, default=None: self.config_values.get(key, default)
        
        # Patch token_analytics
        self.token_analytics_patcher = patch('src.trading.risk_manager.token_analytics')
        self.mock_token_analytics = self.token_analytics_patcher.start()
        self.mock_token_analytics.get_risk_level.return_value = RiskLevel.MEDIUM.value
        
        # Patch wallet_manager
        self.wallet_manager_patcher = patch('src.trading.risk_manager.wallet_manager')
        self.mock_wallet_manager = self.wallet_manager_patcher.start()
        self.mock_wallet_manager.get_sol_balance.return_value = 10.0
        
        # Create a risk manager
        self.risk_manager = RiskManager()
        
        # Test token mint
        self.test_token_mint = "TestMint123"
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patchers
        self.get_config_patcher.stop()
        self.token_analytics_patcher.stop()
        self.wallet_manager_patcher.stop()
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_init(self):
        """Test initialization of RiskManager."""
        # Verify risk profiles were created
        self.assertIn("conservative", self.risk_manager.risk_profiles)
        self.assertIn("moderate", self.risk_manager.risk_profiles)
        self.assertIn("aggressive", self.risk_manager.risk_profiles)
        
        # Verify current profile
        self.assertEqual(self.risk_manager.current_profile_name, "moderate")
        
        # Verify risk allocation limits
        self.assertEqual(self.risk_manager.risk_allocation_limits[RiskLevel.LOW.value], 60.0)
        self.assertEqual(self.risk_manager.risk_allocation_limits[RiskLevel.MEDIUM.value], 30.0)
        self.assertEqual(self.risk_manager.risk_allocation_limits[RiskLevel.HIGH.value], 10.0)
        self.assertEqual(self.risk_manager.risk_allocation_limits[RiskLevel.VERY_HIGH.value], 5.0)
    
    def test_get_current_profile(self):
        """Test getting the current risk profile."""
        profile = self.risk_manager.get_current_profile()
        self.assertEqual(profile.name, "moderate")
        self.assertEqual(profile.max_allocation_percent, 40.0)
        self.assertEqual(profile.max_position_percent, 5.0)
        self.assertEqual(profile.stop_loss_percent, 10.0)
        self.assertEqual(profile.max_drawdown_percent, 20.0)
    
    def test_set_profile(self):
        """Test setting the risk profile."""
        # Set to conservative
        self.risk_manager.set_profile("conservative")
        self.assertEqual(self.risk_manager.current_profile_name, "conservative")
        profile = self.risk_manager.get_current_profile()
        self.assertEqual(profile.name, "conservative")
        
        # Set to aggressive
        self.risk_manager.set_profile("aggressive")
        self.assertEqual(self.risk_manager.current_profile_name, "aggressive")
        profile = self.risk_manager.get_current_profile()
        self.assertEqual(profile.name, "aggressive")
        
        # Set to invalid profile (should default to moderate)
        self.risk_manager.set_profile("invalid")
        self.assertEqual(self.risk_manager.current_profile_name, "moderate")
    
    def test_get_token_risk_level(self):
        """Test getting token risk level."""
        # Set up mock
        self.mock_token_analytics.get_risk_level.return_value = RiskLevel.HIGH.value
        
        # Get risk level
        risk_level = self.risk_manager.get_token_risk_level(self.test_token_mint)
        
        # Verify risk level
        self.assertEqual(risk_level, RiskLevel.HIGH)
        
        # Verify token_analytics was called
        self.mock_token_analytics.get_risk_level.assert_called_once_with(self.test_token_mint)
        
        # Test cache
        self.mock_token_analytics.get_risk_level.reset_mock()
        risk_level = self.risk_manager.get_token_risk_level(self.test_token_mint)
        self.assertEqual(risk_level, RiskLevel.HIGH)
        self.mock_token_analytics.get_risk_level.assert_not_called()
    
    def test_calculate_position_size(self):
        """Test calculating position size."""
        # Set up mocks
        self.mock_token_analytics.get_risk_level.return_value = RiskLevel.MEDIUM.value
        
        # Calculate position size
        result = self.risk_manager.calculate_position_size(self.test_token_mint, max_amount_sol=1.0)
        
        # Verify result
        self.assertIn("position_size_sol", result)
        self.assertIn("risk_level", result)
        self.assertIn("stop_loss_percentage", result)
        
        # Verify position size is within limits
        self.assertLessEqual(result["position_size_sol"], 1.0)
        
        # Test with risk management disabled
        self.risk_manager.enabled = False
        result = self.risk_manager.calculate_position_size(self.test_token_mint)
        self.assertEqual(result["position_size_sol"], 0.1)
    
    def test_update_portfolio_metrics(self):
        """Test updating portfolio metrics."""
        # Mock position manager
        with patch('src.trading.risk_manager.position_manager') as mock_position_manager:
            # Create mock positions
            mock_positions = {
                "token1": MagicMock(token_mint="token1", current_value_sol=1.0),
                "token2": MagicMock(token_mint="token2", current_value_sol=2.0),
                "token3": MagicMock(token_mint="token3", current_value_sol=3.0)
            }
            
            # Set up mock risk levels
            self.mock_token_analytics.get_risk_level.side_effect = lambda token_mint: {
                "token1": RiskLevel.LOW.value,
                "token2": RiskLevel.MEDIUM.value,
                "token3": RiskLevel.HIGH.value
            }.get(token_mint, RiskLevel.MEDIUM.value)
            
            # Set up mock position manager
            mock_position_manager.positions = mock_positions
            mock_position_manager.get_portfolio_value.return_value = 6.0
            mock_position_manager.get_portfolio_drawdown.return_value = 10.0
            
            # Update portfolio metrics
            self.risk_manager.update_portfolio_metrics()
            
            # Verify portfolio value
            self.assertEqual(self.risk_manager.portfolio_value_sol, 6.0)
            
            # Verify portfolio drawdown
            self.assertEqual(self.risk_manager.portfolio_drawdown, 10.0)
            
            # Verify portfolio allocation
            self.assertEqual(self.risk_manager.portfolio_allocation[RiskLevel.LOW.value], 1.0)
            self.assertEqual(self.risk_manager.portfolio_allocation[RiskLevel.MEDIUM.value], 2.0)
            self.assertEqual(self.risk_manager.portfolio_allocation[RiskLevel.HIGH.value], 3.0)
            self.assertEqual(self.risk_manager.portfolio_allocation[RiskLevel.VERY_HIGH.value], 0.0)
    
    def test_check_portfolio_risk(self):
        """Test checking portfolio risk."""
        # Mock update_portfolio_metrics
        with patch.object(self.risk_manager, 'update_portfolio_metrics'):
            # Set portfolio metrics
            self.risk_manager.portfolio_value_sol = 10.0
            self.risk_manager.portfolio_drawdown = 15.0
            self.risk_manager.portfolio_allocation = {
                RiskLevel.LOW.value: 5.0,
                RiskLevel.MEDIUM.value: 3.0,
                RiskLevel.HIGH.value: 1.5,
                RiskLevel.VERY_HIGH.value: 0.5
            }
            
            # Check portfolio risk
            result = self.risk_manager.check_portfolio_risk()
            
            # Verify result
            self.assertTrue(result["enabled"])
            self.assertEqual(result["portfolio_value_sol"], 10.0)
            self.assertEqual(result["portfolio_drawdown"], 15.0)
            self.assertEqual(result["max_drawdown_percent"], 20.0)
            self.assertFalse(result["drawdown_exceeded"])
            
            # Verify risk allocation
            for risk_level in [RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value, RiskLevel.VERY_HIGH.value]:
                self.assertIn(risk_level, result["risk_allocation"])
                self.assertEqual(result["risk_allocation"][risk_level]["value_sol"], self.risk_manager.portfolio_allocation[risk_level])
    
    def test_should_reduce_position(self):
        """Test checking if a position should be reduced."""
        # Mock update_portfolio_metrics
        with patch.object(self.risk_manager, 'update_portfolio_metrics'):
            # Set portfolio metrics
            self.risk_manager.portfolio_value_sol = 10.0
            self.risk_manager.portfolio_drawdown = 25.0  # Exceeds moderate max drawdown (20%)
            
            # Check if position should be reduced
            should_reduce, reason, percentage = self.risk_manager.should_reduce_position(self.test_token_mint)
            
            # Verify result
            self.assertTrue(should_reduce)
            self.assertIn("drawdown", reason.lower())
            self.assertEqual(percentage, 50.0)
            
            # Test with normal drawdown
            self.risk_manager.portfolio_drawdown = 15.0
            
            # Set up excessive allocation for HIGH risk
            self.risk_manager.portfolio_allocation = {
                RiskLevel.LOW.value: 5.0,
                RiskLevel.MEDIUM.value: 3.0,
                RiskLevel.HIGH.value: 2.0,  # 20% of portfolio, exceeds 10% limit
                RiskLevel.VERY_HIGH.value: 0.0
            }
            self.mock_token_analytics.get_risk_level.return_value = RiskLevel.HIGH.value
            
            # Check if position should be reduced
            should_reduce, reason, percentage = self.risk_manager.should_reduce_position(self.test_token_mint)
            
            # Verify result
            self.assertTrue(should_reduce)
            self.assertIn("allocation", reason.lower())
            self.assertGreater(percentage, 0.0)
    
    def test_refine_risk_parameters(self):
        """Test refining risk parameters based on performance."""
        # Create mock performance data
        performance_data = {
            "trades": [
                {"token_mint": "token1", "profit_loss_percent": 20.0, "risk_level": RiskLevel.MEDIUM.value},
                {"token_mint": "token2", "profit_loss_percent": -15.0, "risk_level": RiskLevel.HIGH.value},
                {"token_mint": "token3", "profit_loss_percent": 5.0, "risk_level": RiskLevel.LOW.value},
                {"token_mint": "token4", "profit_loss_percent": -30.0, "risk_level": RiskLevel.VERY_HIGH.value}
            ],
            "portfolio_drawdown_max": 18.0,
            "win_rate": 0.6
        }
        
        # Refine parameters
        self.risk_manager.refine_risk_parameters(performance_data)
        
        # Verify risk allocation limits were adjusted
        # High risk should be reduced due to negative performance
        self.assertLess(self.risk_manager.risk_allocation_limits[RiskLevel.HIGH.value], 10.0)
        
        # Very high risk should be reduced significantly due to large loss
        self.assertLess(self.risk_manager.risk_allocation_limits[RiskLevel.VERY_HIGH.value], 5.0)
        
        # Medium risk should be increased due to positive performance
        self.assertGreater(self.risk_manager.risk_allocation_limits[RiskLevel.MEDIUM.value], 30.0)


if __name__ == '__main__':
    unittest.main()
