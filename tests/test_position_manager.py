"""
Tests for the position manager module.
"""

import unittest
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime

from src.trading.position_manager import Position, PositionManager


class TestPosition(unittest.TestCase):
    """Test cases for the Position class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.token_mint = "TokenMint123"
        self.token_name = "TestToken"
        self.amount = 100.0
        self.entry_price = 0.01  # in SOL
        self.entry_time = datetime.now()
        self.stop_loss = 0.009  # 10% below entry
        self.take_profit = 0.012  # 20% above entry
        self.decimals = 9
        
        self.position = Position(
            token_mint=self.token_mint,
            token_name=self.token_name,
            amount=self.amount,
            entry_price=self.entry_price,
            entry_time=self.entry_time,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            decimals=self.decimals
        )
    
    def test_update_price(self):
        """Test updating the price of a position."""
        new_price = 0.011
        self.position.update_price(new_price)
        
        self.assertEqual(self.position.current_price, new_price)
        self.assertGreater(self.position.last_updated, self.entry_time)
    
    def test_get_pnl(self):
        """Test calculating profit/loss."""
        # Set current price to 20% above entry
        self.position.update_price(0.012)
        
        # Calculate expected P&L
        expected_pnl = 20.0  # 20%
        
        # Verify P&L calculation
        self.assertEqual(self.position.get_pnl(), expected_pnl)
    
    def test_get_value(self):
        """Test calculating position value."""
        # Set current price
        current_price = 0.015
        self.position.update_price(current_price)
        
        # Calculate expected value
        expected_value = self.amount * current_price
        
        # Verify value calculation
        self.assertEqual(self.position.get_value(), expected_value)
    
    def test_to_dict_from_dict(self):
        """Test converting position to dict and back."""
        # Convert to dict
        position_dict = self.position.to_dict()
        
        # Verify dict contains expected keys
        self.assertIn("token_mint", position_dict)
        self.assertIn("token_name", position_dict)
        self.assertIn("amount", position_dict)
        self.assertIn("entry_price", position_dict)
        self.assertIn("entry_time", position_dict)
        self.assertIn("stop_loss", position_dict)
        self.assertIn("take_profit", position_dict)
        self.assertIn("decimals", position_dict)
        
        # Convert back to position
        new_position = Position.from_dict(position_dict)
        
        # Verify position attributes
        self.assertEqual(new_position.token_mint, self.token_mint)
        self.assertEqual(new_position.token_name, self.token_name)
        self.assertEqual(new_position.amount, self.amount)
        self.assertEqual(new_position.entry_price, self.entry_price)
        self.assertEqual(new_position.stop_loss, self.stop_loss)
        self.assertEqual(new_position.take_profit, self.take_profit)
        self.assertEqual(new_position.decimals, self.decimals)


class TestPositionManager(unittest.TestCase):
    """Test cases for the PositionManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for position files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a position manager with the temporary directory
        with patch('src.trading.position_manager.Path.home') as mock_home:
            mock_home.return_value = Path(self.temp_dir.name)
            self.position_manager = PositionManager()
        
        # Create a test position
        self.test_position = Position(
            token_mint="TestMint123",
            token_name="TestToken",
            amount=100.0,
            entry_price=0.01,
            entry_time=datetime.now(),
            stop_loss=0.009,
            take_profit=0.012,
            decimals=9
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop monitoring thread if running
        self.position_manager.stop_monitoring_thread()
        self.temp_dir.cleanup()
    
    def test_add_get_remove_position(self):
        """Test adding, getting, and removing a position."""
        # Add position
        self.position_manager.add_position(self.test_position)
        
        # Verify position was added
        self.assertIn(self.test_position.token_mint, self.position_manager.positions)
        
        # Get position
        position = self.position_manager.get_position(self.test_position.token_mint)
        
        # Verify position is the same
        self.assertEqual(position.token_mint, self.test_position.token_mint)
        self.assertEqual(position.token_name, self.test_position.token_name)
        
        # Remove position
        removed_position = self.position_manager.remove_position(self.test_position.token_mint)
        
        # Verify position was removed
        self.assertNotIn(self.test_position.token_mint, self.position_manager.positions)
        self.assertEqual(removed_position.token_mint, self.test_position.token_mint)
    
    def test_get_all_positions(self):
        """Test getting all positions."""
        # Add multiple positions
        position1 = Position(
            token_mint="TestMint1",
            token_name="TestToken1",
            amount=100.0,
            entry_price=0.01,
            entry_time=datetime.now(),
            stop_loss=0.009,
            take_profit=0.012,
            decimals=9
        )
        
        position2 = Position(
            token_mint="TestMint2",
            token_name="TestToken2",
            amount=200.0,
            entry_price=0.02,
            entry_time=datetime.now(),
            stop_loss=0.018,
            take_profit=0.024,
            decimals=9
        )
        
        self.position_manager.add_position(position1)
        self.position_manager.add_position(position2)
        
        # Get all positions
        positions = self.position_manager.get_all_positions()
        
        # Verify positions
        self.assertEqual(len(positions), 2)
        self.assertIn(position1.token_mint, [p.token_mint for p in positions])
        self.assertIn(position2.token_mint, [p.token_mint for p in positions])
    
    @patch('src.trading.position_manager.jupiter_api')
    def test_update_position_price(self, mock_jupiter_api):
        """Test updating position price."""
        # Set up mock
        mock_jupiter_api.get_token_price.return_value = 0.015
        
        # Add position
        self.position_manager.add_position(self.test_position)
        
        # Update price
        self.position_manager.update_position_price(self.test_position.token_mint)
        
        # Verify price was updated
        position = self.position_manager.get_position(self.test_position.token_mint)
        self.assertEqual(position.current_price, 0.015)
        
        # Verify mock was called
        mock_jupiter_api.get_token_price.assert_called_once_with(self.test_position.token_mint)
    
    @patch('src.trading.position_manager.jupiter_api')
    def test_check_stop_loss_take_profit_not_triggered(self, mock_jupiter_api):
        """Test checking stop-loss/take-profit when not triggered."""
        # Set up mock
        mock_jupiter_api.get_token_price.return_value = 0.011  # Between SL and TP
        
        # Add position
        self.position_manager.add_position(self.test_position)
        
        # Check SL/TP
        triggered = self.position_manager.check_stop_loss_take_profit(self.test_position.token_mint)
        
        # Verify not triggered
        self.assertFalse(triggered)
        self.assertIn(self.test_position.token_mint, self.position_manager.positions)
    
    @patch('src.trading.position_manager.jupiter_api')
    def test_check_stop_loss_triggered(self, mock_jupiter_api):
        """Test checking stop-loss when triggered."""
        # Set up mocks
        mock_jupiter_api.get_token_price.return_value = 0.008  # Below SL
        
        # Mock execute_sell to avoid actual execution
        with patch.object(self.position_manager, 'execute_sell', return_value="mock_signature"):
            # Add position
            self.position_manager.add_position(self.test_position)
            
            # Update price to trigger SL
            self.position_manager.update_position_price(self.test_position.token_mint, 0.008)
            
            # Check SL/TP
            triggered = self.position_manager.check_stop_loss_take_profit(self.test_position.token_mint)
            
            # Verify triggered
            self.assertTrue(triggered)
            self.assertNotIn(self.test_position.token_mint, self.position_manager.positions)
            
            # Verify execute_sell was called
            self.position_manager.execute_sell.assert_called_once()
    
    @patch('src.trading.position_manager.jupiter_api')
    def test_check_take_profit_triggered(self, mock_jupiter_api):
        """Test checking take-profit when triggered."""
        # Set up mocks
        mock_jupiter_api.get_token_price.return_value = 0.013  # Above TP
        
        # Mock execute_sell to avoid actual execution
        with patch.object(self.position_manager, 'execute_sell', return_value="mock_signature"):
            # Add position
            self.position_manager.add_position(self.test_position)
            
            # Update price to trigger TP
            self.position_manager.update_position_price(self.test_position.token_mint, 0.013)
            
            # Check SL/TP
            triggered = self.position_manager.check_stop_loss_take_profit(self.test_position.token_mint)
            
            # Verify triggered
            self.assertTrue(triggered)
            self.assertNotIn(self.test_position.token_mint, self.position_manager.positions)
            
            # Verify execute_sell was called
            self.position_manager.execute_sell.assert_called_once()
    
    def test_save_load_positions(self):
        """Test saving and loading positions."""
        # Add position
        self.position_manager.add_position(self.test_position)
        
        # Create a new position manager to load the positions
        with patch('src.trading.position_manager.Path.home') as mock_home:
            mock_home.return_value = Path(self.temp_dir.name)
            new_position_manager = PositionManager()
        
        # Verify position was loaded
        self.assertIn(self.test_position.token_mint, new_position_manager.positions)
        loaded_position = new_position_manager.get_position(self.test_position.token_mint)
        self.assertEqual(loaded_position.token_mint, self.test_position.token_mint)
        self.assertEqual(loaded_position.token_name, self.test_position.token_name)
        
        # Clean up
        new_position_manager.stop_monitoring_thread()


if __name__ == "__main__":
    unittest.main()
