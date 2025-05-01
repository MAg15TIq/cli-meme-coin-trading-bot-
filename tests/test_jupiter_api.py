"""
Tests for the Jupiter API module.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.trading.jupiter_api import JupiterAPI


class TestJupiterAPI(unittest.TestCase):
    """Test cases for the JupiterAPI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.jupiter_api = JupiterAPI()
        self.sol_mint = "So11111111111111111111111111111111111111112"
        self.test_token_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
    
    @patch('src.trading.jupiter_api.requests.get')
    def test_get_token_price(self, mock_get):
        """Test getting a token price."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                self.test_token_mint: {
                    "id": self.test_token_mint,
                    "mintSymbol": "USDC",
                    "vsToken": self.sol_mint,
                    "vsTokenSymbol": "SOL",
                    "price": "0.05"
                }
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Get the price
        price = self.jupiter_api.get_token_price(self.test_token_mint)
        
        # Verify the price is correct
        self.assertEqual(price, 0.05)
        
        # Verify the mock was called correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], f"{self.jupiter_api.api_url}/price")
        self.assertEqual(kwargs["params"]["ids"], self.test_token_mint)
    
    @patch('src.trading.jupiter_api.requests.get')
    def test_get_quote(self, mock_get):
        """Test getting a quote."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "inputMint": self.sol_mint,
            "outputMint": self.test_token_mint,
            "inAmount": "1000000000",  # 1 SOL in lamports
            "outAmount": "20000000",   # 20 USDC (6 decimals)
            "otherAmountThreshold": "19800000",
            "swapMode": "ExactIn",
            "slippageBps": 50,
            "routes": []
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Get the quote
        quote = self.jupiter_api.get_quote(self.sol_mint, self.test_token_mint, 1000000000)
        
        # Verify the quote is correct
        self.assertEqual(quote["inAmount"], "1000000000")
        self.assertEqual(quote["outAmount"], "20000000")
        
        # Verify the mock was called correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], f"{self.jupiter_api.api_url}/quote")
        self.assertEqual(kwargs["params"]["inputMint"], self.sol_mint)
        self.assertEqual(kwargs["params"]["outputMint"], self.test_token_mint)
    
    @patch('src.trading.jupiter_api.requests.post')
    def test_execute_swap(self, mock_post):
        """Test executing a swap."""
        # Create a mock quote
        quote_data = {
            "inputMint": self.sol_mint,
            "outputMint": self.test_token_mint,
            "inAmount": "1000000000",
            "outAmount": "20000000",
            "otherAmountThreshold": "19800000",
            "swapMode": "ExactIn",
            "slippageBps": 50,
            "routes": []
        }
        
        # Create a mock wallet
        mock_wallet = MagicMock()
        mock_wallet.pubkey.return_value = "mock_pubkey"
        
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "swapTransaction": "mock_transaction_data"
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Mock the sign_and_send_transaction method
        with patch.object(self.jupiter_api, '_sign_and_send_transaction', return_value="mock_signature"):
            # Execute the swap
            signature = self.jupiter_api.execute_swap(quote_data, mock_wallet)
            
            # Verify the signature is correct
            self.assertEqual(signature, "mock_signature")
            
            # Verify the mock was called correctly
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            self.assertEqual(args[0], f"{self.jupiter_api.api_url}/swap")
            self.assertEqual(kwargs["json"]["quoteResponse"], quote_data)


if __name__ == "__main__":
    unittest.main()
