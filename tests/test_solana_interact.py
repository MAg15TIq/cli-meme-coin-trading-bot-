"""
Tests for the Solana interaction module.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.solana.solana_interact import SolanaClient
# No need to import PublicKey as we're using a string


class TestSolanaClient(unittest.TestCase):
    """Test cases for the SolanaClient class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a SolanaClient with a mock RPC URL
        self.rpc_url = "https://api.devnet.solana.com"
        self.solana_client = SolanaClient(self.rpc_url)

        # Create a test public key using a dummy value
        self.test_pubkey = "4rL4RCWHz3iNCdCgF3S73Y2qKAYwFCfCKKn3tTyZMzgp"

    @patch('src.solana.solana_interact.Client')
    def test_get_balance(self, mock_client_class):
        """Test getting a wallet balance."""
        # Set up the mock
        mock_client = mock_client_class.return_value
        mock_client.get_balance.return_value = {
            "result": {
                "value": 1_000_000_000  # 1 SOL in lamports
            }
        }

        # Create a new client with the mock
        client = SolanaClient(self.rpc_url)

        # Get the balance
        balance = client.get_balance(self.test_pubkey)

        # Verify the balance is correct
        self.assertEqual(balance, 1.0)  # 1 SOL

        # Verify the mock was called correctly
        mock_client.get_balance.assert_called_once_with(self.test_pubkey)

    @patch('src.solana.solana_interact.Client')
    def test_get_token_accounts(self, mock_client_class):
        """Test getting token accounts."""
        # Set up the mock
        mock_client = mock_client_class.return_value
        mock_client.get_token_accounts_by_owner.return_value = {
            "result": {
                "value": [
                    {
                        "pubkey": "TokenAccount1",
                        "account": {
                            "data": {
                                "parsed": {
                                    "info": {
                                        "mint": "TokenMint1",
                                        "tokenAmount": {
                                            "uiAmount": 100.5,
                                            "decimals": 6
                                        }
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        }

        # Create a new client with the mock
        client = SolanaClient(self.rpc_url)

        # Get the token accounts
        token_accounts = client.get_token_accounts(self.test_pubkey)

        # Verify the token accounts are correct
        self.assertEqual(len(token_accounts), 1)
        self.assertEqual(token_accounts[0]["mint"], "TokenMint1")
        self.assertEqual(token_accounts[0]["token_amount"], 100.5)

        # Verify the mock was called correctly
        mock_client.get_token_accounts_by_owner.assert_called_once()

    @patch('src.solana.solana_interact.Client')
    def test_get_latest_blockhash(self, mock_client_class):
        """Test getting the latest blockhash."""
        # Set up the mock
        mock_client = mock_client_class.return_value
        mock_client.get_latest_blockhash.return_value = {
            "result": {
                "value": {
                    "blockhash": "TestBlockhash123"
                }
            }
        }

        # Create a new client with the mock
        client = SolanaClient(self.rpc_url)

        # Get the latest blockhash
        blockhash = client.get_latest_blockhash()

        # Verify the blockhash is correct
        self.assertEqual(blockhash, "TestBlockhash123")

        # Verify the mock was called correctly
        mock_client.get_latest_blockhash.assert_called_once()


if __name__ == "__main__":
    unittest.main()
