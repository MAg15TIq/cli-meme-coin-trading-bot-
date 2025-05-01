"""
Tests for the wallet module.
"""

import os
import unittest
import tempfile
from pathlib import Path

from src.wallet.wallet import WalletManager
from solders.keypair import Keypair


class TestWalletManager(unittest.TestCase):
    """Test cases for the WalletManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for wallet files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wallet_path = Path(self.temp_dir.name)
        
        # Create a wallet manager with the temporary directory
        self.wallet_manager = WalletManager()
        self.wallet_manager.wallet_path = self.wallet_path
        self.wallet_manager.encrypted_key_file = self.wallet_path / "test_encrypted_key.json"
        
        # Create a test keypair
        self.test_keypair = Keypair()
        self.test_password = "test_password"
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_encrypt_decrypt_keypair(self):
        """Test encrypting and decrypting a keypair."""
        # Encrypt the keypair
        encrypted_data = self.wallet_manager.encrypt_keypair(self.test_keypair, self.test_password)
        
        # Verify the encrypted data contains the expected fields
        self.assertIn("public_key", encrypted_data)
        self.assertIn("encrypted_private_key", encrypted_data)
        self.assertIn("salt", encrypted_data)
        
        # Verify the public key matches
        self.assertEqual(encrypted_data["public_key"], str(self.test_keypair.pubkey()))
        
        # Decrypt the keypair
        decrypted_keypair = self.wallet_manager.decrypt_keypair(encrypted_data, self.test_password)
        
        # Verify the decrypted keypair matches the original
        self.assertEqual(str(decrypted_keypair.pubkey()), str(self.test_keypair.pubkey()))
        self.assertEqual(decrypted_keypair.secret(), self.test_keypair.secret())
    
    def test_save_load_encrypted_keypair(self):
        """Test saving and loading an encrypted keypair."""
        # Save the encrypted keypair
        filename = self.wallet_manager.save_encrypted_keypair(self.test_keypair, self.test_password)
        
        # Verify the file exists
        self.assertTrue(os.path.exists(filename))
        
        # Load the encrypted keypair
        loaded_keypair = self.wallet_manager.load_encrypted_keypair(self.test_password, filename)
        
        # Verify the loaded keypair matches the original
        self.assertEqual(str(loaded_keypair.pubkey()), str(self.test_keypair.pubkey()))
        self.assertEqual(loaded_keypair.secret(), self.test_keypair.secret())
    
    def test_wrong_password(self):
        """Test decrypting with the wrong password."""
        # Encrypt the keypair
        encrypted_data = self.wallet_manager.encrypt_keypair(self.test_keypair, self.test_password)
        
        # Try to decrypt with the wrong password
        with self.assertRaises(Exception):
            self.wallet_manager.decrypt_keypair(encrypted_data, "wrong_password")
    
    def test_create_new_keypair(self):
        """Test creating a new keypair."""
        # Create a new keypair
        keypair = self.wallet_manager.create_new_keypair()
        
        # Verify it's a valid keypair
        self.assertIsInstance(keypair, Keypair)
        self.assertIsNotNone(keypair.pubkey())
        self.assertIsNotNone(keypair.secret())


if __name__ == "__main__":
    unittest.main()
