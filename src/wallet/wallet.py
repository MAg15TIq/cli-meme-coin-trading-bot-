"""
Wallet management module for the Solana Memecoin Trading Bot.
Handles secure storage and loading of wallet keys.
"""

import os
import json
import base64
import logging
import getpass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from solders.keypair import Keypair

from config import get_config_value, ensure_config_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WalletManager:
    """Manager for Solana wallet operations and secure key storage."""

    def __init__(self):
        """Initialize the wallet manager."""
        self.wallet_path = Path(get_config_value("wallet_path"))
        self.encrypted_key_file = self.wallet_path / get_config_value("encrypted_key_file")
        self.current_keypair: Optional[Keypair] = None
        self.current_wallet_name: Optional[str] = None

        # Dictionary of loaded wallets: name -> keypair
        self.wallets: Dict[str, Keypair] = {}

        # Wallet metadata: name -> metadata
        self.wallet_metadata: Dict[str, Dict[str, Any]] = {}

        # Ensure wallet directory exists
        self.wallet_path.mkdir(parents=True, exist_ok=True)

        # Load wallet metadata
        self._load_wallet_metadata()

    def _derive_key(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Derive an encryption key from a password.

        Args:
            password: The password to derive the key from
            salt: Optional salt for key derivation

        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt

    def encrypt_keypair(self, keypair: Keypair, password: str) -> Dict[str, Any]:
        """
        Encrypt a keypair with a password.

        Args:
            keypair: The keypair to encrypt
            password: The password to encrypt with

        Returns:
            Dictionary with encrypted key data
        """
        # Derive encryption key
        salt = os.urandom(16)
        key, salt = self._derive_key(password, salt)

        # Create Fernet cipher
        cipher = Fernet(key)

        # Encrypt the private key - ensure we have the full 64 bytes
        # The secret() method returns only the first 32 bytes, but we need all 64 bytes
        private_key_bytes = bytes(keypair)
        encrypted_private_key = cipher.encrypt(private_key_bytes)

        # Create encrypted key data
        encrypted_data = {
            "public_key": str(keypair.pubkey()),
            "encrypted_private_key": encrypted_private_key.decode(),
            "salt": base64.b64encode(salt).decode(),
        }

        return encrypted_data

    def decrypt_keypair(self, encrypted_data: Dict[str, Any], password: str) -> Keypair:
        """
        Decrypt a keypair with a password.

        Args:
            encrypted_data: The encrypted key data
            password: The password to decrypt with

        Returns:
            The decrypted keypair
        """
        # Get salt and derive key
        salt = base64.b64decode(encrypted_data["salt"])
        key, _ = self._derive_key(password, salt)

        # Create Fernet cipher
        cipher = Fernet(key)

        # Decrypt the private key
        encrypted_private_key = encrypted_data["encrypted_private_key"].encode()
        private_key_bytes = cipher.decrypt(encrypted_private_key)

        # Create keypair from private key bytes (should be 64 bytes)
        keypair = Keypair.from_bytes(private_key_bytes)

        # Verify the public key matches
        if str(keypair.pubkey()) != encrypted_data["public_key"]:
            raise ValueError("Decrypted keypair does not match expected public key")

        return keypair

    def save_encrypted_keypair(self, keypair: Keypair, password: str, filename: Optional[str] = None) -> str:
        """
        Encrypt and save a keypair to a file.

        Args:
            keypair: The keypair to save
            password: The password to encrypt with
            filename: Optional filename to save to

        Returns:
            The path to the saved file
        """
        # Encrypt the keypair
        encrypted_data = self.encrypt_keypair(keypair, password)

        # Determine file path
        if filename is None:
            filename = self.encrypted_key_file
        else:
            filename = self.wallet_path / filename

        # Save to file
        with open(filename, 'w') as f:
            json.dump(encrypted_data, f, indent=4)

        logger.info(f"Encrypted keypair saved to {filename}")
        return str(filename)

    def load_encrypted_keypair(self, password: str, filename: Optional[str] = None) -> Keypair:
        """
        Load and decrypt a keypair from a file.

        Args:
            password: The password to decrypt with
            filename: Optional filename to load from

        Returns:
            The decrypted keypair
        """
        # Determine file path
        if filename is None:
            filename = self.encrypted_key_file
        else:
            filename = self.wallet_path / filename

        # Check if file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Encrypted key file not found: {filename}")

        # Load from file
        with open(filename, 'r') as f:
            encrypted_data = json.load(f)

        # Decrypt the keypair
        keypair = self.decrypt_keypair(encrypted_data, password)

        logger.info(f"Loaded keypair for {keypair.pubkey()} from {filename}")
        self.current_keypair = keypair
        return keypair

    def create_new_keypair(self) -> Keypair:
        """
        Create a new random keypair.

        Returns:
            The new keypair
        """
        keypair = Keypair()
        logger.info(f"Created new keypair with public key: {keypair.pubkey()}")
        return keypair

    def import_keypair_from_bytes(self, private_key_bytes: bytes) -> Keypair:
        """
        Import a keypair from private key bytes.

        Args:
            private_key_bytes: The private key bytes

        Returns:
            The imported keypair
        """
        keypair = Keypair.from_bytes(private_key_bytes)
        logger.info(f"Imported keypair with public key: {keypair.pubkey()}")
        return keypair

    def import_keypair_from_base58(self, private_key_base58: str) -> Keypair:
        """
        Import a keypair from a base58-encoded private key.

        Args:
            private_key_base58: The base58-encoded private key

        Returns:
            The imported keypair
        """
        keypair = Keypair.from_base58_string(private_key_base58)
        logger.info(f"Imported keypair with public key: {keypair.pubkey()}")
        return keypair

    def _load_wallet_metadata(self) -> None:
        """
        Load wallet metadata from file.
        """
        metadata_file = self.wallet_path / "wallet_metadata.json"
        if not metadata_file.exists():
            logger.info("No wallet metadata file found")
            return

        try:
            with open(metadata_file, 'r') as f:
                self.wallet_metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(self.wallet_metadata)} wallets")
        except Exception as e:
            logger.error(f"Error loading wallet metadata: {e}")

    def _save_wallet_metadata(self) -> None:
        """
        Save wallet metadata to file.
        """
        metadata_file = self.wallet_path / "wallet_metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.wallet_metadata, f, indent=4)
            logger.info(f"Saved metadata for {len(self.wallet_metadata)} wallets")
        except Exception as e:
            logger.error(f"Error saving wallet metadata: {e}")

    def get_current_keypair(self) -> Optional[Keypair]:
        """
        Get the currently loaded keypair.

        Returns:
            The current keypair or None if not loaded
        """
        return self.current_keypair

    def get_current_wallet_name(self) -> Optional[str]:
        """
        Get the name of the currently active wallet.

        Returns:
            The current wallet name or None if no wallet is active
        """
        return self.current_wallet_name

    def list_wallets(self) -> List[Dict[str, Any]]:
        """
        List all available wallets.

        Returns:
            List of wallet information dictionaries
        """
        wallets = []

        # Check all files in the wallet directory
        for file in self.wallet_path.glob("*.json"):
            if file.name == "wallet_metadata.json":
                continue

            wallet_name = file.stem
            wallet_info = {
                "name": wallet_name,
                "path": str(file),
                "active": wallet_name == self.current_wallet_name,
                "metadata": self.wallet_metadata.get(wallet_name, {})
            }
            wallets.append(wallet_info)

        return wallets

    def switch_wallet(self, wallet_name: str, password: str) -> Keypair:
        """
        Switch to a different wallet.

        Args:
            wallet_name: The name of the wallet to switch to
            password: The password to decrypt the wallet

        Returns:
            The loaded keypair
        """
        # Check if wallet is already loaded
        if wallet_name in self.wallets:
            self.current_keypair = self.wallets[wallet_name]
            self.current_wallet_name = wallet_name
            logger.info(f"Switched to already loaded wallet: {wallet_name}")
            return self.current_keypair

        # Load the wallet
        filename = f"{wallet_name}.json"
        keypair = self.load_encrypted_keypair(password, filename)

        # Store in wallets dictionary
        self.wallets[wallet_name] = keypair
        self.current_keypair = keypair
        self.current_wallet_name = wallet_name

        logger.info(f"Switched to wallet: {wallet_name}")
        return keypair

    def add_wallet_metadata(self, wallet_name: str, metadata: Dict[str, Any]) -> None:
        """
        Add or update metadata for a wallet.

        Args:
            wallet_name: The name of the wallet
            metadata: The metadata to store
        """
        # Update metadata
        if wallet_name in self.wallet_metadata:
            self.wallet_metadata[wallet_name].update(metadata)
        else:
            self.wallet_metadata[wallet_name] = metadata

        # Save to file
        self._save_wallet_metadata()
        logger.info(f"Updated metadata for wallet: {wallet_name}")

    def setup_wallet_interactive(self) -> Keypair:
        """
        Interactive setup for a wallet.
        Prompts the user to create a new wallet or import an existing one.

        Returns:
            The configured keypair
        """
        print("\n=== Wallet Setup ===")
        print("1. Create a new wallet")
        print("2. Import from private key (base58)")
        choice = input("Choose an option (1-2): ")

        keypair = None
        if choice == "1":
            keypair = self.create_new_keypair()
            print(f"Created new wallet with public key: {keypair.pubkey()}")
        elif choice == "2":
            private_key = getpass.getpass("Enter private key (base58): ")
            keypair = self.import_keypair_from_base58(private_key)
            print(f"Imported wallet with public key: {keypair.pubkey()}")
        else:
            print("Invalid choice. Please try again.")
            return self.setup_wallet_interactive()

        # Get wallet name
        wallet_name = input("Enter a name for this wallet: ")
        while not wallet_name:
            print("Wallet name is required.")
            wallet_name = input("Enter a name for this wallet: ")

        # Get wallet metadata
        description = input("Enter a description for this wallet (optional): ")
        tags = input("Enter tags for this wallet (comma-separated, optional): ")

        metadata = {
            "description": description,
            "tags": [tag.strip() for tag in tags.split(",")] if tags else [],
            "created_at": datetime.now().isoformat(),
            "public_key": str(keypair.pubkey())
        }

        # Encrypt and save the keypair
        password = getpass.getpass("Create a password to encrypt your wallet: ")
        confirm_password = getpass.getpass("Confirm password: ")

        if password != confirm_password:
            print("Passwords do not match. Please try again.")
            return self.setup_wallet_interactive()

        # Save with the wallet name as filename
        filename = f"{wallet_name}.json"

        self.save_encrypted_keypair(keypair, password, filename)
        self.current_keypair = keypair
        self.current_wallet_name = wallet_name

        # Store in wallets dictionary
        self.wallets[wallet_name] = keypair

        # Save metadata
        self.add_wallet_metadata(wallet_name, metadata)

        print(f"Wallet setup complete. Name: {wallet_name}, Public key: {keypair.pubkey()}")
        return keypair


# Create a singleton instance
wallet_manager = WalletManager()
