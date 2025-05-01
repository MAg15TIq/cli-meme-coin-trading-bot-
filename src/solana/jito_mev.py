"""
Jito MEV protection module for the Solana Memecoin Trading Bot.
Handles integration with Jito for MEV-protected transactions.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List, Union, Tuple

import requests
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solders.pubkey import Pubkey
from solana.transaction import Transaction
from solders.keypair import Keypair
from solders.signature import Signature
from solders.message import Message
from solders.transaction import VersionedTransaction

from config import get_config_value
from src.utils.logging_utils import get_logger
from src.solana.solana_interact import solana_client

# Get logger for this module
logger = get_logger(__name__)


class JitoMEV:
    """Client for interacting with Jito MEV protection services."""
    
    def __init__(self):
        """Initialize the Jito MEV client."""
        self.enabled = get_config_value("jito_mev_enabled", False)
        self.jito_rpc_url = get_config_value("jito_rpc_url", "")
        self.jito_auth_key = get_config_value("jito_auth_key", "")
        self.jito_bundle_url = get_config_value("jito_bundle_url", "https://mainnet.bundle-api.jito.wtf/api/v1/bundles")
        
        # Initialize Jito RPC client if enabled
        if self.enabled and self.jito_rpc_url:
            self.jito_client = Client(self.jito_rpc_url)
            logger.info(f"Initialized Jito MEV client with RPC URL: {self.jito_rpc_url}")
        else:
            self.jito_client = None
            if self.enabled:
                logger.warning("Jito MEV protection is enabled but no RPC URL is configured")
    
    def is_available(self) -> bool:
        """
        Check if Jito MEV protection is available.
        
        Returns:
            True if Jito MEV protection is available, False otherwise
        """
        return self.enabled and self.jito_client is not None
    
    def send_transaction(self, transaction: Transaction, signer: Keypair,
                        priority_fee: Optional[int] = None, compute_limit: Optional[int] = None) -> str:
        """
        Send a transaction with Jito MEV protection.
        
        Args:
            transaction: The transaction to send
            signer: The keypair to sign the transaction with
            priority_fee: Optional priority fee in micro-lamports
            compute_limit: Optional compute unit limit
            
        Returns:
            The transaction signature
        """
        if not self.is_available():
            logger.warning("Jito MEV protection is not available, falling back to regular transaction")
            return solana_client.send_transaction(transaction, signer, priority_fee, compute_limit)
        
        try:
            # Get recent blockhash from Jito
            blockhash = self._get_latest_blockhash()
            transaction.recent_blockhash = blockhash
            
            # Add compute budget instructions if needed
            if priority_fee is not None or compute_limit is not None:
                # Use solana_client to add compute budget instructions
                # This is a bit of a hack, but it works
                temp_tx = Transaction()
                temp_tx.recent_blockhash = blockhash
                temp_tx = solana_client._add_compute_budget_instructions(temp_tx, priority_fee, compute_limit)
                
                # Add the compute budget instructions to our transaction
                for ix in temp_tx.instructions:
                    if ix not in transaction.instructions:
                        transaction.instructions.insert(0, ix)
            
            # Sign transaction
            transaction.sign(signer)
            
            # Send as a Jito bundle
            return self._send_bundle([transaction], [signer])
        except Exception as e:
            logger.error(f"Error sending transaction with Jito MEV protection: {e}")
            logger.info("Falling back to regular transaction")
            return solana_client.send_transaction(transaction, signer, priority_fee, compute_limit)
    
    def send_versioned_transaction(self, message: Message, signers: List[Keypair],
                                  priority_fee: Optional[int] = None, 
                                  compute_limit: Optional[int] = None) -> str:
        """
        Send a versioned transaction with Jito MEV protection.
        
        Args:
            message: The transaction message
            signers: List of keypairs to sign the transaction with
            priority_fee: Optional priority fee in micro-lamports
            compute_limit: Optional compute unit limit
            
        Returns:
            The transaction signature
        """
        if not self.is_available():
            logger.warning("Jito MEV protection is not available, falling back to regular transaction")
            return solana_client.send_versioned_transaction(message, signers)
        
        try:
            # Create versioned transaction
            tx = VersionedTransaction(message, [signer.sign_message(bytes(message)) for signer in signers])
            
            # Send as a Jito bundle
            return self._send_versioned_bundle([tx])
        except Exception as e:
            logger.error(f"Error sending versioned transaction with Jito MEV protection: {e}")
            logger.info("Falling back to regular transaction")
            return solana_client.send_versioned_transaction(message, signers)
    
    def _get_latest_blockhash(self) -> str:
        """
        Get the latest blockhash from Jito.
        
        Returns:
            The latest blockhash as a string
        """
        try:
            response = self.jito_client.get_latest_blockhash()
            return response["result"]["value"]["blockhash"]
        except Exception as e:
            logger.error(f"Error getting latest blockhash from Jito: {e}")
            logger.info("Falling back to regular RPC for blockhash")
            return solana_client.get_latest_blockhash()
    
    def _send_bundle(self, transactions: List[Transaction], signers: List[Keypair]) -> str:
        """
        Send a bundle of transactions to Jito.
        
        Args:
            transactions: List of transactions to send
            signers: List of keypairs that signed the transactions
            
        Returns:
            The transaction signature of the first transaction
        """
        try:
            # Convert transactions to wire format
            tx_data = []
            for tx in transactions:
                tx_bytes = tx.serialize()
                tx_data.append(tx_bytes.hex())
            
            # Create bundle request
            bundle_request = {
                "transactions": tx_data,
                "metadata": {
                    "description": "Solana Memecoin Trading Bot transaction"
                }
            }
            
            # Add auth header if available
            headers = {
                "Content-Type": "application/json"
            }
            if self.jito_auth_key:
                headers["Authorization"] = f"Bearer {self.jito_auth_key}"
            
            # Send bundle request
            response = requests.post(self.jito_bundle_url, json=bundle_request, headers=headers)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            if "uuid" in result:
                logger.info(f"Bundle submitted to Jito: {result['uuid']}")
                
                # Return the signature of the first transaction
                return transactions[0].signatures[0].to_string()
            else:
                logger.error(f"Error submitting bundle to Jito: {result}")
                raise Exception(f"Error submitting bundle to Jito: {result}")
        except Exception as e:
            logger.error(f"Error sending bundle to Jito: {e}")
            
            # Fall back to regular transaction submission
            logger.info("Falling back to regular transaction submission")
            signatures = []
            for i, tx in enumerate(transactions):
                signer = signers[i] if i < len(signers) else signers[-1]
                sig = solana_client.send_transaction(tx, signer)
                signatures.append(sig)
            
            return signatures[0]
    
    def _send_versioned_bundle(self, transactions: List[VersionedTransaction]) -> str:
        """
        Send a bundle of versioned transactions to Jito.
        
        Args:
            transactions: List of versioned transactions to send
            
        Returns:
            The transaction signature of the first transaction
        """
        try:
            # Convert transactions to wire format
            tx_data = []
            for tx in transactions:
                tx_bytes = bytes(tx)
                tx_data.append(tx_bytes.hex())
            
            # Create bundle request
            bundle_request = {
                "transactions": tx_data,
                "metadata": {
                    "description": "Solana Memecoin Trading Bot versioned transaction"
                }
            }
            
            # Add auth header if available
            headers = {
                "Content-Type": "application/json"
            }
            if self.jito_auth_key:
                headers["Authorization"] = f"Bearer {self.jito_auth_key}"
            
            # Send bundle request
            response = requests.post(self.jito_bundle_url, json=bundle_request, headers=headers)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            if "uuid" in result:
                logger.info(f"Bundle submitted to Jito: {result['uuid']}")
                
                # Return the signature of the first transaction
                return transactions[0].signatures[0].to_string()
            else:
                logger.error(f"Error submitting bundle to Jito: {result}")
                raise Exception(f"Error submitting bundle to Jito: {result}")
        except Exception as e:
            logger.error(f"Error sending versioned bundle to Jito: {e}")
            
            # Fall back to regular transaction submission
            logger.info("Falling back to regular transaction submission")
            signatures = []
            for tx in transactions:
                sig = solana_client.client.send_transaction(tx, opts=TxOpts(skip_preflight=False, preflight_commitment="confirmed"))
                signatures.append(sig["result"])
            
            return signatures[0]


# Create a singleton instance
jito_mev = JitoMEV()
