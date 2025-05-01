"""
Performance tracking module for the Solana Memecoin Trading Bot.
Collects and stores performance data for risk management and gas optimization refinement.
"""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class PerformanceTracker:
    """Tracks performance metrics for refinement of bot parameters."""

    def __init__(self):
        """Initialize the performance tracker."""
        # Data storage paths
        self.data_dir = Path(get_config_value("data_dir", str(Path.home() / ".solana-trading-bot")))
        self.trades_file = self.data_dir / "performance" / "trades.json"
        self.transactions_file = self.data_dir / "performance" / "transactions.json"
        self.portfolio_file = self.data_dir / "performance" / "portfolio.json"
        
        # Ensure directories exist
        (self.data_dir / "performance").mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self.trades = self._load_data(self.trades_file, "trades")
        self.transactions = self._load_data(self.transactions_file, "transactions")
        self.portfolio_metrics = self._load_data(self.portfolio_file, "portfolio")
        
        # Performance metrics
        self.win_rate = 0.0
        self.max_drawdown = 0.0
        self.avg_profit_loss = 0.0
        self.transaction_success_rate = 0.0
        
        # Update metrics if data exists
        if self.trades or self.transactions or self.portfolio_metrics:
            self._calculate_metrics()
        
        # Auto-refinement settings
        self.auto_refinement_enabled = get_config_value("auto_refinement_enabled", False)
        self.risk_refinement_interval_days = int(get_config_value("risk_refinement_interval_days", "7"))
        self.gas_refinement_interval_days = int(get_config_value("gas_refinement_interval_days", "3"))
        self.last_risk_refinement = float(get_config_value("last_risk_refinement", "0"))
        self.last_gas_refinement = float(get_config_value("last_gas_refinement", "0"))
        
        logger.info("Performance tracker initialized")
    
    def _load_data(self, file_path: Path, data_type: str) -> List[Dict[str, Any]]:
        """
        Load data from file.
        
        Args:
            file_path: Path to the data file
            data_type: Type of data being loaded
            
        Returns:
            List of data entries
        """
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} {data_type} entries from {file_path}")
                    return data
        except Exception as e:
            logger.error(f"Error loading {data_type} data: {e}")
        
        return []
    
    def _save_data(self, data: List[Dict[str, Any]], file_path: Path, data_type: str) -> bool:
        """
        Save data to file.
        
        Args:
            data: Data to save
            file_path: Path to save to
            data_type: Type of data being saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(data)} {data_type} entries to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving {data_type} data: {e}")
            return False
    
    def record_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Record a trade for performance tracking.
        
        Args:
            trade_data: Trade data including:
                - token_mint: Token mint address
                - token_name: Token name
                - entry_price: Entry price
                - exit_price: Exit price (if closed)
                - entry_time: Entry timestamp
                - exit_time: Exit timestamp (if closed)
                - amount: Token amount
                - value_sol: SOL value
                - profit_loss_sol: Profit/loss in SOL (if closed)
                - profit_loss_percent: Profit/loss percentage (if closed)
                - risk_level: Token risk level
                - status: "open" or "closed"
                
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add timestamp if not provided
            if "timestamp" not in trade_data:
                trade_data["timestamp"] = datetime.now().timestamp()
            
            # Add trade ID if not provided
            if "trade_id" not in trade_data:
                trade_data["trade_id"] = f"trade_{int(time.time())}_{len(self.trades)}"
            
            # Add to trades list
            self.trades.append(trade_data)
            
            # Save to file
            success = self._save_data(self.trades, self.trades_file, "trades")
            
            # Recalculate metrics
            if success:
                self._calculate_metrics()
            
            return success
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return False
    
    def record_transaction(self, tx_data: Dict[str, Any]) -> bool:
        """
        Record a transaction for performance tracking.
        
        Args:
            tx_data: Transaction data including:
                - signature: Transaction signature
                - success: Whether the transaction succeeded
                - priority_fee: The priority fee used
                - tx_type: The transaction type
                - timestamp: Transaction timestamp
                - congestion: Network congestion at the time
                - time_of_day: Hour of day (UTC)
                
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add timestamp if not provided
            if "timestamp" not in tx_data:
                tx_data["timestamp"] = datetime.now().timestamp()
            
            # Add time_of_day if not provided
            if "time_of_day" not in tx_data:
                tx_data["time_of_day"] = datetime.fromtimestamp(tx_data["timestamp"]).hour
            
            # Add to transactions list
            self.transactions.append(tx_data)
            
            # Save to file
            success = self._save_data(self.transactions, self.transactions_file, "transactions")
            
            # Recalculate metrics
            if success:
                self._calculate_metrics()
            
            return success
        except Exception as e:
            logger.error(f"Error recording transaction: {e}")
            return False
    
    def record_portfolio_metrics(self, portfolio_data: Dict[str, Any]) -> bool:
        """
        Record portfolio metrics for performance tracking.
        
        Args:
            portfolio_data: Portfolio data including:
                - portfolio_value_sol: Total portfolio value in SOL
                - wallet_balance_sol: Wallet balance in SOL
                - position_value_sol: Total position value in SOL
                - portfolio_drawdown: Current portfolio drawdown percentage
                - risk_allocation: Allocation by risk level
                - timestamp: Timestamp of the metrics
                
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add timestamp if not provided
            if "timestamp" not in portfolio_data:
                portfolio_data["timestamp"] = datetime.now().timestamp()
            
            # Add to portfolio metrics list
            self.portfolio_metrics.append(portfolio_data)
            
            # Keep only the last 1000 entries to prevent file growth
            if len(self.portfolio_metrics) > 1000:
                self.portfolio_metrics = self.portfolio_metrics[-1000:]
            
            # Save to file
            success = self._save_data(self.portfolio_metrics, self.portfolio_file, "portfolio")
            
            # Recalculate metrics
            if success:
                self._calculate_metrics()
            
            return success
        except Exception as e:
            logger.error(f"Error recording portfolio metrics: {e}")
            return False
    
    def _calculate_metrics(self) -> None:
        """Calculate performance metrics from stored data."""
        try:
            # Calculate trade metrics
            closed_trades = [t for t in self.trades if t.get("status") == "closed"]
            
            if closed_trades:
                # Calculate win rate
                winning_trades = [t for t in closed_trades if t.get("profit_loss_percent", 0) > 0]
                self.win_rate = len(winning_trades) / len(closed_trades)
                
                # Calculate average profit/loss
                pnl_values = [t.get("profit_loss_percent", 0) for t in closed_trades]
                self.avg_profit_loss = sum(pnl_values) / len(pnl_values)
            
            # Calculate transaction metrics
            if self.transactions:
                successful_txs = [tx for tx in self.transactions if tx.get("success", False)]
                self.transaction_success_rate = len(successful_txs) / len(self.transactions)
            
            # Calculate portfolio metrics
            if self.portfolio_metrics:
                # Calculate max drawdown
                drawdown_values = [p.get("portfolio_drawdown", 0) for p in self.portfolio_metrics]
                self.max_drawdown = max(drawdown_values) if drawdown_values else 0
            
            logger.info(f"Performance metrics calculated: win_rate={self.win_rate:.2f}, " +
                       f"avg_pnl={self.avg_profit_loss:.2f}%, max_drawdown={self.max_drawdown:.2f}%, " +
                       f"tx_success_rate={self.transaction_success_rate:.2f}")
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    def get_risk_refinement_data(self) -> Dict[str, Any]:
        """
        Get data for risk parameter refinement.
        
        Returns:
            Dictionary with data for risk refinement
        """
        try:
            # Get closed trades
            closed_trades = [t for t in self.trades if t.get("status") == "closed"]
            
            # Format trades for risk manager
            trades_for_refinement = []
            for trade in closed_trades:
                trades_for_refinement.append({
                    "token_mint": trade.get("token_mint", ""),
                    "profit_loss_percent": trade.get("profit_loss_percent", 0),
                    "risk_level": trade.get("risk_level", "medium"),
                    "entry_time": trade.get("entry_time", 0),
                    "exit_time": trade.get("exit_time", 0),
                    "value_sol": trade.get("value_sol", 0)
                })
            
            # Get portfolio drawdown
            portfolio_drawdown_max = self.max_drawdown
            
            return {
                "trades": trades_for_refinement,
                "portfolio_drawdown_max": portfolio_drawdown_max,
                "win_rate": self.win_rate,
                "avg_profit_loss": self.avg_profit_loss,
                "trade_count": len(closed_trades)
            }
        except Exception as e:
            logger.error(f"Error getting risk refinement data: {e}")
            return {"error": str(e)}
    
    def get_gas_refinement_data(self) -> Dict[str, Any]:
        """
        Get data for gas parameter refinement.
        
        Returns:
            Dictionary with data for gas refinement
        """
        try:
            # Get recent transactions (last 7 days)
            cutoff_time = datetime.now().timestamp() - (7 * 24 * 3600)
            recent_txs = [tx for tx in self.transactions if tx.get("timestamp", 0) > cutoff_time]
            
            # Format transactions for gas optimizer
            txs_for_refinement = []
            for tx in recent_txs:
                txs_for_refinement.append({
                    "signature": tx.get("signature", ""),
                    "success": tx.get("success", False),
                    "priority_fee": tx.get("priority_fee", 0),
                    "tx_type": tx.get("tx_type", "default"),
                    "congestion": tx.get("congestion", 0.5),
                    "time_of_day": tx.get("time_of_day", 0)
                })
            
            return {
                "transactions": txs_for_refinement,
                "success_rate": self.transaction_success_rate,
                "transaction_count": len(recent_txs)
            }
        except Exception as e:
            logger.error(f"Error getting gas refinement data: {e}")
            return {"error": str(e)}
    
    def check_auto_refinement(self) -> Dict[str, bool]:
        """
        Check if auto-refinement should be performed.
        
        Returns:
            Dictionary indicating which refinements should be performed
        """
        if not self.auto_refinement_enabled:
            return {"risk": False, "gas": False}
        
        current_time = datetime.now().timestamp()
        
        # Check risk refinement
        risk_due = (current_time - self.last_risk_refinement) > (self.risk_refinement_interval_days * 24 * 3600)
        
        # Check gas refinement
        gas_due = (current_time - self.last_gas_refinement) > (self.gas_refinement_interval_days * 24 * 3600)
        
        return {"risk": risk_due, "gas": gas_due}
    
    def record_refinement(self, refinement_type: str) -> None:
        """
        Record that refinement was performed.
        
        Args:
            refinement_type: Type of refinement ("risk" or "gas")
        """
        current_time = datetime.now().timestamp()
        
        if refinement_type == "risk":
            self.last_risk_refinement = current_time
            update_config("last_risk_refinement", str(current_time))
            logger.info(f"Recorded risk refinement at {datetime.fromtimestamp(current_time).isoformat()}")
        elif refinement_type == "gas":
            self.last_gas_refinement = current_time
            update_config("last_gas_refinement", str(current_time))
            logger.info(f"Recorded gas refinement at {datetime.fromtimestamp(current_time).isoformat()}")
    
    def clear_data(self, data_type: Optional[str] = None) -> bool:
        """
        Clear performance data.
        
        Args:
            data_type: Type of data to clear ("trades", "transactions", "portfolio", or None for all)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if data_type == "trades" or data_type is None:
                self.trades = []
                self._save_data(self.trades, self.trades_file, "trades")
            
            if data_type == "transactions" or data_type is None:
                self.transactions = []
                self._save_data(self.transactions, self.transactions_file, "transactions")
            
            if data_type == "portfolio" or data_type is None:
                self.portfolio_metrics = []
                self._save_data(self.portfolio_metrics, self.portfolio_file, "portfolio")
            
            # Reset metrics
            self._calculate_metrics()
            
            logger.info(f"Cleared performance data: {data_type or 'all'}")
            return True
        except Exception as e:
            logger.error(f"Error clearing performance data: {e}")
            return False


# Create a singleton instance
performance_tracker = PerformanceTracker()
