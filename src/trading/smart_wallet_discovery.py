"""
Smart Wallet Discovery System for the Solana Memecoin Trading Bot.
Automatically discovers and ranks profitable wallets for copy trading.
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import requests

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.solana.solana_interact import solana_client
from src.trading.jupiter_api import jupiter_api

# Get logger for this module
logger = get_logger(__name__)


@dataclass
class WalletMetrics:
    """Wallet performance metrics for scoring."""
    address: str
    total_trades: int = 0
    successful_trades: int = 0
    total_volume_sol: float = 0.0
    total_pnl_sol: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_hold_time_hours: float = 0.0
    last_active: datetime = None
    risk_score: float = 0.0
    consistency_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        if self.last_active:
            data['last_active'] = self.last_active.isoformat()
        return data


@dataclass
class WalletScore:
    """Composite wallet score with breakdown."""
    address: str
    total_score: float
    performance_score: float
    consistency_score: float
    risk_score: float
    activity_score: float
    volume_score: float
    confidence_level: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class SmartWalletDiscovery:
    """Automated discovery and ranking of profitable wallets for copy trading."""

    def __init__(self):
        """Initialize the smart wallet discovery system."""
        self.enabled = get_config_value("smart_copy_discovery_enabled", False)
        self.discovery_interval_hours = get_config_value("wallet_discovery_interval_hours", 24)
        self.min_wallet_score = get_config_value("min_wallet_score", 0.7)
        self.min_trades = get_config_value("min_trades_for_discovery", 10)
        self.min_volume_sol = get_config_value("min_volume_for_discovery", 50.0)

        # Discovery criteria
        self.discovery_criteria = {
            'min_trades': self.min_trades,
            'min_success_rate': 0.6,
            'min_profit_ratio': 1.5,
            'max_drawdown': 0.3,
            'min_volume': self.min_volume_sol,
            'min_activity_days': 7
        }

        # Wallet data storage
        self.wallet_metrics = {}  # address -> WalletMetrics
        self.wallet_scores = {}   # address -> WalletScore
        self.discovered_wallets = set()

        # Analysis parameters
        self.analysis_timeframe_days = 30
        self.max_wallets_to_analyze = 1000
        self.score_weights = {
            'performance': 0.3,
            'consistency': 0.25,
            'risk': 0.2,
            'activity': 0.15,
            'volume': 0.1
        }

        # Cache for API calls
        self.transaction_cache = {}
        self.cache_ttl = 3600  # 1 hour

        logger.info("Smart wallet discovery system initialized")

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable smart wallet discovery."""
        self.enabled = enabled
        update_config("smart_copy_discovery_enabled", enabled)
        logger.info(f"Smart wallet discovery {'enabled' if enabled else 'disabled'}")

    async def discover_wallets(self, timeframe_days: int = 30) -> List[Dict]:
        """
        Discover profitable wallets from on-chain data.

        Args:
            timeframe_days: Number of days to analyze

        Returns:
            List of discovered wallet data
        """
        if not self.enabled:
            logger.warning("Smart wallet discovery is disabled")
            return []

        logger.info(f"Starting wallet discovery for {timeframe_days} days")

        try:
            # Step 1: Get active wallets from recent transactions
            active_wallets = await self._get_active_wallets(timeframe_days)
            logger.info(f"Found {len(active_wallets)} active wallets")

            # Step 2: Analyze wallet performance
            analyzed_wallets = []
            for i, wallet_address in enumerate(active_wallets[:self.max_wallets_to_analyze]):
                if i % 100 == 0:
                    logger.info(f"Analyzing wallet {i+1}/{min(len(active_wallets), self.max_wallets_to_analyze)}")

                try:
                    metrics = await self._analyze_wallet_performance(wallet_address, timeframe_days)
                    if metrics and self._meets_discovery_criteria(metrics):
                        analyzed_wallets.append(metrics)
                        self.wallet_metrics[wallet_address] = metrics
                except Exception as e:
                    logger.warning(f"Error analyzing wallet {wallet_address}: {e}")

                # Rate limiting
                await asyncio.sleep(0.1)

            # Step 3: Score and rank wallets
            scored_wallets = []
            for metrics in analyzed_wallets:
                score = self.calculate_wallet_score(metrics)
                if score.total_score >= self.min_wallet_score:
                    scored_wallets.append(score)
                    self.wallet_scores[metrics.address] = score

            # Sort by total score
            scored_wallets.sort(key=lambda x: x.total_score, reverse=True)

            logger.info(f"Discovered {len(scored_wallets)} high-quality wallets")

            # Convert to list of dictionaries
            return [wallet.to_dict() for wallet in scored_wallets]

        except Exception as e:
            logger.error(f"Error in wallet discovery: {e}")
            return []

    async def _get_active_wallets(self, timeframe_days: int) -> List[str]:
        """Get list of active wallets from recent transactions."""
        try:
            # Get recent successful transactions from known DEX programs
            dex_programs = [
                "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium
                "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP",   # Orca
                "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"    # Jupiter
            ]

            active_wallets = set()
            cutoff_time = datetime.now() - timedelta(days=timeframe_days)

            # Use Solscan API to get recent transactions
            for program_id in dex_programs:
                try:
                    url = f"https://api.solscan.io/account/transaction"
                    params = {
                        "account": program_id,
                        "limit": 1000
                    }

                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()

                        for tx in data.get("data", []):
                            tx_time = datetime.fromtimestamp(tx.get("blockTime", 0))
                            if tx_time >= cutoff_time:
                                # Extract wallet addresses from transaction
                                for account in tx.get("parsedInstruction", []):
                                    if "accounts" in account:
                                        for acc in account["accounts"]:
                                            if len(acc) == 44:  # Valid Solana address length
                                                active_wallets.add(acc)

                    await asyncio.sleep(0.5)  # Rate limiting

                except Exception as e:
                    logger.warning(f"Error getting transactions for {program_id}: {e}")

            return list(active_wallets)

        except Exception as e:
            logger.error(f"Error getting active wallets: {e}")
            return []

    async def _analyze_wallet_performance(self, wallet_address: str,
                                        timeframe_days: int) -> Optional[WalletMetrics]:
        """
        Analyze wallet performance metrics.

        Args:
            wallet_address: Wallet address to analyze
            timeframe_days: Analysis timeframe

        Returns:
            WalletMetrics if analysis successful, None otherwise
        """
        try:
            # Get wallet transaction history
            transactions = await self._get_wallet_transactions(wallet_address, timeframe_days)

            if len(transactions) < self.min_trades:
                return None

            # Analyze transactions
            trades = self._parse_trading_transactions(transactions)

            if len(trades) < self.min_trades:
                return None

            # Calculate metrics
            metrics = WalletMetrics(address=wallet_address)

            # Basic trade statistics
            metrics.total_trades = len(trades)
            metrics.successful_trades = sum(1 for trade in trades if trade['pnl'] > 0)
            metrics.win_rate = metrics.successful_trades / metrics.total_trades

            # Financial metrics
            metrics.total_volume_sol = sum(trade['volume_sol'] for trade in trades)
            metrics.total_pnl_sol = sum(trade['pnl'] for trade in trades)

            # Risk metrics
            returns = [trade['return_pct'] for trade in trades]
            if returns:
                metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
                metrics.max_drawdown = self._calculate_max_drawdown(returns)

            # Activity metrics
            hold_times = [trade['hold_time_hours'] for trade in trades if trade['hold_time_hours'] > 0]
            if hold_times:
                metrics.avg_hold_time_hours = np.mean(hold_times)

            # Last activity
            if transactions:
                metrics.last_active = max(tx['timestamp'] for tx in transactions)

            # Risk and consistency scores
            metrics.risk_score = self._calculate_risk_score(trades)
            metrics.consistency_score = self._calculate_consistency_score(trades)

            return metrics

        except Exception as e:
            logger.warning(f"Error analyzing wallet {wallet_address}: {e}")
            return None

    async def _get_wallet_transactions(self, wallet_address: str,
                                     timeframe_days: int) -> List[Dict]:
        """Get wallet transaction history."""
        # Check cache first
        cache_key = f"{wallet_address}_{timeframe_days}"
        if cache_key in self.transaction_cache:
            cache_entry = self.transaction_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                return cache_entry['data']

        try:
            # Use Solscan API to get transaction history
            url = f"https://api.solscan.io/account/transaction"
            params = {
                "account": wallet_address,
                "limit": 1000
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                transactions = []

                cutoff_time = datetime.now() - timedelta(days=timeframe_days)

                for tx in data.get("data", []):
                    tx_time = datetime.fromtimestamp(tx.get("blockTime", 0))
                    if tx_time >= cutoff_time:
                        transactions.append({
                            'signature': tx.get("txHash"),
                            'timestamp': tx_time,
                            'status': tx.get("status"),
                            'fee': tx.get("fee", 0),
                            'instructions': tx.get("parsedInstruction", [])
                        })

                # Cache the result
                self.transaction_cache[cache_key] = {
                    'data': transactions,
                    'timestamp': time.time()
                }

                return transactions

            return []

        except Exception as e:
            logger.warning(f"Error getting transactions for {wallet_address}: {e}")
            return []

    def _parse_trading_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """Parse transactions to extract trading information."""
        trades = []

        try:
            # Group transactions by token pairs to identify trades
            token_positions = defaultdict(list)

            for tx in transactions:
                if tx['status'] != 'Success':
                    continue

                # Parse swap instructions
                for instruction in tx.get('instructions', []):
                    if self._is_swap_instruction(instruction):
                        trade_info = self._extract_trade_info(instruction, tx)
                        if trade_info:
                            token_mint = trade_info['token_mint']
                            token_positions[token_mint].append(trade_info)

            # Calculate P&L for each token
            for token_mint, position_trades in token_positions.items():
                # Sort by timestamp
                position_trades.sort(key=lambda x: x['timestamp'])

                # Calculate P&L for buy/sell pairs
                current_position = 0
                entry_price = 0
                entry_time = None

                for trade in position_trades:
                    if trade['side'] == 'buy':
                        if current_position == 0:
                            entry_price = trade['price']
                            entry_time = trade['timestamp']
                        current_position += trade['amount']

                    elif trade['side'] == 'sell' and current_position > 0:
                        sell_amount = min(trade['amount'], current_position)

                        if entry_price > 0:
                            pnl = (trade['price'] - entry_price) * sell_amount
                            return_pct = ((trade['price'] - entry_price) / entry_price) * 100

                            hold_time_hours = 0
                            if entry_time:
                                hold_time_hours = (trade['timestamp'] - entry_time).total_seconds() / 3600

                            trades.append({
                                'token_mint': token_mint,
                                'entry_price': entry_price,
                                'exit_price': trade['price'],
                                'amount': sell_amount,
                                'pnl': pnl,
                                'return_pct': return_pct,
                                'volume_sol': sell_amount * trade['price'],
                                'hold_time_hours': hold_time_hours,
                                'entry_time': entry_time,
                                'exit_time': trade['timestamp']
                            })

                        current_position -= sell_amount

                        if current_position <= 0:
                            current_position = 0
                            entry_price = 0
                            entry_time = None

            return trades

        except Exception as e:
            logger.warning(f"Error parsing trading transactions: {e}")
            return []

    def _is_swap_instruction(self, instruction: Dict) -> bool:
        """Check if instruction is a swap/trade."""
        program_id = instruction.get('programId', '')
        instruction_type = instruction.get('type', '')

        # Known DEX program IDs and instruction types
        dex_programs = [
            "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium
            "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP",   # Orca
            "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"    # Jupiter
        ]

        swap_types = ['swap', 'swapV2', 'route', 'exchange']

        return program_id in dex_programs or instruction_type.lower() in swap_types

    def _extract_trade_info(self, instruction: Dict, transaction: Dict) -> Optional[Dict]:
        """Extract trade information from swap instruction."""
        try:
            # This is a simplified extraction - in practice, you'd need to parse
            # the specific instruction data for each DEX

            # For now, return a placeholder structure
            return {
                'token_mint': 'placeholder_mint',
                'side': 'buy',  # or 'sell'
                'amount': 1.0,
                'price': 1.0,
                'timestamp': transaction['timestamp']
            }

        except Exception as e:
            logger.warning(f"Error extracting trade info: {e}")
            return None

    def _meets_discovery_criteria(self, metrics: WalletMetrics) -> bool:
        """Check if wallet meets discovery criteria."""
        criteria = self.discovery_criteria

        # Check minimum trades
        if metrics.total_trades < criteria['min_trades']:
            return False

        # Check success rate
        if metrics.win_rate < criteria['min_success_rate']:
            return False

        # Check profit ratio
        if metrics.total_pnl_sol <= 0:
            return False

        total_losses = sum(abs(trade['pnl']) for trade in [] if trade['pnl'] < 0)  # Simplified
        if total_losses > 0:
            profit_ratio = metrics.total_pnl_sol / total_losses
            if profit_ratio < criteria['min_profit_ratio']:
                return False

        # Check maximum drawdown
        if metrics.max_drawdown > criteria['max_drawdown']:
            return False

        # Check minimum volume
        if metrics.total_volume_sol < criteria['min_volume']:
            return False

        # Check activity
        if metrics.last_active:
            days_since_active = (datetime.now() - metrics.last_active).days
            if days_since_active > criteria['min_activity_days']:
                return False

        return True

    def calculate_wallet_score(self, metrics: WalletMetrics) -> WalletScore:
        """
        Calculate composite score for wallet performance.

        Args:
            metrics: Wallet performance metrics

        Returns:
            WalletScore with breakdown
        """
        try:
            # Performance score (0-100)
            performance_score = self._calculate_performance_score(metrics)

            # Consistency score (0-100)
            consistency_score = metrics.consistency_score * 100

            # Risk score (0-100, higher is better/lower risk)
            risk_score = max(0, 100 - (metrics.risk_score * 100))

            # Activity score (0-100)
            activity_score = self._calculate_activity_score(metrics)

            # Volume score (0-100)
            volume_score = self._calculate_volume_score(metrics)

            # Calculate weighted total score
            weights = self.score_weights
            total_score = (
                performance_score * weights['performance'] +
                consistency_score * weights['consistency'] +
                risk_score * weights['risk'] +
                activity_score * weights['activity'] +
                volume_score * weights['volume']
            )

            # Confidence level based on data quality
            confidence_level = self._calculate_confidence_level(metrics)

            return WalletScore(
                address=metrics.address,
                total_score=total_score,
                performance_score=performance_score,
                consistency_score=consistency_score,
                risk_score=risk_score,
                activity_score=activity_score,
                volume_score=volume_score,
                confidence_level=confidence_level
            )

        except Exception as e:
            logger.error(f"Error calculating wallet score: {e}")
            return WalletScore(
                address=metrics.address,
                total_score=0.0,
                performance_score=0.0,
                consistency_score=0.0,
                risk_score=0.0,
                activity_score=0.0,
                volume_score=0.0,
                confidence_level=0.0
            )

    def _calculate_performance_score(self, metrics: WalletMetrics) -> float:
        """Calculate performance score (0-100)."""
        if metrics.total_trades == 0:
            return 0.0

        # Combine win rate, Sharpe ratio, and total return
        win_rate_score = metrics.win_rate * 50  # 0-50 points
        sharpe_score = min(max(metrics.sharpe_ratio * 10, 0), 30)  # 0-30 points

        # Total return score
        if metrics.total_volume_sol > 0:
            return_ratio = metrics.total_pnl_sol / metrics.total_volume_sol
            return_score = min(max(return_ratio * 100, 0), 20)  # 0-20 points
        else:
            return_score = 0

        return win_rate_score + sharpe_score + return_score

    def _calculate_activity_score(self, metrics: WalletMetrics) -> float:
        """Calculate activity score (0-100)."""
        if not metrics.last_active:
            return 0.0

        days_since_active = (datetime.now() - metrics.last_active).days

        # Score decreases with time since last activity
        if days_since_active <= 1:
            return 100.0
        elif days_since_active <= 7:
            return 80.0
        elif days_since_active <= 30:
            return 60.0
        else:
            return max(0, 60 - (days_since_active - 30) * 2)

    def _calculate_volume_score(self, metrics: WalletMetrics) -> float:
        """Calculate volume score (0-100)."""
        # Logarithmic scaling for volume
        if metrics.total_volume_sol <= 0:
            return 0.0

        # Score based on volume tiers
        if metrics.total_volume_sol >= 1000:
            return 100.0
        elif metrics.total_volume_sol >= 500:
            return 80.0
        elif metrics.total_volume_sol >= 100:
            return 60.0
        elif metrics.total_volume_sol >= 50:
            return 40.0
        else:
            return 20.0

    def _calculate_confidence_level(self, metrics: WalletMetrics) -> float:
        """Calculate confidence level in the score (0-1)."""
        # Based on number of trades and data quality
        trade_confidence = min(metrics.total_trades / 50, 1.0)  # Max confidence at 50+ trades

        # Time-based confidence
        if metrics.last_active:
            days_since_active = (datetime.now() - metrics.last_active).days
            time_confidence = max(0, 1 - (days_since_active / 30))
        else:
            time_confidence = 0.0

        return (trade_confidence + time_confidence) / 2

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Assume 0% risk-free rate for simplicity
        return mean_return / std_return

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0.0

        cumulative = np.cumprod([1 + r/100 for r in returns])
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak

        return abs(np.min(drawdown))

    def _calculate_risk_score(self, trades: List[Dict]) -> float:
        """Calculate risk score (0-1, lower is better)."""
        if not trades:
            return 1.0

        # Calculate volatility of returns
        returns = [trade['return_pct'] for trade in trades]
        if len(returns) < 2:
            return 0.5

        volatility = np.std(returns) / 100  # Convert percentage to decimal

        # Normalize volatility to 0-1 scale
        return min(volatility / 0.5, 1.0)  # Assume 50% volatility as maximum

    def _calculate_consistency_score(self, trades: List[Dict]) -> float:
        """Calculate consistency score (0-1)."""
        if not trades:
            return 0.0

        # Calculate consistency based on win streaks and loss streaks
        results = [1 if trade['pnl'] > 0 else 0 for trade in trades]

        if not results:
            return 0.0

        # Calculate streak statistics
        streaks = []
        current_streak = 1

        for i in range(1, len(results)):
            if results[i] == results[i-1]:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        streaks.append(current_streak)

        # Consistency is higher with more moderate streaks
        avg_streak = np.mean(streaks)
        max_streak = max(streaks)

        # Penalize very long streaks (both winning and losing)
        consistency = 1.0 - (max_streak - avg_streak) / len(results)

        return max(0.0, min(1.0, consistency))

    def get_top_wallets(self, limit: int = 20) -> List[Dict]:
        """
        Get top-ranked wallets for copy trading.

        Args:
            limit: Maximum number of wallets to return

        Returns:
            List of top wallet scores
        """
        # Sort wallets by total score
        sorted_wallets = sorted(
            self.wallet_scores.values(),
            key=lambda x: x.total_score,
            reverse=True
        )

        # Return top wallets
        return [wallet.to_dict() for wallet in sorted_wallets[:limit]]

    def get_wallet_details(self, wallet_address: str) -> Optional[Dict]:
        """Get detailed information about a specific wallet."""
        if wallet_address in self.wallet_metrics and wallet_address in self.wallet_scores:
            return {
                'metrics': self.wallet_metrics[wallet_address].to_dict(),
                'score': self.wallet_scores[wallet_address].to_dict()
            }
        return None


# Create a singleton instance
smart_wallet_discovery = SmartWalletDiscovery()
