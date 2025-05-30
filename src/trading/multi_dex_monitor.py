"""
Multi-DEX Monitor for the Solana Memecoin Trading Bot.
Monitors multiple DEXes for liquidity opportunities and arbitrage.
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
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
class PoolInfo:
    """Information about a liquidity pool."""
    pool_address: str
    dex_name: str
    token_a: str
    token_b: str
    token_a_symbol: str = ""
    token_b_symbol: str = ""
    liquidity_sol: float = 0.0
    volume_24h_sol: float = 0.0
    fee_percentage: float = 0.0
    price: float = 0.0
    price_impact_1pct: float = 0.0
    created_at: datetime = None
    last_updated: datetime = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.last_updated:
            data['last_updated'] = self.last_updated.isoformat()
        return data


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity between DEXes."""
    token_mint: str
    token_symbol: str
    buy_dex: str
    sell_dex: str
    buy_price: float
    sell_price: float
    price_difference: float
    profit_percentage: float
    estimated_profit_sol: float
    max_trade_size_sol: float
    detected_at: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['detected_at'] = self.detected_at.isoformat()
        return data


class MultiDEXMonitor:
    """Monitor multiple DEXes for liquidity opportunities."""

    def __init__(self):
        """Initialize the multi-DEX monitor."""
        self.enabled = get_config_value("multi_dex_enabled", False)
        self.arbitrage_enabled = get_config_value("arbitrage_detection_enabled", False)
        self.min_arbitrage_profit_bps = get_config_value("min_arbitrage_profit_bps", 50)  # 0.5%

        # DEX configurations
        self.dex_configs = {
            'raydium': {
                'program_id': '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',
                'api_url': 'https://api.raydium.io/v2',
                'enabled': True
            },
            'orca': {
                'program_id': '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP',
                'api_url': 'https://api.orca.so',
                'enabled': True
            },
            'jupiter': {
                'program_id': 'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4',
                'api_url': 'https://quote-api.jup.ag/v6',
                'enabled': True
            },
            'meteora': {
                'program_id': 'Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB',
                'api_url': 'https://app.meteora.ag/api',
                'enabled': True
            }
        }

        # Data storage
        self.pools = {}  # pool_address -> PoolInfo
        self.arbitrage_opportunities = []
        self.monitoring_stats = {
            'pools_discovered': 0,
            'arbitrage_opportunities': 0,
            'last_scan_time': None,
            'scan_duration_seconds': 0
        }

        # Monitoring control
        self._monitoring_thread = None
        self._stop_monitoring = False
        self.scan_interval_seconds = 60  # Scan every minute

        # Cache for API calls
        self.price_cache = {}
        self.cache_ttl = 30  # 30 seconds

        logger.info("Multi-DEX monitor initialized")

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable multi-DEX monitoring."""
        self.enabled = enabled
        update_config("multi_dex_enabled", enabled)

        if enabled:
            self.start_monitoring()
        else:
            self.stop_monitoring()

        logger.info(f"Multi-DEX monitoring {'enabled' if enabled else 'disabled'}")

    def start_monitoring(self) -> None:
        """Start monitoring all enabled DEXes."""
        if not self.enabled:
            logger.warning("Multi-DEX monitoring is disabled")
            return

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Multi-DEX monitoring is already running")
            return

        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()

        logger.info("Multi-DEX monitoring started")

    def stop_monitoring(self) -> None:
        """Stop monitoring all DEXes."""
        self._stop_monitoring = True

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)

        logger.info("Multi-DEX monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring:
            try:
                start_time = time.time()

                # Scan all enabled DEXes
                asyncio.run(self._scan_all_dexes())

                # Detect arbitrage opportunities if enabled
                if self.arbitrage_enabled:
                    self._detect_arbitrage_opportunities()

                # Update statistics
                scan_duration = time.time() - start_time
                self.monitoring_stats['last_scan_time'] = datetime.now()
                self.monitoring_stats['scan_duration_seconds'] = scan_duration

                logger.debug(f"DEX scan completed in {scan_duration:.2f} seconds")

                # Wait for next scan
                time.sleep(self.scan_interval_seconds)

            except Exception as e:
                logger.error(f"Error in multi-DEX monitoring loop: {e}")
                time.sleep(10)  # Wait before retrying

    async def _scan_all_dexes(self) -> None:
        """Scan all enabled DEXes for pools."""
        tasks = []

        for dex_name, config in self.dex_configs.items():
            if config.get('enabled', False):
                tasks.append(self._scan_dex(dex_name, config))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            total_pools = 0
            for i, result in enumerate(results):
                dex_name = list(self.dex_configs.keys())[i]
                if isinstance(result, Exception):
                    logger.warning(f"Error scanning {dex_name}: {result}")
                else:
                    total_pools += len(result)
                    logger.debug(f"Found {len(result)} pools on {dex_name}")

            self.monitoring_stats['pools_discovered'] = total_pools

    async def _scan_dex(self, dex_name: str, config: Dict) -> List[PoolInfo]:
        """Scan a specific DEX for pools."""
        try:
            if dex_name == 'raydium':
                return await self._scan_raydium()
            elif dex_name == 'orca':
                return await self._scan_orca()
            elif dex_name == 'jupiter':
                return await self._scan_jupiter()
            elif dex_name == 'meteora':
                return await self._scan_meteora()
            else:
                logger.warning(f"Unknown DEX: {dex_name}")
                return []

        except Exception as e:
            logger.error(f"Error scanning {dex_name}: {e}")
            return []

    async def _scan_raydium(self) -> List[PoolInfo]:
        """Scan Raydium for pools."""
        try:
            # Use Raydium API to get pool information
            url = "https://api.raydium.io/v2/sdk/liquidity/mainnet.json"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pools = []

                for pool_data in data.get('official', []):
                    pool_info = PoolInfo(
                        pool_address=pool_data.get('id', ''),
                        dex_name='raydium',
                        token_a=pool_data.get('baseMint', ''),
                        token_b=pool_data.get('quoteMint', ''),
                        token_a_symbol=pool_data.get('baseSymbol', ''),
                        token_b_symbol=pool_data.get('quoteSymbol', ''),
                        liquidity_sol=float(pool_data.get('liquidity', 0)),
                        volume_24h_sol=float(pool_data.get('volume24h', 0)),
                        last_updated=datetime.now()
                    )

                    pools.append(pool_info)
                    self.pools[pool_info.pool_address] = pool_info

                return pools

            return []

        except Exception as e:
            logger.error(f"Error scanning Raydium: {e}")
            return []

    async def _scan_orca(self) -> List[PoolInfo]:
        """Scan Orca for pools."""
        try:
            # Use Orca API to get pool information
            url = "https://api.orca.so/v1/whirlpool/list"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pools = []

                for pool_data in data.get('whirlpools', []):
                    pool_info = PoolInfo(
                        pool_address=pool_data.get('address', ''),
                        dex_name='orca',
                        token_a=pool_data.get('tokenA', {}).get('mint', ''),
                        token_b=pool_data.get('tokenB', {}).get('mint', ''),
                        token_a_symbol=pool_data.get('tokenA', {}).get('symbol', ''),
                        token_b_symbol=pool_data.get('tokenB', {}).get('symbol', ''),
                        liquidity_sol=float(pool_data.get('tvl', 0)),
                        volume_24h_sol=float(pool_data.get('volume24h', 0)),
                        fee_percentage=float(pool_data.get('feeRate', 0)) / 10000,  # Convert from basis points
                        last_updated=datetime.now()
                    )

                    pools.append(pool_info)
                    self.pools[pool_info.pool_address] = pool_info

                return pools

            return []

        except Exception as e:
            logger.error(f"Error scanning Orca: {e}")
            return []

    async def _scan_jupiter(self) -> List[PoolInfo]:
        """Scan Jupiter for routing information."""
        try:
            # Jupiter is primarily a router, so we get token price information
            # rather than specific pools
            url = "https://price.jup.ag/v4/price"
            params = {"ids": "SOL"}  # Get SOL price as reference

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Jupiter data would be processed differently
                # For now, return empty list as it's primarily a router
                return []

            return []

        except Exception as e:
            logger.error(f"Error scanning Jupiter: {e}")
            return []

    async def _scan_meteora(self) -> List[PoolInfo]:
        """Scan Meteora for pools."""
        try:
            # Meteora API endpoint (this is a placeholder - actual API may differ)
            url = "https://app.meteora.ag/api/pools"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pools = []

                for pool_data in data.get('pools', []):
                    pool_info = PoolInfo(
                        pool_address=pool_data.get('address', ''),
                        dex_name='meteora',
                        token_a=pool_data.get('tokenA', ''),
                        token_b=pool_data.get('tokenB', ''),
                        liquidity_sol=float(pool_data.get('liquidity', 0)),
                        volume_24h_sol=float(pool_data.get('volume24h', 0)),
                        last_updated=datetime.now()
                    )

                    pools.append(pool_info)
                    self.pools[pool_info.pool_address] = pool_info

                return pools

            return []

        except Exception as e:
            logger.error(f"Error scanning Meteora: {e}")
            return []

    def _detect_arbitrage_opportunities(self) -> None:
        """Detect arbitrage opportunities between DEXes."""
        try:
            # Group pools by token pair
            token_pairs = defaultdict(list)

            for pool in self.pools.values():
                # Create standardized token pair key (always put smaller mint first)
                token_a, token_b = sorted([pool.token_a, pool.token_b])
                pair_key = f"{token_a}-{token_b}"
                token_pairs[pair_key].append(pool)

            # Find arbitrage opportunities
            new_opportunities = []

            for pair_key, pair_pools in token_pairs.items():
                if len(pair_pools) < 2:
                    continue

                # Calculate prices for each pool
                pool_prices = []
                for pool in pair_pools:
                    price = self._calculate_pool_price(pool)
                    if price > 0:
                        pool_prices.append((pool, price))

                if len(pool_prices) < 2:
                    continue

                # Find price discrepancies
                pool_prices.sort(key=lambda x: x[1])  # Sort by price

                lowest_price_pool, lowest_price = pool_prices[0]
                highest_price_pool, highest_price = pool_prices[-1]

                # Calculate arbitrage metrics
                price_difference = highest_price - lowest_price
                profit_percentage = (price_difference / lowest_price) * 100

                # Check if opportunity meets minimum criteria
                if profit_percentage >= (self.min_arbitrage_profit_bps / 100):
                    # Estimate maximum trade size and profit
                    max_trade_size = min(
                        lowest_price_pool.liquidity_sol * 0.1,  # Max 10% of liquidity
                        highest_price_pool.liquidity_sol * 0.1
                    )

                    estimated_profit = max_trade_size * (profit_percentage / 100)

                    opportunity = ArbitrageOpportunity(
                        token_mint=lowest_price_pool.token_a,  # Assuming token_a is the traded token
                        token_symbol=lowest_price_pool.token_a_symbol,
                        buy_dex=lowest_price_pool.dex_name,
                        sell_dex=highest_price_pool.dex_name,
                        buy_price=lowest_price,
                        sell_price=highest_price,
                        price_difference=price_difference,
                        profit_percentage=profit_percentage,
                        estimated_profit_sol=estimated_profit,
                        max_trade_size_sol=max_trade_size,
                        detected_at=datetime.now()
                    )

                    new_opportunities.append(opportunity)

            # Update opportunities list (keep only recent ones)
            cutoff_time = datetime.now() - timedelta(minutes=10)
            self.arbitrage_opportunities = [
                opp for opp in self.arbitrage_opportunities
                if opp.detected_at >= cutoff_time
            ]

            # Add new opportunities
            self.arbitrage_opportunities.extend(new_opportunities)

            # Update statistics
            self.monitoring_stats['arbitrage_opportunities'] = len(self.arbitrage_opportunities)

            if new_opportunities:
                logger.info(f"Found {len(new_opportunities)} new arbitrage opportunities")

        except Exception as e:
            logger.error(f"Error detecting arbitrage opportunities: {e}")

    def _calculate_pool_price(self, pool: PoolInfo) -> float:
        """Calculate the current price for a pool."""
        try:
            # This is a simplified price calculation
            # In practice, you'd need to query the actual pool state

            # For now, use cached price or fetch from Jupiter
            cache_key = f"{pool.token_a}_{pool.token_b}"

            if cache_key in self.price_cache:
                cache_entry = self.price_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    return cache_entry['price']

            # Try to get price from Jupiter API
            try:
                price = jupiter_api.get_token_price(pool.token_a)

                # Cache the price
                self.price_cache[cache_key] = {
                    'price': price,
                    'timestamp': time.time()
                }

                return price

            except Exception:
                # Fallback to pool data if available
                if pool.price > 0:
                    return pool.price

                return 0.0

        except Exception as e:
            logger.warning(f"Error calculating pool price: {e}")
            return 0.0

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status for all DEXes."""
        status = {}

        for dex_name, config in self.dex_configs.items():
            dex_pools = [pool for pool in self.pools.values() if pool.dex_name == dex_name]

            status[dex_name] = {
                'enabled': config.get('enabled', False),
                'active': self.enabled and config.get('enabled', False),
                'pools_found': len(dex_pools),
                'opportunities': len([
                    opp for opp in self.arbitrage_opportunities
                    if opp.buy_dex == dex_name or opp.sell_dex == dex_name
                ])
            }

        return status

    def get_arbitrage_opportunities(self, limit: int = 10) -> List[Dict]:
        """Get current arbitrage opportunities."""
        # Sort by profit percentage (highest first)
        sorted_opportunities = sorted(
            self.arbitrage_opportunities,
            key=lambda x: x.profit_percentage,
            reverse=True
        )

        return [opp.to_dict() for opp in sorted_opportunities[:limit]]

    def get_pool_info(self, pool_address: str) -> Optional[Dict]:
        """Get information about a specific pool."""
        if pool_address in self.pools:
            return self.pools[pool_address].to_dict()
        return None

    def get_pools_by_token(self, token_mint: str) -> List[Dict]:
        """Get all pools containing a specific token."""
        matching_pools = []

        for pool in self.pools.values():
            if pool.token_a == token_mint or pool.token_b == token_mint:
                matching_pools.append(pool.to_dict())

        return matching_pools

    def get_best_price_for_token(self, token_mint: str, side: str = 'buy') -> Optional[Dict]:
        """Get the best price for a token across all DEXes."""
        token_pools = self.get_pools_by_token(token_mint)

        if not token_pools:
            return None

        best_pool = None
        best_price = None

        for pool_data in token_pools:
            pool = PoolInfo(**pool_data)
            price = self._calculate_pool_price(pool)

            if price > 0:
                if best_price is None:
                    best_price = price
                    best_pool = pool_data
                else:
                    # For buy orders, we want the lowest price
                    # For sell orders, we want the highest price
                    if (side == 'buy' and price < best_price) or (side == 'sell' and price > best_price):
                        best_price = price
                        best_pool = pool_data

        if best_pool:
            return {
                'pool': best_pool,
                'price': best_price,
                'side': side
            }

        return None

    def execute_arbitrage(self, opportunity: ArbitrageOpportunity, amount_sol: float) -> Dict[str, Any]:
        """Execute an arbitrage opportunity (placeholder for future implementation)."""
        logger.info(f"Arbitrage execution requested: {opportunity.token_symbol} "
                   f"({opportunity.buy_dex} -> {opportunity.sell_dex}) "
                   f"for {amount_sol} SOL")

        # This would implement the actual arbitrage execution
        # For now, return a placeholder response
        return {
            'success': False,
            'message': 'Arbitrage execution not yet implemented',
            'opportunity': opportunity.to_dict(),
            'requested_amount': amount_sol
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            **self.monitoring_stats,
            'total_pools': len(self.pools),
            'active_opportunities': len(self.arbitrage_opportunities),
            'enabled_dexes': sum(1 for config in self.dex_configs.values() if config.get('enabled', False)),
            'monitoring_active': self.enabled and self._monitoring_thread and self._monitoring_thread.is_alive()
        }


# Create a singleton instance
multi_dex_monitor = MultiDEXMonitor()
