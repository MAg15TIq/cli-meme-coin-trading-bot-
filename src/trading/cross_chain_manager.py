"""
Cross-Chain Manager - Phase 4C Implementation
Multi-blockchain support and cross-chain arbitrage detection
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from decimal import Decimal
import json
from web3 import Web3
from web3.middleware import geth_poa_middleware
import aiohttp
from eth_account import Account

from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class ChainConfig:
    """Blockchain configuration"""
    chain_id: int
    name: str
    rpc_url: str
    explorer_url: str
    native_token: str
    gas_token: str
    dex_routers: List[str]
    enabled: bool = True

@dataclass
class CrossChainAsset:
    """Cross-chain asset representation"""
    symbol: str
    name: str
    addresses: Dict[str, str]  # chain_name -> contract_address
    decimals: Dict[str, int]   # chain_name -> decimals
    total_supply: Optional[float] = None
    
    def get_address(self, chain: str) -> Optional[str]:
        return self.addresses.get(chain)
    
    def get_decimals(self, chain: str) -> int:
        return self.decimals.get(chain, 18)

@dataclass
class CrossChainPrice:
    """Cross-chain price information"""
    asset: str
    chain: str
    dex: str
    price_usd: float
    liquidity_usd: float
    volume_24h: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ArbitrageOpportunity:
    """Cross-chain arbitrage opportunity"""
    asset: str
    buy_chain: str
    sell_chain: str
    buy_price: float
    sell_price: float
    profit_percentage: float
    profit_usd: float
    max_trade_size: float
    gas_cost_estimate: float
    net_profit: float
    confidence: float
    timestamp: datetime

class ChainConnector:
    """Connector for individual blockchain"""
    
    def __init__(self, config: ChainConfig):
        self.config = config
        self.web3 = None
        self.connected = False
        
    async def connect(self):
        """Connect to blockchain"""
        try:
            self.web3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
            
            # Add PoA middleware for BSC and Polygon
            if self.config.chain_id in [56, 137]:  # BSC, Polygon
                self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Test connection
            if self.web3.is_connected():
                self.connected = True
                logger.info(f"Connected to {self.config.name} (Chain ID: {self.config.chain_id})")
            else:
                logger.error(f"Failed to connect to {self.config.name}")
                
        except Exception as e:
            logger.error(f"Error connecting to {self.config.name}: {e}")
            self.connected = False
    
    async def get_token_balance(self, token_address: str, wallet_address: str) -> float:
        """Get token balance for wallet"""
        if not self.connected:
            return 0.0
        
        try:
            # ERC-20 ABI for balanceOf
            erc20_abi = [
                {
                    "constant": True,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"name": "", "type": "uint8"}],
                    "type": "function"
                }
            ]
            
            contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=erc20_abi
            )
            
            balance = contract.functions.balanceOf(
                Web3.to_checksum_address(wallet_address)
            ).call()
            
            decimals = contract.functions.decimals().call()
            
            return balance / (10 ** decimals)
            
        except Exception as e:
            logger.error(f"Error getting token balance on {self.config.name}: {e}")
            return 0.0
    
    async def get_gas_price(self) -> float:
        """Get current gas price"""
        if not self.connected:
            return 0.0
        
        try:
            gas_price_wei = self.web3.eth.gas_price
            gas_price_gwei = self.web3.from_wei(gas_price_wei, 'gwei')
            return float(gas_price_gwei)
            
        except Exception as e:
            logger.error(f"Error getting gas price on {self.config.name}: {e}")
            return 0.0
    
    async def estimate_swap_gas(self, token_in: str, token_out: str, amount: float) -> float:
        """Estimate gas cost for token swap"""
        # Simplified estimation - would integrate with actual DEX contracts
        base_gas = 150000  # Base gas for swap
        gas_price = await self.get_gas_price()
        
        # Convert to USD (simplified)
        if self.config.chain_id == 1:  # Ethereum
            eth_price = 2000  # Mock ETH price
            gas_cost_usd = (base_gas * gas_price * eth_price) / (10**9)
        elif self.config.chain_id == 56:  # BSC
            bnb_price = 300  # Mock BNB price
            gas_cost_usd = (base_gas * gas_price * bnb_price) / (10**9)
        elif self.config.chain_id == 137:  # Polygon
            matic_price = 1  # Mock MATIC price
            gas_cost_usd = (base_gas * gas_price * matic_price) / (10**9)
        else:
            gas_cost_usd = 5.0  # Default estimate
        
        return gas_cost_usd

class PriceAggregator:
    """Aggregates prices from multiple chains and DEXes"""
    
    def __init__(self):
        self.price_cache = {}
        self.cache_duration = 30  # seconds
        
        # DEX APIs for different chains
        self.dex_apis = {
            'ethereum': {
                'uniswap': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
                'sushiswap': 'https://api.sushi.com/v1/pools'
            },
            'bsc': {
                'pancakeswap': 'https://api.pancakeswap.info/api/v2/tokens',
                'biswap': 'https://api.biswap.org/api/v1/pools'
            },
            'polygon': {
                'quickswap': 'https://api.quickswap.exchange/v1/pools',
                'sushiswap': 'https://api.sushi.com/v1/pools'
            }
        }
    
    async def get_prices(self, asset: CrossChainAsset) -> List[CrossChainPrice]:
        """Get prices for asset across all chains"""
        prices = []
        
        for chain_name, address in asset.addresses.items():
            if address:
                chain_prices = await self._get_chain_prices(asset.symbol, chain_name, address)
                prices.extend(chain_prices)
        
        return prices
    
    async def _get_chain_prices(self, symbol: str, chain: str, address: str) -> List[CrossChainPrice]:
        """Get prices for asset on specific chain"""
        prices = []
        
        # Check cache first
        cache_key = f"{symbol}_{chain}"
        if cache_key in self.price_cache:
            cached_data, timestamp = self.price_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_duration:
                return cached_data
        
        try:
            if chain in self.dex_apis:
                for dex_name, api_url in self.dex_apis[chain].items():
                    price_data = await self._fetch_dex_price(api_url, dex_name, symbol, address)
                    if price_data:
                        prices.append(CrossChainPrice(
                            asset=symbol,
                            chain=chain,
                            dex=dex_name,
                            price_usd=price_data['price'],
                            liquidity_usd=price_data['liquidity'],
                            volume_24h=price_data['volume'],
                            timestamp=datetime.now()
                        ))
            
            # Cache results
            self.price_cache[cache_key] = (prices, datetime.now())
            
        except Exception as e:
            logger.error(f"Error getting prices for {symbol} on {chain}: {e}")
        
        return prices
    
    async def _fetch_dex_price(self, api_url: str, dex: str, symbol: str, address: str) -> Optional[Dict]:
        """Fetch price from DEX API"""
        try:
            async with aiohttp.ClientSession() as session:
                # This would be implemented based on each DEX's API
                # For now, return mock data
                return {
                    'price': 100.0 + hash(symbol) % 50,  # Mock price
                    'liquidity': 1000000.0,
                    'volume': 500000.0
                }
                
        except Exception as e:
            logger.error(f"Error fetching price from {dex}: {e}")
            return None

class ArbitrageDetector:
    """Detects cross-chain arbitrage opportunities"""
    
    def __init__(self, min_profit_percentage: float = 1.0):
        self.min_profit_percentage = min_profit_percentage
        self.bridge_costs = {
            ('ethereum', 'bsc'): 25.0,      # USD
            ('ethereum', 'polygon'): 15.0,
            ('bsc', 'polygon'): 5.0,
        }
    
    async def find_opportunities(self, prices: List[CrossChainPrice]) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities from price data"""
        opportunities = []
        
        # Group prices by asset
        asset_prices = {}
        for price in prices:
            if price.asset not in asset_prices:
                asset_prices[price.asset] = []
            asset_prices[price.asset].append(price)
        
        # Find arbitrage opportunities for each asset
        for asset, price_list in asset_prices.items():
            asset_opportunities = await self._find_asset_arbitrage(asset, price_list)
            opportunities.extend(asset_opportunities)
        
        # Sort by profit percentage
        opportunities.sort(key=lambda x: x.profit_percentage, reverse=True)
        
        return opportunities
    
    async def _find_asset_arbitrage(self, asset: str, prices: List[CrossChainPrice]) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities for a specific asset"""
        opportunities = []
        
        # Compare all price pairs
        for i, buy_price in enumerate(prices):
            for j, sell_price in enumerate(prices):
                if i != j and buy_price.chain != sell_price.chain:
                    opportunity = await self._calculate_arbitrage(
                        asset, buy_price, sell_price
                    )
                    
                    if (opportunity and 
                        opportunity.profit_percentage >= self.min_profit_percentage):
                        opportunities.append(opportunity)
        
        return opportunities
    
    async def _calculate_arbitrage(self, asset: str, buy_price: CrossChainPrice, 
                                 sell_price: CrossChainPrice) -> Optional[ArbitrageOpportunity]:
        """Calculate arbitrage opportunity between two prices"""
        try:
            # Basic profit calculation
            profit_percentage = ((sell_price.price_usd - buy_price.price_usd) / 
                               buy_price.price_usd) * 100
            
            if profit_percentage <= 0:
                return None
            
            # Estimate maximum trade size based on liquidity
            max_trade_size = min(
                buy_price.liquidity_usd * 0.1,  # 10% of buy liquidity
                sell_price.liquidity_usd * 0.1   # 10% of sell liquidity
            )
            
            profit_usd = max_trade_size * (profit_percentage / 100)
            
            # Estimate costs
            bridge_cost = self._get_bridge_cost(buy_price.chain, sell_price.chain)
            gas_cost = 50.0  # Simplified gas cost estimate
            total_costs = bridge_cost + gas_cost
            
            net_profit = profit_usd - total_costs
            
            # Calculate confidence based on liquidity and volume
            confidence = min(
                (buy_price.liquidity_usd + sell_price.liquidity_usd) / 2000000,  # Liquidity factor
                (buy_price.volume_24h + sell_price.volume_24h) / 1000000,        # Volume factor
                1.0
            )
            
            return ArbitrageOpportunity(
                asset=asset,
                buy_chain=buy_price.chain,
                sell_chain=sell_price.chain,
                buy_price=buy_price.price_usd,
                sell_price=sell_price.price_usd,
                profit_percentage=profit_percentage,
                profit_usd=profit_usd,
                max_trade_size=max_trade_size,
                gas_cost_estimate=total_costs,
                net_profit=net_profit,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating arbitrage for {asset}: {e}")
            return None
    
    def _get_bridge_cost(self, from_chain: str, to_chain: str) -> float:
        """Get estimated bridge cost between chains"""
        bridge_key = (from_chain, to_chain)
        reverse_key = (to_chain, from_chain)
        
        return self.bridge_costs.get(bridge_key, 
               self.bridge_costs.get(reverse_key, 20.0))  # Default cost

class CrossChainManager:
    """Main cross-chain management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.chains = {}
        self.connectors = {}
        self.price_aggregator = PriceAggregator()
        self.arbitrage_detector = ArbitrageDetector(
            min_profit_percentage=config.get('min_arbitrage_profit_percentage', 1.0)
        )
        
        # Supported assets
        self.assets = {}
        self.running = False
        
        # Initialize chain configurations
        self._initialize_chains()
        
    def _initialize_chains(self):
        """Initialize supported blockchain configurations"""
        self.chains = {
            'ethereum': ChainConfig(
                chain_id=1,
                name='Ethereum',
                rpc_url='https://mainnet.infura.io/v3/YOUR_PROJECT_ID',
                explorer_url='https://etherscan.io',
                native_token='ETH',
                gas_token='ETH',
                dex_routers=['0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D'],  # Uniswap V2
                enabled=self.config.get('ethereum_enabled', False)
            ),
            'bsc': ChainConfig(
                chain_id=56,
                name='Binance Smart Chain',
                rpc_url='https://bsc-dataseed1.binance.org/',
                explorer_url='https://bscscan.com',
                native_token='BNB',
                gas_token='BNB',
                dex_routers=['0x10ED43C718714eb63d5aA57B78B54704E256024E'],  # PancakeSwap
                enabled=self.config.get('bsc_enabled', False)
            ),
            'polygon': ChainConfig(
                chain_id=137,
                name='Polygon',
                rpc_url='https://polygon-rpc.com/',
                explorer_url='https://polygonscan.com',
                native_token='MATIC',
                gas_token='MATIC',
                dex_routers=['0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff'],  # QuickSwap
                enabled=self.config.get('polygon_enabled', False)
            )
        }
        
        # Initialize example cross-chain assets
        self._initialize_assets()
    
    def _initialize_assets(self):
        """Initialize cross-chain asset configurations"""
        self.assets = {
            'USDC': CrossChainAsset(
                symbol='USDC',
                name='USD Coin',
                addresses={
                    'ethereum': '0xA0b86a33E6441b8435b662303c0f218C8c7c8e8e',
                    'bsc': '0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d',
                    'polygon': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174'
                },
                decimals={
                    'ethereum': 6,
                    'bsc': 18,
                    'polygon': 6
                }
            ),
            'USDT': CrossChainAsset(
                symbol='USDT',
                name='Tether USD',
                addresses={
                    'ethereum': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
                    'bsc': '0x55d398326f99059fF775485246999027B3197955',
                    'polygon': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F'
                },
                decimals={
                    'ethereum': 6,
                    'bsc': 18,
                    'polygon': 6
                }
            )
        }
    
    async def start(self):
        """Start cross-chain monitoring"""
        self.running = True
        logger.info("Starting Cross-Chain Manager...")
        
        # Connect to enabled chains
        for chain_name, chain_config in self.chains.items():
            if chain_config.enabled:
                connector = ChainConnector(chain_config)
                await connector.connect()
                self.connectors[chain_name] = connector
        
        # Start monitoring tasks
        monitoring_task = asyncio.create_task(self._monitor_arbitrage())
        
        await monitoring_task
    
    async def stop(self):
        """Stop cross-chain monitoring"""
        self.running = False
        logger.info("Cross-Chain Manager stopped")
    
    async def _monitor_arbitrage(self):
        """Monitor for arbitrage opportunities"""
        while self.running:
            try:
                # Get prices for all assets
                all_prices = []
                for asset in self.assets.values():
                    asset_prices = await self.price_aggregator.get_prices(asset)
                    all_prices.extend(asset_prices)
                
                # Find arbitrage opportunities
                opportunities = await self.arbitrage_detector.find_opportunities(all_prices)
                
                # Log significant opportunities
                for opp in opportunities[:5]:  # Top 5 opportunities
                    if opp.net_profit > 100:  # Minimum $100 profit
                        logger.info(
                            f"Arbitrage opportunity: {opp.asset} "
                            f"{opp.buy_chain} -> {opp.sell_chain} "
                            f"Profit: {opp.profit_percentage:.2f}% "
                            f"(${opp.net_profit:.2f})"
                        )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring arbitrage: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def get_portfolio_balance(self, wallet_address: str) -> Dict:
        """Get cross-chain portfolio balance"""
        portfolio = {}
        
        for chain_name, connector in self.connectors.items():
            if connector.connected:
                chain_balances = {}
                
                for asset_symbol, asset in self.assets.items():
                    token_address = asset.get_address(chain_name)
                    if token_address:
                        balance = await connector.get_token_balance(token_address, wallet_address)
                        if balance > 0:
                            chain_balances[asset_symbol] = balance
                
                if chain_balances:
                    portfolio[chain_name] = chain_balances
        
        return portfolio
    
    async def get_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get current arbitrage opportunities"""
        all_prices = []
        for asset in self.assets.values():
            asset_prices = await self.price_aggregator.get_prices(asset)
            all_prices.extend(asset_prices)
        
        return await self.arbitrage_detector.find_opportunities(all_prices)
    
    def get_supported_chains(self) -> List[str]:
        """Get list of supported chains"""
        return [name for name, config in self.chains.items() if config.enabled]
    
    def get_supported_assets(self) -> List[str]:
        """Get list of supported cross-chain assets"""
        return list(self.assets.keys())

# Global instance
cross_chain_manager = None

def get_cross_chain_manager(config: Dict = None) -> CrossChainManager:
    """Get or create cross-chain manager instance"""
    global cross_chain_manager
    if cross_chain_manager is None and config:
        cross_chain_manager = CrossChainManager(config)
    return cross_chain_manager
