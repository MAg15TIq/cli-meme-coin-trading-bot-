"""
Chart generator module for the Solana Memecoin Trading Bot.
Generates charts for technical analysis and price visualization.
"""

import json
import logging
import os
import io
import base64
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

from config import get_config_value, update_config
from src.trading.technical_analysis import technical_analyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generator for technical analysis charts."""
    
    def __init__(self):
        """Initialize the chart generator."""
        self.enabled = get_config_value("charts_enabled", False)
        self.chart_dir = Path(get_config_value("chart_dir", "charts"))
        self.chart_format = get_config_value("chart_format", "ascii")  # ascii, png, svg
        
        # Create chart directory if it doesn't exist
        self.chart_dir.mkdir(parents=True, exist_ok=True)
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable chart generation.
        
        Args:
            enabled: Whether chart generation should be enabled
        """
        self.enabled = enabled
        update_config("charts_enabled", enabled)
        logger.info(f"Chart generation {'enabled' if enabled else 'disabled'}")
    
    def set_chart_format(self, chart_format: str) -> None:
        """
        Set the chart format.
        
        Args:
            chart_format: Chart format (ascii, png, svg)
        """
        if chart_format not in ["ascii", "png", "svg"]:
            logger.warning(f"Invalid chart format: {chart_format}")
            return
        
        self.chart_format = chart_format
        update_config("chart_format", chart_format)
        logger.info(f"Chart format set to {chart_format}")
    
    def generate_price_chart(self, token_mint: str, time_range: str = "1d") -> Optional[str]:
        """
        Generate a price chart for a token.
        
        Args:
            token_mint: The token mint address
            time_range: Time range (1h, 4h, 1d, 1w, 1m)
            
        Returns:
            Path to the generated chart or ASCII chart string
        """
        if not self.enabled:
            logger.warning("Chart generation is disabled")
            return None
        
        # Get price data
        price_data = technical_analyzer.price_data.get(token_mint)
        if price_data is None or len(price_data) == 0:
            logger.warning(f"No price data available for {token_mint}")
            return None
        
        # Filter data by time range
        end_time = datetime.now()
        if time_range == "1h":
            start_time = end_time - timedelta(hours=1)
        elif time_range == "4h":
            start_time = end_time - timedelta(hours=4)
        elif time_range == "1d":
            start_time = end_time - timedelta(days=1)
        elif time_range == "1w":
            start_time = end_time - timedelta(weeks=1)
        elif time_range == "1m":
            start_time = end_time - timedelta(days=30)
        else:
            logger.warning(f"Invalid time range: {time_range}")
            start_time = end_time - timedelta(days=1)
        
        # Filter data
        filtered_data = price_data[price_data['timestamp'] >= start_time]
        
        if len(filtered_data) == 0:
            logger.warning(f"No price data available for {token_mint} in time range {time_range}")
            return None
        
        # Get token info
        tracked_tokens = get_config_value("tracked_tokens", {})
        token_info = tracked_tokens.get(token_mint, {})
        token_symbol = token_info.get("symbol", token_mint[:6])
        
        # Generate chart based on format
        if self.chart_format == "ascii":
            return self._generate_ascii_chart(filtered_data, token_symbol, time_range)
        elif self.chart_format == "png" or self.chart_format == "svg":
            return self._generate_image_chart(filtered_data, token_symbol, time_range, token_mint)
        else:
            logger.warning(f"Unsupported chart format: {self.chart_format}")
            return None
    
    def _generate_ascii_chart(self, data: pd.DataFrame, token_symbol: str, time_range: str) -> str:
        """
        Generate an ASCII chart.
        
        Args:
            data: Price data
            token_symbol: Token symbol
            time_range: Time range
            
        Returns:
            ASCII chart string
        """
        # Constants
        width = 80
        height = 20
        
        # Extract price data
        prices = data['price'].values
        timestamps = data['timestamp'].values
        
        # Calculate min and max prices
        min_price = np.min(prices)
        max_price = np.max(prices)
        price_range = max_price - min_price
        
        if price_range == 0:
            price_range = max_price * 0.1  # 10% of max price if range is zero
        
        # Create empty chart
        chart = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot prices
        for i, price in enumerate(prices):
            # Calculate x position
            x = int((i / len(prices)) * (width - 1))
            
            # Calculate y position
            y = int(((price - min_price) / price_range) * (height - 1))
            y = height - 1 - y  # Invert y-axis
            
            # Ensure within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            # Plot point
            chart[y][x] = '█'
        
        # Convert chart to string
        chart_str = f"{token_symbol} - {time_range} Chart\n"
        chart_str += f"Max: {max_price:.8f}\n"
        
        for row in chart:
            chart_str += ''.join(row) + '\n'
        
        chart_str += f"Min: {min_price:.8f}\n"
        chart_str += f"Last: {prices[-1]:.8f}\n"
        
        return chart_str
    
    def _generate_image_chart(self, data: pd.DataFrame, token_symbol: str, 
                             time_range: str, token_mint: str) -> Optional[str]:
        """
        Generate an image chart.
        
        Args:
            data: Price data
            token_symbol: Token symbol
            time_range: Time range
            token_mint: Token mint address
            
        Returns:
            Path to the generated chart
        """
        try:
            # This is a placeholder for actual chart generation
            # In a real implementation, we would use a library like matplotlib
            
            # For now, just create a text file with chart data
            chart_file = self.chart_dir / f"{token_symbol}_{time_range}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            
            with open(chart_file, 'w') as f:
                f.write(f"{token_symbol} - {time_range} Chart\n\n")
                f.write(f"Data points: {len(data)}\n")
                f.write(f"Time range: {data['timestamp'].min()} to {data['timestamp'].max()}\n")
                f.write(f"Price range: {data['price'].min():.8f} to {data['price'].max():.8f}\n")
                f.write(f"Current price: {data['price'].iloc[-1]:.8f}\n")
                
                # Add indicators if available
                indicators = technical_analyzer.get_token_indicators(token_mint)
                if indicators:
                    f.write("\nTechnical Indicators:\n")
                    if 'rsi' in indicators:
                        f.write(f"RSI: {indicators['rsi']:.2f}\n")
                    
                    if 'macd' in indicators:
                        macd = indicators['macd']
                        f.write(f"MACD: {macd['macd']:.8f}, Signal: {macd['signal']:.8f}, Histogram: {macd['histogram']:.8f}\n")
                    
                    if 'bollinger_bands' in indicators:
                        bb = indicators['bollinger_bands']
                        f.write(f"Bollinger Bands: Upper: {bb['upper']:.8f}, Middle: {bb['middle']:.8f}, Lower: {bb['lower']:.8f}\n")
                    
                    if 'sma_50' in indicators:
                        f.write(f"SMA 50: {indicators['sma_50']:.8f}\n")
                    
                    if 'sma_200' in indicators:
                        f.write(f"SMA 200: {indicators['sma_200']:.8f}\n")
                
                # Add trading signals
                signals = technical_analyzer.get_trading_signals(token_mint)
                if signals:
                    f.write("\nTrading Signals:\n")
                    f.write(f"Overall: {signals['overall'].upper()}\n")
                    
                    if signals['buy_signals']:
                        f.write("Buy Signals:\n")
                        for signal in signals['buy_signals']:
                            f.write(f"- {signal}\n")
                    
                    if signals['sell_signals']:
                        f.write("Sell Signals:\n")
                        for signal in signals['sell_signals']:
                            f.write(f"- {signal}\n")
            
            return str(chart_file)
        except Exception as e:
            logger.error(f"Error generating image chart: {e}")
            return None
    
    def generate_indicator_chart(self, token_mint: str, indicator: str, 
                                time_range: str = "1d") -> Optional[str]:
        """
        Generate a chart for a specific indicator.
        
        Args:
            token_mint: The token mint address
            indicator: Indicator name (rsi, macd, bollinger_bands)
            time_range: Time range (1h, 4h, 1d, 1w, 1m)
            
        Returns:
            Path to the generated chart or ASCII chart string
        """
        if not self.enabled:
            logger.warning("Chart generation is disabled")
            return None
        
        # Get price data
        price_data = technical_analyzer.price_data.get(token_mint)
        if price_data is None or len(price_data) == 0:
            logger.warning(f"No price data available for {token_mint}")
            return None
        
        # Get indicators
        indicators = technical_analyzer.get_token_indicators(token_mint)
        if not indicators or indicator not in indicators:
            logger.warning(f"Indicator {indicator} not available for {token_mint}")
            return None
        
        # Filter data by time range
        end_time = datetime.now()
        if time_range == "1h":
            start_time = end_time - timedelta(hours=1)
        elif time_range == "4h":
            start_time = end_time - timedelta(hours=4)
        elif time_range == "1d":
            start_time = end_time - timedelta(days=1)
        elif time_range == "1w":
            start_time = end_time - timedelta(weeks=1)
        elif time_range == "1m":
            start_time = end_time - timedelta(days=30)
        else:
            logger.warning(f"Invalid time range: {time_range}")
            start_time = end_time - timedelta(days=1)
        
        # Filter data
        filtered_data = price_data[price_data['timestamp'] >= start_time]
        
        if len(filtered_data) == 0:
            logger.warning(f"No price data available for {token_mint} in time range {time_range}")
            return None
        
        # Get token info
        tracked_tokens = get_config_value("tracked_tokens", {})
        token_info = tracked_tokens.get(token_mint, {})
        token_symbol = token_info.get("symbol", token_mint[:6])
        
        # Generate chart based on format and indicator
        if self.chart_format == "ascii":
            return self._generate_ascii_indicator_chart(filtered_data, indicators, indicator, token_symbol, time_range)
        elif self.chart_format == "png" or self.chart_format == "svg":
            return self._generate_image_indicator_chart(filtered_data, indicators, indicator, token_symbol, time_range, token_mint)
        else:
            logger.warning(f"Unsupported chart format: {self.chart_format}")
            return None
    
    def _generate_ascii_indicator_chart(self, data: pd.DataFrame, indicators: Dict[str, Any], 
                                       indicator: str, token_symbol: str, time_range: str) -> str:
        """
        Generate an ASCII chart for an indicator.
        
        Args:
            data: Price data
            indicators: Technical indicators
            indicator: Indicator name
            token_symbol: Token symbol
            time_range: Time range
            
        Returns:
            ASCII chart string
        """
        # This is a simplified implementation
        # In a real implementation, we would generate proper ASCII charts for each indicator
        
        indicator_data = indicators[indicator]
        
        if indicator == "rsi":
            return f"{token_symbol} - RSI ({time_range})\nCurrent RSI: {indicator_data:.2f}\n"
        elif indicator == "macd":
            return (f"{token_symbol} - MACD ({time_range})\n"
                    f"MACD: {indicator_data['macd']:.8f}\n"
                    f"Signal: {indicator_data['signal']:.8f}\n"
                    f"Histogram: {indicator_data['histogram']:.8f}\n")
        elif indicator == "bollinger_bands":
            return (f"{token_symbol} - Bollinger Bands ({time_range})\n"
                    f"Upper: {indicator_data['upper']:.8f}\n"
                    f"Middle: {indicator_data['middle']:.8f}\n"
                    f"Lower: {indicator_data['lower']:.8f}\n"
                    f"Current Price: {data['price'].iloc[-1]:.8f}\n")
        else:
            return f"{token_symbol} - {indicator} ({time_range})\nValue: {indicator_data}\n"
    
    def _generate_image_indicator_chart(self, data: pd.DataFrame, indicators: Dict[str, Any], 
                                       indicator: str, token_symbol: str, time_range: str, 
                                       token_mint: str) -> Optional[str]:
        """
        Generate an image chart for an indicator.
        
        Args:
            data: Price data
            indicators: Technical indicators
            indicator: Indicator name
            token_symbol: Token symbol
            time_range: Time range
            token_mint: Token mint address
            
        Returns:
            Path to the generated chart
        """
        try:
            # This is a placeholder for actual chart generation
            # In a real implementation, we would use a library like matplotlib
            
            # For now, just create a text file with indicator data
            chart_file = self.chart_dir / f"{token_symbol}_{indicator}_{time_range}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            
            with open(chart_file, 'w') as f:
                f.write(f"{token_symbol} - {indicator} Chart ({time_range})\n\n")
                
                if indicator == "rsi":
                    f.write(f"Current RSI: {indicators['rsi']:.2f}\n")
                    f.write("RSI Levels:\n")
                    f.write("- Overbought: 70+\n")
                    f.write("- Neutral: 30-70\n")
                    f.write("- Oversold: 30-\n")
                elif indicator == "macd":
                    macd = indicators['macd']
                    f.write(f"MACD: {macd['macd']:.8f}\n")
                    f.write(f"Signal: {macd['signal']:.8f}\n")
                    f.write(f"Histogram: {macd['histogram']:.8f}\n")
                    f.write("\nInterpretation:\n")
                    if macd['macd'] > macd['signal']:
                        f.write("- MACD above Signal Line: Bullish\n")
                    else:
                        f.write("- MACD below Signal Line: Bearish\n")
                    
                    if macd['histogram'] > 0:
                        f.write("- Positive Histogram: Bullish momentum\n")
                    else:
                        f.write("- Negative Histogram: Bearish momentum\n")
                elif indicator == "bollinger_bands":
                    bb = indicators['bollinger_bands']
                    current_price = data['price'].iloc[-1]
                    f.write(f"Upper Band: {bb['upper']:.8f}\n")
                    f.write(f"Middle Band: {bb['middle']:.8f}\n")
                    f.write(f"Lower Band: {bb['lower']:.8f}\n")
                    f.write(f"Current Price: {current_price:.8f}\n\n")
                    f.write("Interpretation:\n")
                    if current_price > bb['upper']:
                        f.write("- Price above Upper Band: Potentially overbought\n")
                    elif current_price < bb['lower']:
                        f.write("- Price below Lower Band: Potentially oversold\n")
                    else:
                        f.write("- Price within bands: Neutral\n")
                else:
                    f.write(f"Value: {indicators[indicator]}\n")
                
                # Add trading signals
                signals = technical_analyzer.get_trading_signals(token_mint)
                if signals:
                    f.write("\nTrading Signals:\n")
                    f.write(f"Overall: {signals['overall'].upper()}\n")
                    
                    if signals['buy_signals']:
                        f.write("Buy Signals:\n")
                        for signal in signals['buy_signals']:
                            f.write(f"- {signal}\n")
                    
                    if signals['sell_signals']:
                        f.write("Sell Signals:\n")
                        for signal in signals['sell_signals']:
                            f.write(f"- {signal}\n")
            
            return str(chart_file)
        except Exception as e:
            logger.error(f"Error generating image indicator chart: {e}")
            return None


# Create singleton instance
chart_generator = ChartGenerator()
