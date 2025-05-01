"""
Technical analysis module for the Solana Memecoin Trading Bot.
Provides technical indicators and analysis tools for trading decisions.
Includes enhanced indicators for exit strategies.
"""

import json
import logging
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta

from config import get_config_value, update_config
from src.trading.jupiter_api import jupiter_api
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class TechnicalAnalysis:
    """Technical analysis tools and indicators with enhanced exit strategy support."""

    def __init__(self):
        """Initialize the technical analysis module."""
        self.enabled = get_config_value("technical_analysis_enabled", False)
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.indicators: Dict[str, Dict[str, Any]] = {}
        self.update_interval = int(get_config_value("ta_update_interval", "300"))  # Default: 5 minutes

        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()

        # Load saved data
        self._load_data()

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable technical analysis.

        Args:
            enabled: Whether technical analysis should be enabled
        """
        self.enabled = enabled
        update_config("technical_analysis_enabled", enabled)
        logger.info(f"Technical analysis {'enabled' if enabled else 'disabled'}")

        if enabled and not self.monitoring_thread:
            self.start_monitoring_thread()
        elif not enabled and self.monitoring_thread:
            self.stop_monitoring_thread()

    def start_monitoring_thread(self) -> None:
        """Start the monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Technical analysis monitoring thread already running")
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Technical analysis monitoring thread started")

    def stop_monitoring_thread(self) -> None:
        """Stop the monitoring thread."""
        if not self.monitoring_thread:
            logger.warning("Technical analysis monitoring thread not running")
            return

        self.stop_monitoring.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.monitoring_thread = None
        logger.info("Technical analysis monitoring thread stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Update price data for all tracked tokens
                self._update_price_data()

                # Calculate indicators
                self._calculate_indicators()

                # Save data
                self._save_data()
            except Exception as e:
                logger.error(f"Error in technical analysis monitoring loop: {e}")

            # Sleep for the update interval
            self.stop_monitoring.wait(self.update_interval)

    def _load_data(self) -> None:
        """Load saved price data and indicators."""
        try:
            # Load price data
            price_data_file = get_config_value("price_data_file", "price_data.json")
            try:
                with open(price_data_file, 'r') as f:
                    price_data_json = json.load(f)

                for token_mint, data in price_data_json.items():
                    self.price_data[token_mint] = pd.DataFrame(data)
                    # Convert timestamp strings to datetime
                    if 'timestamp' in self.price_data[token_mint].columns:
                        self.price_data[token_mint]['timestamp'] = pd.to_datetime(
                            self.price_data[token_mint]['timestamp']
                        )
            except (FileNotFoundError, json.JSONDecodeError):
                logger.info("No saved price data found or invalid format")

            # Load indicators
            indicators_file = get_config_value("indicators_file", "indicators.json")
            try:
                with open(indicators_file, 'r') as f:
                    self.indicators = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logger.info("No saved indicators found or invalid format")
        except Exception as e:
            logger.error(f"Error loading technical analysis data: {e}")

    def _save_data(self) -> None:
        """Save price data and indicators."""
        try:
            # Save price data
            price_data_file = get_config_value("price_data_file", "price_data.json")
            price_data_json = {}
            for token_mint, df in self.price_data.items():
                # Convert datetime to string for JSON serialization
                df_copy = df.copy()
                if 'timestamp' in df_copy.columns:
                    df_copy['timestamp'] = df_copy['timestamp'].astype(str)
                price_data_json[token_mint] = df_copy.to_dict('records')

            with open(price_data_file, 'w') as f:
                json.dump(price_data_json, f, indent=2)

            # Save indicators
            indicators_file = get_config_value("indicators_file", "indicators.json")
            with open(indicators_file, 'w') as f:
                json.dump(self.indicators, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving technical analysis data: {e}")

    def _update_price_data(self) -> None:
        """Update price data for tracked tokens."""
        # Get list of tokens to track
        tracked_tokens = get_config_value("tracked_tokens", {})

        for token_mint, token_info in tracked_tokens.items():
            try:
                # Get token price history
                price_history = self._get_token_price_history(token_mint)

                if not price_history:
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(price_history)

                # Ensure timestamp is datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Sort by timestamp
                df = df.sort_values('timestamp')

                # Store in price data dictionary
                self.price_data[token_mint] = df

                logger.debug(f"Updated price data for {token_info.get('symbol', token_mint)}")
            except Exception as e:
                logger.error(f"Error updating price data for {token_mint}: {e}")

    def _get_token_price_history(self, token_mint: str) -> List[Dict[str, Any]]:
        """
        Get price history for a token.

        Args:
            token_mint: The token mint address

        Returns:
            List of price data points
        """
        try:
            # This is a simplified implementation
            # In a real implementation, we would use a proper price API

            # Get time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)  # 7 days of data

            # Get price data from Jupiter API
            # Note: Jupiter API doesn't provide historical data directly
            # This is a placeholder for actual implementation

            # For now, generate some random data for testing
            price_history = []
            current_time = start_time

            # Get current price from Jupiter API
            current_price = jupiter_api.get_token_price(token_mint)

            if not current_price:
                return []

            # Generate random price history
            price = current_price
            while current_time <= end_time:
                # Add some random variation
                price_change = price * (np.random.normal(0, 0.02))  # 2% standard deviation
                price += price_change

                price_history.append({
                    'timestamp': current_time.isoformat(),
                    'price': max(0.000000001, price),  # Ensure price is positive
                    'volume': np.random.uniform(1000, 100000)
                })

                current_time += timedelta(hours=1)

            return price_history
        except Exception as e:
            logger.error(f"Error getting price history for {token_mint}: {e}")
            return []

    def _calculate_indicators(self) -> None:
        """Calculate technical indicators for all tokens."""
        for token_mint, df in self.price_data.items():
            try:
                if len(df) < 20:  # Need at least 20 data points for meaningful indicators
                    continue

                # Initialize indicators dictionary for this token
                if token_mint not in self.indicators:
                    self.indicators[token_mint] = {}

                # Calculate RSI
                self.indicators[token_mint]['rsi'] = self._calculate_rsi(df, 14)

                # Calculate MACD
                macd, signal, histogram = self._calculate_macd(df, 12, 26, 9)
                self.indicators[token_mint]['macd'] = {
                    'macd': macd,
                    'signal': signal,
                    'histogram': histogram
                }

                # Calculate Bollinger Bands
                upper, middle, lower = self._calculate_bollinger_bands(df, 20, 2)
                self.indicators[token_mint]['bollinger_bands'] = {
                    'upper': upper,
                    'middle': middle,
                    'lower': lower
                }

                # Calculate Moving Averages
                self.indicators[token_mint]['sma_50'] = self._calculate_sma(df, 50)
                self.indicators[token_mint]['sma_200'] = self._calculate_sma(df, 200)
                self.indicators[token_mint]['ema_20'] = self._calculate_ema(df, 20)

                # Calculate ATR (Average True Range)
                self.indicators[token_mint]['atr'] = self._calculate_atr(df, 14)

                # Calculate OBV (On-Balance Volume)
                self.indicators[token_mint]['obv'] = self._calculate_obv(df)

                logger.debug(f"Calculated indicators for {token_mint}")
            except Exception as e:
                logger.error(f"Error calculating indicators for {token_mint}: {e}")

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate the Relative Strength Index (RSI).

        Args:
            df: DataFrame with price data
            period: RSI period

        Returns:
            Current RSI value
        """
        # Extract price series
        prices = df['price'].values

        # Calculate price changes
        deltas = np.diff(prices)

        # Initialize gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate average gains and losses
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        # Calculate RS and RSI
        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_macd(self, df: pd.DataFrame, fast_period: int = 12,
                       slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
        """
        Calculate the Moving Average Convergence Divergence (MACD).

        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period

        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        # Extract price series
        prices = df['price'].values

        # Calculate EMAs
        ema_fast = self._calculate_ema_series(prices, fast_period)
        ema_slow = self._calculate_ema_series(prices, slow_period)

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate signal line
        signal_line = self._calculate_ema_series(macd_line, signal_period)

        # Calculate histogram
        histogram = macd_line - signal_line

        # Return the most recent values
        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20,
                                  std_dev: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with price data
            period: SMA period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        # Extract price series
        prices = df['price'].values

        # Calculate middle band (SMA)
        middle_band = np.mean(prices[-period:])

        # Calculate standard deviation
        std = np.std(prices[-period:])

        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)

        return float(upper_band), float(middle_band), float(lower_band)

    def _calculate_sma(self, df: pd.DataFrame, period: int) -> float:
        """
        Calculate Simple Moving Average (SMA).

        Args:
            df: DataFrame with price data
            period: SMA period

        Returns:
            Current SMA value
        """
        # Extract price series
        prices = df['price'].values

        # Calculate SMA
        if len(prices) < period:
            return float(np.mean(prices))

        return float(np.mean(prices[-period:]))

    def _calculate_ema(self, df: pd.DataFrame, period: int) -> float:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            df: DataFrame with price data
            period: EMA period

        Returns:
            Current EMA value
        """
        # Extract price series
        prices = df['price'].values

        # Calculate EMA
        ema_series = self._calculate_ema_series(prices, period)

        return float(ema_series[-1])

    def _calculate_ema_series(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate EMA series for an array of prices.

        Args:
            prices: Array of prices
            period: EMA period

        Returns:
            Array of EMA values
        """
        # Calculate multiplier
        multiplier = 2 / (period + 1)

        # Initialize EMA with SMA
        ema = np.zeros_like(prices)
        ema[:period] = np.mean(prices[:period])

        # Calculate EMA
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).

        Args:
            df: DataFrame with price data
            period: ATR period

        Returns:
            Current ATR value
        """
        # Need high, low, close data
        # For simplicity, we'll estimate from the price
        if len(df) < 2:
            return 0.0

        # Extract price series
        prices = df['price'].values

        # Estimate high and low from price
        high = prices * 1.01  # 1% above price
        low = prices * 0.99   # 1% below price

        # Calculate true range
        tr = np.zeros(len(prices))

        # First value is just high - low
        tr[0] = high[0] - low[0]

        # Calculate subsequent values
        for i in range(1, len(prices)):
            tr[i] = max(
                high[i] - low[i],                # Current high - low
                abs(high[i] - prices[i-1]),      # Current high - previous close
                abs(low[i] - prices[i-1])        # Current low - previous close
            )

        # Calculate ATR
        atr = np.mean(tr[-period:])

        return float(atr)

    def _calculate_obv(self, df: pd.DataFrame) -> float:
        """
        Calculate On-Balance Volume (OBV).

        Args:
            df: DataFrame with price data

        Returns:
            Current OBV value
        """
        if len(df) < 2 or 'volume' not in df.columns:
            return 0.0

        # Extract price and volume series
        prices = df['price'].values
        volumes = df['volume'].values

        # Calculate OBV
        obv = np.zeros(len(prices))

        # First value is just the first volume
        obv[0] = volumes[0]

        # Calculate subsequent values
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif prices[i] < prices[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]

        return float(obv[-1])

    def get_token_indicators(self, token_mint: str) -> Dict[str, Any]:
        """
        Get technical indicators for a token.

        Args:
            token_mint: The token mint address

        Returns:
            Dictionary of technical indicators
        """
        if token_mint not in self.indicators:
            return {}

        return self.indicators[token_mint]

    def get_trading_signals(self, token_mint: str) -> Dict[str, Any]:
        """
        Get trading signals for a token based on technical indicators.

        Args:
            token_mint: The token mint address

        Returns:
            Dictionary of trading signals
        """
        if token_mint not in self.indicators:
            return {
                'buy_signals': [],
                'sell_signals': [],
                'overall': 'neutral'
            }

        indicators = self.indicators[token_mint]

        # Initialize signals
        buy_signals = []
        sell_signals = []

        # Check RSI
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < 30:
                buy_signals.append(f"RSI oversold ({rsi:.2f})")
            elif rsi > 70:
                sell_signals.append(f"RSI overbought ({rsi:.2f})")

        # Check MACD
        if 'macd' in indicators:
            macd = indicators['macd']
            if macd['macd'] > macd['signal'] and macd['histogram'] > 0:
                buy_signals.append("MACD bullish crossover")
            elif macd['macd'] < macd['signal'] and macd['histogram'] < 0:
                sell_signals.append("MACD bearish crossover")

        # Check Bollinger Bands
        if 'bollinger_bands' in indicators and token_mint in self.price_data:
            bb = indicators['bollinger_bands']
            current_price = self.price_data[token_mint]['price'].iloc[-1]

            if current_price < bb['lower']:
                buy_signals.append("Price below lower Bollinger Band")
            elif current_price > bb['upper']:
                sell_signals.append("Price above upper Bollinger Band")

        # Check Moving Averages
        if 'sma_50' in indicators and 'sma_200' in indicators and token_mint in self.price_data:
            sma_50 = indicators['sma_50']
            sma_200 = indicators['sma_200']
            current_price = self.price_data[token_mint]['price'].iloc[-1]

            if sma_50 > sma_200 and current_price > sma_50:
                buy_signals.append("Price above 50 SMA, 50 SMA above 200 SMA (Golden Cross)")
            elif sma_50 < sma_200 and current_price < sma_50:
                sell_signals.append("Price below 50 SMA, 50 SMA below 200 SMA (Death Cross)")

        # Determine overall signal
        overall = 'neutral'
        if len(buy_signals) > len(sell_signals) and len(buy_signals) >= 2:
            overall = 'buy'
        elif len(sell_signals) > len(buy_signals) and len(sell_signals) >= 2:
            overall = 'sell'

        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'overall': overall
        }


    def get_rsi(self, token_mint: str, period: int = 14) -> Optional[float]:
        """
        Get the current RSI value for a token.

        Args:
            token_mint: The token mint address
            period: RSI period

        Returns:
            Current RSI value, or None if not available
        """
        if not self.enabled:
            return None

        if token_mint not in self.indicators or 'rsi' not in self.indicators[token_mint]:
            # Try to calculate it on demand
            if token_mint in self.price_data and len(self.price_data[token_mint]) >= period:
                try:
                    return self._calculate_rsi(self.price_data[token_mint], period)
                except Exception as e:
                    logger.error(f"Error calculating RSI on demand: {e}")
                    return None
            return None

        return self.indicators[token_mint]['rsi']

    def get_macd(self, token_mint: str) -> Optional[Tuple[float, float]]:
        """
        Get the current MACD values for a token.

        Args:
            token_mint: The token mint address

        Returns:
            Tuple of (MACD line, signal line), or None if not available
        """
        if not self.enabled:
            return None

        if token_mint not in self.indicators or 'macd' not in self.indicators[token_mint]:
            # Try to calculate it on demand
            if token_mint in self.price_data and len(self.price_data[token_mint]) >= 26:
                try:
                    macd, signal, _ = self._calculate_macd(self.price_data[token_mint])
                    return (macd, signal)
                except Exception as e:
                    logger.error(f"Error calculating MACD on demand: {e}")
                    return None
            return None

        macd_data = self.indicators[token_mint]['macd']
        return (macd_data['macd'], macd_data['signal'])

    def get_bollinger_bands(self, token_mint: str) -> Optional[Tuple[float, float, float]]:
        """
        Get the current Bollinger Bands for a token.

        Args:
            token_mint: The token mint address

        Returns:
            Tuple of (upper band, middle band, lower band), or None if not available
        """
        if not self.enabled:
            return None

        if token_mint not in self.indicators or 'bollinger_bands' not in self.indicators[token_mint]:
            # Try to calculate it on demand
            if token_mint in self.price_data and len(self.price_data[token_mint]) >= 20:
                try:
                    return self._calculate_bollinger_bands(self.price_data[token_mint])
                except Exception as e:
                    logger.error(f"Error calculating Bollinger Bands on demand: {e}")
                    return None
            return None

        bb = self.indicators[token_mint]['bollinger_bands']
        return (bb['upper'], bb['middle'], bb['lower'])

    def get_trailing_stop_price(self, token_mint: str, current_price: float, trail_percent: float) -> float:
        """
        Calculate a trailing stop price.

        Args:
            token_mint: The token mint address
            current_price: The current price
            trail_percent: The percentage below the highest price

        Returns:
            The trailing stop price
        """
        # Get price history
        if token_mint in self.price_data and len(self.price_data[token_mint]) > 0:
            # Find highest price in recent history
            recent_prices = self.price_data[token_mint]['price'].values[-20:]  # Last 20 periods
            highest_price = max(max(recent_prices), current_price)
        else:
            highest_price = current_price

        # Calculate trailing stop price
        stop_price = highest_price * (1 - trail_percent / 100)

        return stop_price

    def should_exit_based_on_indicators(self, token_mint: str, entry_price: float,
                                       current_price: float) -> Tuple[bool, str]:
        """
        Determine if a position should be exited based on technical indicators.

        Args:
            token_mint: The token mint address
            entry_price: The entry price
            current_price: The current price

        Returns:
            Tuple of (should exit, reason)
        """
        if not self.enabled:
            return (False, "Technical analysis not enabled")

        # Get trading signals
        signals = self.get_trading_signals(token_mint)

        # Check if we have a sell signal
        if signals['overall'] == 'sell' and len(signals['sell_signals']) >= 2:
            return (True, f"Technical indicators suggest selling: {', '.join(signals['sell_signals'])}")

        # Check RSI for overbought condition
        rsi = self.get_rsi(token_mint)
        if rsi is not None and rsi > 75:  # Strongly overbought
            return (True, f"RSI is overbought at {rsi:.2f}")

        # Check MACD for bearish crossover
        macd = self.get_macd(token_mint)
        if macd is not None:
            macd_line, signal_line = macd
            if macd_line < signal_line and macd_line > 0:  # Bearish crossover in positive territory
                return (True, f"MACD bearish crossover: {macd_line:.6f} < {signal_line:.6f}")

        # Check Bollinger Bands for price above upper band
        bb = self.get_bollinger_bands(token_mint)
        if bb is not None:
            upper_band, _, _ = bb
            if current_price > upper_band * 1.05:  # Price significantly above upper band
                return (True, f"Price ({current_price:.6f}) significantly above upper Bollinger Band ({upper_band:.6f})")

        # Check for significant price increase
        if current_price > entry_price * 2:  # 100% increase
            # Check if price is starting to reverse
            if token_mint in self.price_data and len(self.price_data[token_mint]) >= 3:
                recent_prices = self.price_data[token_mint]['price'].values[-3:]
                if recent_prices[2] < recent_prices[1]:  # Price is starting to decline
                    return (True, f"Price reversal after significant gain: {entry_price:.6f} -> {current_price:.6f}")

        return (False, "No exit signal")


# Create singleton instance
technical_analyzer = TechnicalAnalysis()
