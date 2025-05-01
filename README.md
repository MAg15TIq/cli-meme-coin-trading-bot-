# Advanced Solana Memecoin Trader

A command-line trading system for Solana meme coins with a Python-based core, featuring wallet integration, DEX connections, and proper security measures.

## 🚀 New: Real Blockchain Functionality

The trading bot now features **real blockchain functionality** for all advanced trading features:

- **Token Analytics**: Real-time data on social sentiment, holder metrics, token age, developer activity, market cap, and volume
- **Social Sentiment Analysis**: Live Twitter API integration for real sentiment tracking
- **Holder Metrics**: Actual on-chain holder count and concentration data
- **Developer Activity**: GitHub integration for tracking project development
- **Withdraw Functionality**: Real blockchain transactions for SOL and token withdrawals
- **Limit Orders**: Actual on-chain execution when price conditions are met
- **DCA Orders**: Real recurring buys and sells at specified intervals
- **Price Alerts**: Live price monitoring with customizable alert conditions
- **Wallet Monitoring**: Real-time tracking of external wallet transactions
- **Auto-Buy**: Instant token purchases with blockchain execution

## Features

- **Secure Wallet Management**: Encrypted storage of wallet keys with multi-wallet support
- **Solana Blockchain Integration**: Connect to Solana RPC endpoints with failover and load balancing
- **Trading Capabilities**: Buy and sell tokens using Jupiter V6 DEX
- **Rich Command-Line Interface**: Interactive CLI with numbered and lettered options
- **Balance Checking**: View SOL and SPL token balances
- **Position Tracking**: Monitor open positions with real-time price updates
- **Automated Trading**: Stop-loss and take-profit orders execute automatically
- **Advanced Priority Fee Management**: Dynamic fees with network congestion detection
- **Multi-Tiered Take-Profits**: Set multiple take-profit levels with partial sells
- **Token Sniping**: Automatically detect and buy new tokens when liquidity is added
- **Copy Trading**: Copy trades from successful wallets you choose to track
- **Whale Tracking**: Monitor specific wallets for trading activity
- **Social Sentiment Analysis**: Track and analyze social media mentions and sentiment
- **AI Strategy Generation**: Create and backtest trading strategies
- **Withdraw Functionality**: Transfer SOL and SPL tokens to external wallets
- **Price Alerts**: Set and receive alerts for specific price targets
- **External Wallet Monitoring**: Track transactions of any Solana wallet
- **Limit Orders**: Set buy and sell orders at specific price targets
- **DCA Orders**: Automate dollar-cost averaging with recurring buys or sells
- **Auto-Buy**: Quickly buy tokens by pasting addresses or using commands
- **Enhanced Logging**: Comprehensive logging with different levels and formats

## Project Structure

```
cli-trading-bot/
├── main.py                # Main entry point
├── requirements.txt       # Python dependencies
├── config.py              # Configuration handling
├── src/
│   ├── cli/
│   │   ├── cli_interface.py  # CLI interface implementation
│   │   └── cli_functions.py  # CLI function implementations
│   ├── wallet/
│   │   ├── wallet.py      # Wallet management with multi-wallet support
│   │   ├── hardware_wallet.py # Hardware wallet integration
│   │   └── withdraw.py    # Withdraw functionality for SOL and tokens
│   ├── solana/
│   │   └── solana_interact.py  # Solana blockchain interactions
│   ├── trading/
│   │   ├── jupiter_api.py      # Jupiter DEX integration
│   │   ├── position_manager.py # Position tracking and management
│   │   ├── helius_api.py       # Helius API for wallet tracking
│   │   ├── pool_monitor.py     # Liquidity pool monitoring and sniping
│   │   ├── copy_trading.py     # Copy trading functionality
│   │   ├── sentiment_analysis.py # Social sentiment analysis
│   │   ├── strategy_generator.py # AI-powered trading strategies
│   │   ├── technical_analysis.py # Technical indicators and analysis
│   │   ├── price_alerts.py     # Price alerts functionality
│   │   ├── wallet_monitor.py   # External wallet monitoring
│   │   ├── limit_orders.py     # Limit orders functionality
│   │   ├── dca_orders.py       # DCA orders functionality
│   │   ├── auto_buy.py         # Auto-buy functionality
│   │   └── token_analytics.py  # Comprehensive token analytics
│   ├── charts/
│   │   └── chart_generator.py  # Chart generation for technical analysis
│   ├── mobile/
│   │   └── mobile_app.py       # Mobile companion app integration
│   ├── ml/
│   │   └── token_evaluator.py  # ML-based token evaluation
│   ├── community/
│   │   └── strategy_sharing.py # Community strategy sharing
│   ├── notifications/
│   │   └── notification_service.py # Enhanced notification system
│   └── utils/
│       └── logging_utils.py    # Enhanced logging functionality
└── docs/
    ├── configuration_guide.md  # Guide for configuring API keys
    ├── testing_guide.md        # Guide for testing functionality
    └── user_guide.md           # Comprehensive user guide
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/MAg15TIq/cli-meme-coin-trading-bot-.git
   cd cli-meme-coin-trading-bot-
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure API keys (see `docs/configuration_guide.md`):
   - Solana RPC endpoint
   - Helius API key
   - Twitter API keys (for sentiment analysis)
   - GitHub token (optional, for developer activity metrics)

4. Run the application:
   ```
   python main.py
   ```

## Configuration

The application uses a configuration file located at `~/.solana-trading-bot/config.json`. This file is created automatically on first run with default values.

For detailed configuration instructions, see `docs/configuration_guide.md`.

## Documentation

- **Configuration Guide**: `docs/configuration_guide.md`
- **Testing Guide**: `docs/testing_guide.md`
- **User Guide**: `docs/user_guide.md`

## Usage

1. **Connect Wallet**: Create a new wallet or import an existing one
2. **Check Balance**: View your SOL and token balances
3. **Buy Token**: Purchase tokens using SOL
4. **Sell Token**: Sell tokens for SOL
5. **Withdraw**: Transfer SOL or tokens to external wallets
6. **Bot Status**: View the current status of the bot
7. **Price Alerts**: Set and manage price alerts
8. **Wallet Monitor**: Track external wallets
9. **Limit Orders**: Set buy and sell orders at specific price targets
10. **DCA Orders**: Automate dollar-cost averaging
11. **Auto-Buy**: Quickly buy tokens
12. **Token Analytics**: Analyze token metrics and social sentiment

For detailed usage instructions, see `docs/user_guide.md`.

## Security

- Private keys are encrypted using Fernet symmetric encryption
- Passwords are never stored, only used to derive encryption keys
- Key derivation uses PBKDF2 with SHA-256 and 100,000 iterations
- Withdrawal limits and confirmations for enhanced security

## License

MIT

## Disclaimer

This software is for educational purposes only. Use at your own risk. Trading cryptocurrencies involves significant risk and can result in the loss of your invested capital. You should not invest more than you can afford to lose.

## Support

If you find this project useful, please consider donating SOL to this wallet:

```
8u6XMHBZV8foAi68QM1Vd4meFs3Wc7bfcFABmnqj7Ap
```

Your support helps maintain and improve this trading bot!
