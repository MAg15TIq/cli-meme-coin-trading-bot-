# 🚀 SolSniperX: Advanced Solana Memecoin Trading Bot

A powerful CLI-based trading system for Solana meme coins with advanced risk management, gas optimization, and adaptive learning capabilities.

## ✨ Features

### 💰 Core Trading Features
- **Multi-Wallet Support**: Manage multiple wallets with metadata
- **Jupiter DEX Integration**: Best-in-class swap execution with optimal routing
- **Token Sniping**: Monitor liquidity pools for new tokens
- **Copy Trading**: Copy trades from successful wallets you choose to track
- **Whale Tracking**: Monitor specific wallets for trading activity
- **MEV Protection**: Protect your transactions from frontrunning

### 📊 Advanced Trading Tools
- **Social Sentiment Analysis**: Track and analyze social media mentions and sentiment
- **AI Strategy Generation**: Create and backtest trading strategies
- **Technical Analysis**: Advanced indicators and chart generation
- **Withdraw Functionality**: Transfer SOL and SPL tokens to external wallets
- **Price Alerts**: Set and receive alerts for specific price targets
- **External Wallet Monitoring**: Track transactions of any Solana wallet
- **Limit Orders**: Set buy and sell orders at specific price targets
- **DCA Orders**: Automate dollar-cost averaging with recurring buys or sells
- **Auto-Buy**: Quickly buy tokens by pasting addresses or using commands

### 🛡️ Risk Management
- **Risk Profiles**: Choose from conservative, moderate, or aggressive risk profiles
- **Token Risk Assessment**: Automatic evaluation of token risk factors
- **Position Sizing**: Smart position sizing based on portfolio and risk profile
- **Portfolio Monitoring**: Real-time tracking of portfolio allocation and risk
- **Stop-Loss Management**: Automatic or manual stop-loss settings
- **Take-Profit Levels**: Multi-tiered take-profit levels with partial selling

### ⚡ Performance Optimization
- **Gas Optimization**: Smart fee calculation based on transaction type and network conditions
- **RPC Failover**: Automatic switching between RPC endpoints for reliability
- **Transaction Timing**: Optimal timing for non-urgent transactions
- **Performance Tracking**: Detailed metrics on trades, transactions, and portfolio performance

## 🧠 Parameter Refinement System

SolSniperX features an **advanced parameter refinement system** that learns from your trading history:

- **Performance Tracking**: Collects data on trades, transactions, and portfolio metrics
- **Risk Refinement**: Automatically adjusts risk parameters based on trading performance
- **Gas Optimization**: Fine-tunes transaction fees based on network conditions and success rates
- **Auto-Refinement**: Schedules periodic refinements to continuously improve bot performance
- **Manual Controls**: Run refinements on demand with detailed before/after comparisons
- **Data Management**: View and manage performance data through an intuitive interface

## 🔒 Security Features
- **Encrypted Storage**: Secure storage of wallet information
- **Hardware Wallet Support**: Integration with hardware wallets for enhanced security
- **Transaction Confirmation**: Configurable confirmation requirements for high-value transactions
- **Withdrawal Limits**: Set maximum withdrawal amounts for added security
- **Auto-Logout**: Automatic logout after configurable period of inactivity

## 🖥️ CLI Interface

SolSniperX features a sleek, color-coded CLI interface that makes navigation intuitive and information easy to digest:

```
┌─────────────────────────────────────────────────────────────┐
│                     SOLSNIPERX v1.0.0                       │
└─────────────────────────────────────────────────────────────┘
  Wallet: Connected (8u6X...7bAp)
  Balance: 2.45 SOL

  MAIN MENU:
  1. Connect Wallet
  2. Check Balance
  3. Buy Token
  4. Sell Token
  5. Withdraw
  6. Bot Status
  7. Price Alerts
  8. Wallet Monitor
  9. Limit Orders
  a. DCA Orders
  b. Auto-Buy
  c. Token Analytics
  d. Risk Management
  e. Gas Optimization
  f. Refinement
  g. Settings
  h. Help
  q. Quit

  Enter your choice:
```

The interface includes:
- **Main Menu**: Quick access to all bot functions with numbered and lettered options
- **Dashboard**: Real-time overview of bot status and positions
- **Trading Screens**: Streamlined interfaces for executing trades
- **Risk Management**: Portfolio analysis and risk profile configuration
- **Gas Optimization**: Fee statistics and transaction timing recommendations
- **Parameter Refinement**: Performance metrics and refinement controls

## 📋 Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/solsniperx.git
   cd solsniperx
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

## ⚙️ Configuration

The application uses a configuration file located at `~/.solana-trading-bot/config.json`. This file is created automatically on first run with default values.

For detailed configuration instructions, see `docs/configuration_guide.md`.

## 📚 Documentation

SolSniperX comes with comprehensive documentation:

- **User Guide**: `docs/user_guide.md` - Complete guide to using the bot
- **Risk Management Guide**: `docs/risk_management.md` - Detailed explanation of risk management features
- **Gas Optimization Guide**: `docs/gas_optimization.md` - Guide to optimizing transaction fees
- **Parameter Refinement Guide**: `docs/parameter_refinement.md` - How to use the refinement system
- **Configuration Guide**: `docs/configuration_guide.md` - Setting up API keys and preferences
- **Testing Guide**: `docs/testing_guide.md` - How to test the bot's functionality

## 🔧 Usage

After installation, run the bot with:

```
python main.py
```

The main menu provides access to all bot functions:

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

## 🔐 Security

SolSniperX takes security seriously:

- Private keys are encrypted at rest
- Hardware wallet support for enhanced security
- Configurable confirmation requirements for high-value transactions
- Withdrawal limits to prevent unauthorized transfers
- Auto-logout after configurable period of inactivity

## �️ Development Roadmap

SolSniperX is continuously evolving:

- **Phase 1: Core Functionality** ✅
  - Wallet integration with multi-wallet support
  - Jupiter DEX integration for token swaps
  - Basic trading functionality (buy/sell)
  - CLI interface with intuitive navigation

- **Phase 2: Risk Management & Gas Optimization** ✅
  - Risk profiles and token risk assessment
  - Position sizing and portfolio monitoring
  - Stop-loss and take-profit management
  - Gas optimization with smart fee calculation

- **Phase 3: Advanced Features & Intelligence** ✅
  - Multi-tiered take-profit levels with partial selling
  - Token sniping with liquidity pool monitoring
  - Copy trading functionality with Helius API integration
  - Whale wallet tracking and transaction monitoring
  - Anti-MEV techniques for transaction protection

- **Phase 4: Optimization, Security & User Experience** ✅
  - Performance optimizations for real-time monitoring
  - Enhanced security measures for wallet protection
  - Multi-wallet management with metadata
  - Advanced error handling and logging
  - Social sentiment analysis integration
  - AI-powered trading strategy generation and backtesting

- **Phase 5: Advanced Features** ✅
  - Mobile companion app integration
  - Hardware wallet integration
  - Advanced charting and technical analysis
  - ML for token evaluation
  - Community strategy sharing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ❤️ Support

This bot is complete but needs funds to properly run the bot. Please donate SOL to this wallet:

```
8u6XMHBZV8foAi68QM1Vd4meFs3Wc7bfcFABmnqj7Ap
```

Your support helps maintain and improve SolSniperX!
