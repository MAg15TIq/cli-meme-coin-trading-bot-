# Configuration Guide for Solana Memecoin Trading Bot

This guide explains how to configure the necessary API keys and settings for the trading bot's real functionality.

## Required API Keys

### 1. Solana RPC Endpoints

For reliable blockchain interactions, you should use a dedicated RPC endpoint:

- **Helius**: https://helius.xyz/ (Recommended)
- **QuickNode**: https://www.quicknode.com/
- **Alchemy**: https://www.alchemy.com/

Add your RPC URL to the configuration:
```
SOLANA_RPC_URL=https://your-rpc-endpoint.com
```

### 2. Helius API Key

Helius API is used for token analytics, wallet monitoring, and transaction data:

1. Sign up at https://helius.xyz/
2. Create an API key
3. Add to configuration:
```
HELIUS_API_KEY=your_helius_api_key
```

### 3. Twitter API Keys

For real social sentiment analysis:

1. Apply for Twitter API access at https://developer.twitter.com/
2. Create a project and get your API keys
3. Add to configuration:
```
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
```

### 4. GitHub Token (Optional)

For developer activity metrics:

1. Create a GitHub personal access token at https://github.com/settings/tokens
2. Add to configuration:
```
GITHUB_TOKEN=your_github_token
```

## Configuration Methods

You can configure the bot using one of these methods:

### 1. Environment Variables

Set the variables in your environment before running the bot.

### 2. .env File

Create a `.env` file in the root directory with your configuration:

```
# Solana Network
SOLANA_RPC_URL=https://your-rpc-endpoint.com
SOLANA_NETWORK=mainnet-beta

# API Keys
HELIUS_API_KEY=your_helius_api_key
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
GITHUB_TOKEN=your_github_token

# Feature Toggles
TOKEN_ANALYTICS_ENABLED=true
SENTIMENT_ANALYSIS_ENABLED=true
WALLET_MONITORING_ENABLED=true
PRICE_ALERTS_ENABLED=true
LIMIT_ORDERS_ENABLED=true
DCA_ENABLED=true
AUTO_BUY_ENABLED=true
```

### 3. Config File

The bot automatically creates a config file at `~/.solana-trading-bot/config.json`. You can edit this file directly.

## Feature-Specific Settings

### Token Analytics

```
TOKEN_ANALYTICS_ENABLED=true
ANALYTICS_MONITORING_INTERVAL=3600
MAX_ANALYTICS_HISTORY_ENTRIES=100
```

### Social Sentiment Analysis

```
SENTIMENT_ANALYSIS_ENABLED=true
SENTIMENT_MONITORING_INTERVAL=3600
```

### Wallet Monitoring

```
WALLET_MONITORING_ENABLED=true
WALLET_MONITORING_INTERVAL_SECONDS=300
MAX_TRANSACTIONS_PER_WALLET=100
```

### Price Alerts

```
PRICE_ALERTS_ENABLED=true
PRICE_ALERT_INTERVAL_SECONDS=60
```

### Limit Orders

```
LIMIT_ORDERS_ENABLED=true
LIMIT_ORDER_INTERVAL_SECONDS=30
```

### DCA Orders

```
DCA_ENABLED=true
DCA_INTERVAL_SECONDS=60
```

### Auto-Buy

```
AUTO_BUY_ENABLED=true
AUTO_BUY_DEFAULT_AMOUNT=0.1
AUTO_BUY_REQUIRE_CONFIRMATION=true
```

## Security Recommendations

1. Never share your private keys or API keys
2. Use a dedicated wallet for trading
3. Set reasonable limits for automatic trading
4. Regularly backup your configuration
5. Monitor your wallet activity regularly
