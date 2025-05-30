# Enhanced Features Guide

This guide covers the new advanced features added to the Solana Memecoin Trading Bot as part of the Phase 1 development improvements.

## Table of Contents

1. [Performance Optimizations](#performance-optimizations)
2. [Backtesting Engine](#backtesting-engine)
3. [Advanced Order Types](#advanced-order-types)
4. [Portfolio Analytics](#portfolio-analytics)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)

## Performance Optimizations

### Enhanced Jupiter API

The Jupiter API client has been significantly improved with:

#### Connection Pooling
- Increased connection pool size (20 connections, 50 max)
- Better retry strategies with exponential backoff
- Improved timeout handling

#### Advanced Caching System
- **Price Cache**: 5-second TTL for token prices
- **Quote Cache**: 2-second TTL for swap quotes
- **Token Info Cache**: 5-minute TTL for token metadata
- Automatic cache cleanup and performance metrics

#### Async Operations
```python
# Get multiple token prices concurrently
token_mints = ["token1", "token2", "token3"]
prices = jupiter_api.get_multiple_token_prices(token_mints)

# Performance metrics
metrics = jupiter_api.get_performance_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']}")
```

## Backtesting Engine

### Overview
The backtesting engine allows you to test trading strategies against historical data before deploying them live.

### Key Features
- **Historical Data Support**: Load OHLCV data from CSV files
- **Strategy Testing**: Test custom trading strategies
- **Performance Metrics**: Comprehensive analysis including Sharpe ratio, max drawdown, etc.
- **Order Simulation**: Realistic order execution with fees and slippage

### Basic Usage

```python
from src.trading.backtesting_engine import backtesting_engine, OrderType
import pandas as pd
from datetime import datetime, timedelta

# Load historical data
data = pd.read_csv('price_data.csv')  # timestamp, open, high, low, close, volume
backtesting_engine.load_price_data("token_mint", data)

# Define a simple strategy
def simple_strategy(engine, current_time, price_data):
    # Buy when price drops 5%
    current_price = engine.get_current_price("token_mint")
    if current_price and current_price < previous_price * 0.95:
        engine.place_order(OrderType.BUY, "token_mint", 10.0)

# Set strategy and run backtest
backtesting_engine.set_strategy(simple_strategy)
metrics = backtesting_engine.run_backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    time_step=timedelta(hours=1)
)

print(f"Total Return: {metrics.roi:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

### Performance Metrics
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses

## Advanced Order Types

### Overview
Advanced order types provide sophisticated execution strategies for large orders or specific market conditions.

### Order Types

#### 1. TWAP (Time-Weighted Average Price)
Executes large orders over a specified time period to minimize market impact.

```python
from src.trading.advanced_orders import advanced_order_manager

# Create TWAP order
order_id = advanced_order_manager.create_twap_order(
    token_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    token_name="USDC",
    side="buy",
    total_amount=1000.0,  # 1000 SOL
    duration_minutes=120,  # Over 2 hours
    interval_minutes=5     # Execute every 5 minutes
)
```

#### 2. VWAP (Volume-Weighted Average Price)
Executes orders based on market volume to achieve better average prices.

```python
# Create VWAP order
order_id = advanced_order_manager.create_vwap_order(
    token_mint="token_mint",
    token_name="TOKEN",
    side="sell",
    total_amount=500.0,
    volume_participation_rate=0.1  # 10% of market volume
)
```

#### 3. Iceberg Orders
Breaks large orders into smaller chunks to hide order size from the market.

```python
# Create Iceberg order
order_id = advanced_order_manager.create_iceberg_order(
    token_mint="token_mint",
    token_name="TOKEN",
    side="buy",
    total_amount=5000.0,
    slice_size=100.0,      # Execute 100 at a time
    min_slice_size=50.0    # Minimum final slice
)
```

#### 4. Conditional Orders
Execute orders when specific market conditions are met.

```python
# Create conditional order
conditions = [
    {'type': 'price_above', 'value': 1.5},
    {'type': 'volume_above', 'value': 10000.0}
]

order_id = advanced_order_manager.create_conditional_order(
    token_mint="token_mint",
    token_name="TOKEN",
    side="buy",
    total_amount=200.0,
    conditions=conditions
)
```

### Order Management

```python
# Check order status
status = advanced_order_manager.get_order_status(order_id)
print(f"Fill percentage: {status['fill_percentage']:.1f}%")

# List active orders
active_orders = advanced_order_manager.list_active_orders()

# Cancel order
advanced_order_manager.cancel_order(order_id)
```

## Portfolio Analytics

### Overview
Comprehensive portfolio analysis and performance tracking with real-time metrics and risk assessment.

### Key Features
- **Real-time Snapshots**: Automatic portfolio value tracking
- **Performance Metrics**: Returns, volatility, Sharpe ratio, etc.
- **Risk Analysis**: Concentration risk, correlation analysis, VaR
- **Allocation Tracking**: Asset allocation breakdown

### Usage

```python
from src.trading.portfolio_analytics import portfolio_analytics

# Enable analytics
portfolio_analytics.set_enabled(True)

# Capture current snapshot
snapshot = portfolio_analytics.capture_snapshot()
print(f"Total Value: {snapshot.total_value_sol:.2f} SOL")
print(f"24h P&L: {snapshot.pnl_24h:.2f} SOL")

# Get current allocation
allocation = portfolio_analytics.get_current_allocation()
for token, percentage in allocation.items():
    print(f"{token}: {percentage:.1f}%")

# Calculate performance metrics
performance = portfolio_analytics.calculate_performance_metrics(days=30)
print(f"30-day Return: {performance.total_return:.2%}")
print(f"Volatility: {performance.volatility:.2%}")
print(f"Sharpe Ratio: {performance.sharpe_ratio:.2f}")

# Risk metrics
risk = portfolio_analytics.calculate_risk_metrics()
print(f"Concentration Risk: {risk.concentration_risk:.3f}")
print(f"Max Drawdown: {performance.max_drawdown:.2%}")

# Get dashboard data
dashboard = portfolio_analytics.get_dashboard_data()
```

### Dashboard Data Structure
```json
{
  "portfolio": {
    "total_value_sol": 1250.75,
    "sol_balance": 500.0,
    "token_value_sol": 750.75,
    "position_count": 5,
    "pnl_24h": 25.50,
    "pnl_7d": 125.75,
    "pnl_30d": 250.75
  },
  "allocation": {
    "SOL": 40.0,
    "TOKEN1": 30.0,
    "TOKEN2": 20.0,
    "TOKEN3": 10.0
  },
  "performance": {
    "total_return": 0.25,
    "sharpe_ratio": 1.5,
    "max_drawdown": 0.08
  },
  "risk": {
    "concentration_risk": 0.35,
    "correlation_with_sol": 0.65
  }
}
```

## Configuration

### Environment Variables

Add these to your `.env` file or environment:

```bash
# Backtesting
BACKTESTING_ENABLED=true
BACKTESTING_TRADING_FEE=0.003
BACKTESTING_SLIPPAGE=0.001

# Advanced Orders
ADVANCED_ORDERS_ENABLED=true
ADVANCED_ORDERS_EXECUTION_INTERVAL=30
TWAP_DEFAULT_INTERVAL_MINUTES=5
VWAP_DEFAULT_PARTICIPATION_RATE=0.1

# Portfolio Analytics
PORTFOLIO_ANALYTICS_ENABLED=true
PORTFOLIO_SNAPSHOT_INTERVAL=300
PORTFOLIO_MAX_SNAPSHOTS=1000
```

### Configuration File

The bot will automatically add these settings to your `config.json`:

```json
{
  "backtesting_enabled": false,
  "backtesting_trading_fee": 0.003,
  "advanced_orders_enabled": false,
  "portfolio_analytics_enabled": false,
  "portfolio_snapshot_interval": 300
}
```

## Usage Examples

### Complete Backtesting Workflow

```python
# 1. Load data
import pandas as pd
data = pd.read_csv('historical_data.csv')
backtesting_engine.load_price_data("token_mint", data)

# 2. Define strategy
def momentum_strategy(engine, current_time, price_data):
    token_mint = "token_mint"
    current_price = engine.get_current_price(token_mint)
    
    if current_price:
        # Simple momentum: buy if price increased 2% in last hour
        # (This is a simplified example)
        if should_buy_signal(current_price):
            engine.place_order(OrderType.BUY, token_mint, 50.0)
        elif should_sell_signal(current_price) and token_mint in engine.positions:
            pos = engine.positions[token_mint]
            engine.place_order(OrderType.SELL, token_mint, pos.amount)

# 3. Run backtest
backtesting_engine.set_strategy(momentum_strategy)
metrics = backtesting_engine.run_backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 1)
)

# 4. Analyze results
summary = backtesting_engine.get_backtest_summary(metrics)
backtesting_engine.export_results(metrics, "backtest_results.json")
```

### Advanced Order Execution

```python
# Enable advanced orders
advanced_order_manager.set_enabled(True)

# Create a large TWAP order
twap_order = advanced_order_manager.create_twap_order(
    token_mint="large_position_token",
    token_name="LARGE",
    side="sell",
    total_amount=10000.0,
    duration_minutes=240,  # 4 hours
    interval_minutes=10
)

# Monitor execution
import time
while True:
    status = advanced_order_manager.get_order_status(twap_order)
    if status['status'] in ['filled', 'cancelled', 'expired']:
        break
    
    print(f"Progress: {status['fill_percentage']:.1f}%")
    time.sleep(60)  # Check every minute
```

### Portfolio Monitoring

```python
# Set up continuous monitoring
portfolio_analytics.set_enabled(True)

# Periodic reporting
import time
while True:
    dashboard = portfolio_analytics.get_dashboard_data()
    
    print(f"Portfolio Value: {dashboard['portfolio']['total_value_sol']:.2f} SOL")
    print(f"24h P&L: {dashboard['portfolio']['pnl_24h']:.2f} SOL")
    
    # Check for risk alerts
    risk = dashboard['risk']
    if risk['concentration_risk'] > 0.5:
        print("WARNING: High concentration risk!")
    
    time.sleep(300)  # Update every 5 minutes
```

## Best Practices

1. **Backtesting**: Always backtest strategies before live deployment
2. **Advanced Orders**: Use TWAP/VWAP for large orders to minimize market impact
3. **Portfolio Analytics**: Monitor risk metrics regularly
4. **Performance**: Enable caching for better API performance
5. **Risk Management**: Set appropriate position sizes and stop losses

## Troubleshooting

### Common Issues

1. **Backtesting Data**: Ensure CSV files have correct column names
2. **Advanced Orders**: Check wallet connection before creating orders
3. **Analytics**: Allow time for sufficient data collection
4. **Performance**: Monitor cache hit rates for optimization

### Logs

Check logs for detailed information:
```bash
tail -f ~/.solana-trading-bot/logs/trading_bot.log
```
