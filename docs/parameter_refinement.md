# Parameter Refinement Guide

This guide provides detailed information about the parameter refinement features in the Solana Memecoin Trading Bot.

## Overview

The parameter refinement system helps optimize trading parameters by:

1. Collecting performance data from trades and transactions
2. Analyzing patterns in successful and unsuccessful operations
3. Automatically adjusting risk management and gas optimization parameters
4. Learning from real-world usage to improve over time

## Performance Tracking

The system tracks three types of performance data:

1. **Trade Data**: Information about each trade, including entry/exit prices, profit/loss, and risk level
2. **Transaction Data**: Details about blockchain transactions, including success/failure, fees, and network conditions
3. **Portfolio Metrics**: Snapshots of portfolio value, allocation, and drawdown over time

This data is stored locally and used for parameter refinement.

## Refinement Types

The bot supports two types of parameter refinement:

### 1. Risk Parameter Refinement

Risk refinement optimizes risk management settings based on trading performance:

- **Risk Allocation Limits**: Adjusts allocation limits for different risk levels based on their performance
- **Risk Profile Parameters**: Fine-tunes stop-loss percentages, drawdown limits, and position sizing
- **Portfolio Management**: Optimizes portfolio allocation across risk categories

### 2. Gas Parameter Refinement

Gas refinement optimizes transaction fee settings based on transaction history:

- **Fee Multipliers**: Adjusts multipliers for different transaction types based on success rates
- **Congestion Response**: Optimizes fee adjustments during different network conditions
- **Time-Based Adjustments**: Refines peak hour definitions and time-based fee strategies

## Auto-Refinement

The bot can automatically refine parameters at specified intervals:

1. **Risk Refinement Interval**: Default is 7 days
2. **Gas Refinement Interval**: Default is 3 days
3. **Minimum Data Requirements**: 
   - At least 5 closed trades for risk refinement
   - At least 10 transactions for gas refinement

When auto-refinement is enabled, the bot checks if refinement is due at startup and periodically during operation.

## Manual Refinement

You can also trigger refinement manually through the CLI:

1. **Risk Refinement**: Use the `refinement` command and select "Run Risk Parameter Refinement"
2. **Gas Refinement**: Use the `refinement` command and select "Run Gas Parameter Refinement"

Manual refinement provides a summary of the data being used and allows you to confirm before making changes.

## Configuration Options

You can customize parameter refinement through the following configuration options:

```
# Auto-refinement settings
auto_refinement_enabled=false
risk_refinement_interval_days=7
gas_refinement_interval_days=3

# Minimum data requirements
min_trades_for_refinement=5
min_transactions_for_refinement=10
```

## CLI Commands

The following CLI commands are available for parameter refinement:

| Command | Description |
|---------|-------------|
| `refinement` | Open the refinement management menu |
| `refine_risk` | Run risk parameter refinement |
| `refine_gas` | Run gas parameter refinement |

## Refinement Process

### Risk Refinement Process

1. **Data Collection**: The system collects data on closed trades, including profit/loss and risk levels
2. **Performance Analysis**: It analyzes performance by risk level, calculating metrics like win rate and average P/L
3. **Parameter Adjustment**: Based on performance, it adjusts risk allocation limits and profile parameters
4. **Configuration Update**: The new parameters are saved to the configuration file
5. **Reporting**: A summary of changes is displayed to the user

### Gas Refinement Process

1. **Data Collection**: The system collects transaction data, including success/failure rates and network conditions
2. **Pattern Analysis**: It analyzes patterns in transaction success rates by type, congestion level, and time of day
3. **Multiplier Adjustment**: Based on the analysis, it adjusts fee multipliers for different transaction types
4. **Time-Based Optimization**: It identifies problematic hours and adjusts time-based fee strategies
5. **Configuration Update**: The new parameters are saved to the configuration file
6. **Reporting**: A summary of changes is displayed to the user

## Data Management

The refinement system includes tools for managing performance data:

1. **View Data**: See summary statistics of your performance data
2. **Clear Data**: Selectively clear trade data, transaction data, or portfolio metrics
3. **Export Data**: (Coming soon) Export performance data for external analysis

## Best Practices

1. **Allow Sufficient Data Collection**: Let the bot collect enough data before expecting meaningful refinements
2. **Start with Manual Refinement**: Review the first few refinements manually before enabling auto-refinement
3. **Regular Monitoring**: Periodically check refinement results to ensure they're improving performance
4. **Balanced Intervals**: Set refinement intervals that balance learning speed with stability
5. **Clear Data Selectively**: If market conditions change dramatically, consider clearing older data

## Troubleshooting

### Common Issues

1. **No Refinement Occurring**: Check that you have sufficient data (at least 5 trades or 10 transactions)
2. **Unexpected Parameter Changes**: Review your recent trading history for outliers that might be skewing results
3. **Performance Degradation**: If performance worsens after refinement, consider clearing data and starting fresh
4. **Data Persistence**: If data isn't persisting between sessions, check file permissions in the data directory

### Getting Help

If you encounter issues with parameter refinement:

1. Check the logs at `~/.solana-trading-bot/logs/trading_bot.log`
2. Look for warnings or errors related to performance tracking or refinement
3. Verify your configuration settings
