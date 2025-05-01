# Gas Optimization Guide

This guide provides detailed information about the gas optimization features in the Solana Memecoin Trading Bot.

## Overview

The gas optimization system helps maximize transaction efficiency by:

1. Dynamically calculating optimal priority fees
2. Adjusting compute unit limits
3. Monitoring network congestion
4. Optimizing transaction timing
5. Refining parameters based on real-world performance

## Priority Fee Optimization

The bot uses a sophisticated approach to determine optimal priority fees:

1. **Percentile-Based Fees**: Analyzes recent network fees at different percentiles
2. **Transaction Type Multipliers**: Applies specific multipliers based on transaction type
3. **Congestion Adjustment**: Increases fees during high network congestion
4. **Time-Based Adjustment**: Accounts for peak usage hours
5. **Minimum Fee Protection**: Ensures fees never drop below a safe minimum

## Transaction Priority Levels

Four priority levels are available:

| Priority | Description | Percentile | Use Case |
|----------|-------------|------------|----------|
| Low | Economical, may take longer | 25th | Non-urgent transactions |
| Medium | Balanced (default) | 50th | Standard transactions |
| High | Faster execution | 75th | Time-sensitive transactions |
| Urgent | Fastest execution | 90th | Critical transactions |

## Transaction Type Multipliers

Different transaction types receive specific fee multipliers:

| Transaction Type | Default Multiplier | Rationale |
|------------------|-------------------|-----------|
| Default | 1.0 | Standard baseline |
| Buy | 1.0 | Standard trading operation |
| Sell | 1.0 | Standard trading operation |
| Snipe | 2.0 | Requires fast execution to secure opportunities |
| Swap | 1.2 | Slightly higher priority than standard |
| Limit Order | 0.8 | Not time-sensitive |
| Withdraw | 1.5 | Higher priority for security |

## Compute Unit Optimization

The system optimizes compute unit limits based on:

1. Transaction type requirements
2. Historical transaction data
3. Network conditions

This prevents both under-allocation (transaction failure) and over-allocation (wasted fees).

## Network Congestion Monitoring

The bot continuously monitors Solana network congestion by:

1. Tracking transaction throughput
2. Analyzing slot skipping rates
3. Monitoring validator performance
4. Building historical congestion patterns

This data informs fee calculations and transaction timing recommendations.

## Transaction Timing Optimization

For non-urgent transactions, the system can recommend optimal execution times:

1. **Congestion-Based Timing**: Suggests waiting during high congestion periods
2. **Time-of-Day Analysis**: Identifies peak and off-peak hours
3. **Urgency Levels**: Adjusts recommendations based on transaction urgency
4. **Fee Savings Estimates**: Provides estimated savings from timing optimization

## Fee Parameter Refinement

The gas optimization system can refine its parameters based on real-world performance:

1. **Transaction Analysis**: The system analyzes your transaction history to identify patterns in successful and failed transactions.

2. **Multiplier Adjustment**: If certain transaction types consistently fail or succeed, their multipliers are adjusted accordingly.

3. **Congestion Response**: The system learns how different fee levels perform during various congestion conditions.

4. **Time-of-Day Optimization**: Peak hour definitions are refined based on actual transaction success rates.

## Configuration Options

You can customize gas optimization through the following configuration options:

```
# Enable/disable fee optimization
fee_optimization_enabled=true

# Priority fee percentiles
low_priority_percentile=25
medium_priority_percentile=50
high_priority_percentile=75
urgent_priority_percentile=90

# Transaction type multipliers
default_fee_multiplier=1.0
buy_fee_multiplier=1.0
sell_fee_multiplier=1.0
snipe_fee_multiplier=2.0
swap_fee_multiplier=1.2
limit_order_fee_multiplier=0.8
withdraw_fee_multiplier=1.5

# Compute unit limits
compute_unit_limit=200000
buy_compute_limit=200000
sell_compute_limit=200000
snipe_compute_limit=240000
swap_compute_limit=220000
limit_order_compute_limit=200000
withdraw_compute_limit=200000

# Time-based adjustment
time_based_fee_adjustment=false
urgent_sell_fee_boost=false

# Minimum fee
min_priority_fee=1000

# Congestion check interval (seconds)
congestion_check_interval=60
```

## Fee Statistics

Use the `fees` command to view detailed fee statistics, including:

1. Recent fee trends by percentile
2. Current network congestion
3. Optimal transaction timing recommendations
4. Historical fee data analysis

## Best Practices

1. **Use Appropriate Priority Levels**: Match priority to the urgency of your transaction.

2. **Consider Timing Recommendations**: For non-urgent transactions, follow timing suggestions to save on fees.

3. **Monitor Congestion**: Check network congestion before executing large or important transactions.

4. **Adjust for Token Type**: Use higher priority for low-liquidity or volatile tokens.

5. **Regular Refinement**: Allow the system to refine parameters periodically based on your transaction history.

6. **Balance Speed and Cost**: Higher fees don't always guarantee faster execution; find the optimal balance.

7. **Use Snipe Mode Selectively**: The snipe multiplier is significantly higher, so use it only when necessary.

## Troubleshooting

### Common Issues

1. **Transaction failures**: If transactions are failing despite fee optimization, try:
   - Manually increasing the priority level
   - Checking for other transaction issues (insufficient funds, etc.)
   - Verifying RPC endpoint reliability

2. **High fees**: If fees seem consistently high:
   - Check network congestion levels
   - Consider waiting for off-peak hours
   - Verify transaction type settings

3. **Slow transactions**: If transactions are taking too long:
   - Increase priority level
   - Check network status
   - Consider using a different RPC endpoint

### Getting Help

If you encounter issues with gas optimization:

1. Check the logs at `~/.solana-trading-bot/logs/trading_bot.log`
2. Look for warnings or errors related to fee calculation
3. Verify your configuration settings
