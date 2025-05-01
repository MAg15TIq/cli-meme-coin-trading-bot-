# Testing Guide for Solana Memecoin Trading Bot

This guide provides step-by-step instructions for testing each of the real functionality features implemented in the trading bot.

## Prerequisites

Before testing, ensure you have:

1. Configured all necessary API keys (see `configuration_guide.md`)
2. Connected a wallet with some SOL for testing transactions
3. Identified a few low-value tokens for testing trading features

## 1. Token Analytics Testing

Test the token analytics functionality to ensure it retrieves real data:

1. Start the bot with `python main.py`
2. Select "Token Analytics" from the menu
3. If prompted, enable token analytics
4. Enter a token mint address or select from your positions
5. Verify that the following data appears:
   - Market data (price, market cap, volume)
   - Holder metrics (holder count, concentration)
   - Social sentiment (Twitter mentions, sentiment score)
   - Developer activity (if GitHub token is configured)

**Expected Result**: Real token data should be displayed with non-zero values for most metrics.

## 2. Social Sentiment Analysis Testing

Test the social sentiment analysis functionality:

1. Enable sentiment analysis in settings
2. Select a popular token (like SOL, BONK, etc.)
3. Check the token analytics for this token
4. Verify that Twitter mentions and sentiment scores are populated

**Expected Result**: Twitter mentions should show a non-zero value, and sentiment score should be between -1.0 and 1.0.

## 3. Wallet Monitoring Testing

Test the wallet monitoring functionality:

1. Enable wallet monitoring in settings
2. Add a wallet address to monitor (use a known active wallet)
3. Wait for the monitoring interval to pass
4. Check for transaction notifications

**Expected Result**: The system should detect and display transactions for the monitored wallet.

## 4. Withdraw Functionality Testing

Test the withdraw functionality (use small amounts):

1. Connect your wallet
2. Select "Withdraw" from the menu
3. Choose SOL or a token to withdraw
4. Enter a small amount (e.g., 0.001 SOL)
5. Enter a destination address (preferably another wallet you own)
6. Confirm the transaction

**Expected Result**: The transaction should be processed on the blockchain, and the funds should appear in the destination wallet.

## 5. Limit Orders Testing

Test the limit orders functionality:

1. Enable limit orders in settings
2. Select a token to create a limit order for
3. Set a buy limit order slightly below current price
4. Set a sell limit order slightly above current price
5. Monitor the orders to see if they trigger when price conditions are met

**Expected Result**: Orders should be created and executed when price conditions are met.

## 6. DCA Orders Testing

Test the DCA (Dollar Cost Averaging) functionality:

1. Enable DCA orders in settings
2. Create a DCA buy order for a token with:
   - Small amount (e.g., 0.01 SOL)
   - Short interval (e.g., 5 minutes)
   - Maximum 2 executions
3. Monitor the executions

**Expected Result**: The DCA order should execute at the specified intervals.

## 7. Price Alerts Testing

Test the price alerts functionality:

1. Enable price alerts in settings
2. Set a price alert for a token:
   - Above current price (e.g., +5%)
   - Below current price (e.g., -5%)
3. Monitor for alert notifications

**Expected Result**: Alerts should trigger when price conditions are met.

## 8. Auto-Buy Testing

Test the auto-buy functionality:

1. Enable auto-buy in settings
2. Set a small default amount (e.g., 0.01 SOL)
3. Paste a token mint address in the command line
4. Confirm the auto-buy if prompted

**Expected Result**: The token should be purchased automatically.

## 9. Risk Management Testing

Test the risk management functionality:

1. Select "Risk Management" from the menu
2. Test different risk profiles:
   - Set profile to "conservative"
   - Set profile to "moderate"
   - Set profile to "aggressive"
3. Check portfolio risk metrics:
   - View current portfolio allocation
   - Check risk level distribution
   - Verify drawdown calculation
4. Test position sizing:
   - Get position size recommendation for a low-risk token
   - Get position size recommendation for a high-risk token
   - Verify that position sizes respect allocation limits
5. Test risk parameter refinement:
   - View current risk parameters
   - Run refinement based on trading history
   - Verify parameter adjustments

**Expected Result**: Risk profiles should change settings appropriately, position sizing should respect risk limits, and refinement should adjust parameters based on performance data.

## 10. Gas Optimization Testing

Test the gas optimization functionality:

1. Select "Gas Optimization" from the menu
2. Check fee statistics:
   - View recent fee trends
   - Check current network congestion
   - Verify fee percentiles are populated
3. Test priority fee calculation:
   - Get fee recommendation for different transaction types
   - Get fee recommendation for different priority levels
   - Verify that fees adjust based on network congestion
4. Test transaction timing optimization:
   - Get timing recommendations for different urgency levels
   - Verify wait time recommendations during high congestion
   - Check fee savings estimates
5. Test fee parameter refinement:
   - View current fee multipliers
   - Run refinement based on transaction history
   - Verify multiplier adjustments

**Expected Result**: Fee recommendations should vary by transaction type and priority, timing optimization should suggest appropriate wait times during congestion, and refinement should adjust parameters based on transaction success rates.

## 11. Parameter Refinement Testing

Test the parameter refinement functionality:

1. Select "Refinement" from the menu
2. Test auto-refinement configuration:
   - Toggle auto-refinement on/off
   - Set risk refinement interval
   - Set gas refinement interval
3. Test manual risk refinement:
   - Select "Run Risk Parameter Refinement"
   - Review refinement data summary
   - Confirm refinement
   - Verify parameter adjustments
4. Test manual gas refinement:
   - Select "Run Gas Parameter Refinement"
   - Review refinement data summary
   - Confirm refinement
   - Verify multiplier adjustments
5. Test performance data management:
   - View performance metrics
   - Clear specific data types
   - Verify data was cleared

**Expected Result**: Auto-refinement settings should be saved correctly, manual refinement should adjust parameters based on performance data, and data management functions should work as expected.

## 12. Performance Tracking Testing

Test the performance tracking functionality:

1. Execute several test trades:
   - Buy a token
   - Sell a token
   - Set up and execute a limit order
2. Execute several transactions:
   - Transactions with different priority levels
   - Transactions during different congestion levels
3. Check performance metrics:
   - Select "Refinement" from the menu
   - Verify trade data is recorded
   - Verify transaction data is recorded
   - Check calculated metrics (win rate, success rate, etc.)
4. Test data persistence:
   - Restart the bot
   - Verify performance data is still available

**Expected Result**: All trades and transactions should be recorded correctly, performance metrics should be calculated accurately, and data should persist between bot restarts.

## Troubleshooting Common Issues

### API Connection Issues

If you encounter API connection issues:

1. Verify your API keys are correct
2. Check your internet connection
3. Ensure the API service is operational
4. Check for rate limiting (you may need to wait before trying again)

### Transaction Failures

If transactions fail:

1. Ensure you have enough SOL for the transaction and fees
2. Check that the token has sufficient liquidity
3. Verify the RPC endpoint is responsive
4. Try with a smaller amount

### Data Not Updating

If data doesn't update:

1. Check the monitoring interval settings
2. Restart the bot
3. Verify API keys have necessary permissions
4. Check logs for error messages

## Reporting Issues

If you encounter persistent issues:

1. Check the log file at `~/.solana-trading-bot/logs/trading_bot.log`
2. Note the exact steps to reproduce the issue
3. Report the issue with log excerpts and reproduction steps
