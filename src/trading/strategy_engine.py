"""
Advanced trading strategy engine for the Solana Memecoin Trading Bot.
Implements ML-based trading strategies, backtesting, and strategy optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from src.utils.performance_optimizer import cache_result, parallel_processing
from src.trading.technical_analysis import technical_analyzer
from src.trading.token_analytics import token_analytics
from src.trading.risk_manager import risk_manager

logger = logging.getLogger(__name__)

@dataclass
class StrategyMetrics:
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: float
    total_trades: int
    profitable_trades: int

class StrategyEngine:
    def __init__(self):
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.active_strategies: Dict[str, bool] = {}
        self.ml_models: Dict[str, Any] = {}
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.backtest_results: Dict[str, Dict[str, Any]] = {}
        
        # Load saved ML models
        self._load_ml_models()
        
    def _load_ml_models(self) -> None:
        """Load saved ML models from disk."""
        models_dir = Path("models")
        if not models_dir.exists():
            models_dir.mkdir(parents=True)
            return
            
        for model_file in models_dir.glob("*.joblib"):
            try:
                model = joblib.load(model_file)
                self.ml_models[model_file.stem] = model
                logger.info(f"Loaded ML model: {model_file.stem}")
            except Exception as e:
                logger.error(f"Error loading ML model {model_file}: {e}")
    
    @cache_result(ttl=300)
    def generate_features(self, token_address: str) -> pd.DataFrame:
        """
        Generate features for ML model prediction.
        
        Args:
            token_address: The token's address
            
        Returns:
            DataFrame containing features
        """
        try:
            # Get technical indicators
            ta_data = technical_analyzer.get_indicators(token_address)
            
            # Get token analytics
            analytics = token_analytics.get_token_analytics(token_address)
            
            # Get risk metrics
            risk_metrics = risk_manager.calculate_token_risk_metrics(token_address)
            
            # Combine features
            features = {
                # Technical indicators
                'rsi': ta_data.get('rsi', 50),
                'macd': ta_data.get('macd', 0),
                'macd_signal': ta_data.get('macd_signal', 0),
                'macd_hist': ta_data.get('macd_hist', 0),
                'bb_upper': ta_data.get('bb_upper', 0),
                'bb_middle': ta_data.get('bb_middle', 0),
                'bb_lower': ta_data.get('bb_lower', 0),
                'volume_ma': ta_data.get('volume_ma', 0),
                'price_ma': ta_data.get('price_ma', 0),
                
                # Token analytics
                'liquidity': analytics.get('liquidity', 0),
                'holder_count': analytics.get('holder_count', 0),
                'market_cap': analytics.get('market_cap', 0),
                'volume_24h': analytics.get('volume_24h', 0),
                
                # Risk metrics
                'volatility': risk_metrics.volatility,
                'liquidity_score': risk_metrics.liquidity_score,
                'holder_distribution': risk_metrics.holder_distribution,
                'contract_risk': risk_metrics.contract_risk,
                'social_sentiment': risk_metrics.social_sentiment,
                'overall_risk_score': risk_metrics.overall_risk_score
            }
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error generating features for {token_address}: {e}")
            return pd.DataFrame()
    
    def predict_entry_signal(self, token_address: str) -> Tuple[bool, float]:
        """
        Predict entry signal using ML model.
        
        Args:
            token_address: The token's address
            
        Returns:
            Tuple of (should_enter, confidence)
        """
        try:
            # Generate features
            features = self.generate_features(token_address)
            if features.empty:
                return False, 0.0
                
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Get predictions from all models
            predictions = []
            confidences = []
            
            for model_name, model in self.ml_models.items():
                pred = model.predict(scaled_features)[0]
                prob = model.predict_proba(scaled_features)[0]
                predictions.append(pred)
                confidences.append(prob[1] if pred == 1 else prob[0])
            
            # Ensemble prediction
            final_prediction = np.mean(predictions) > 0.5
            confidence = np.mean(confidences)
            
            return final_prediction, confidence
            
        except Exception as e:
            logger.error(f"Error predicting entry signal for {token_address}: {e}")
            return False, 0.0
    
    def backtest_strategy(self, strategy_name: str, token_address: str,
                         start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Backtest a trading strategy.
        
        Args:
            strategy_name: Name of the strategy
            token_address: Token to backtest
            start_date: Start date
            end_date: End date
            
        Returns:
            Backtest results
        """
        try:
            # Get historical data
            historical_data = token_analytics.get_historical_data(
                token_address, start_date, end_date
            )
            
            if not historical_data:
                return {"error": "No historical data available"}
            
            # Initialize backtest metrics
            trades = []
            current_position = None
            entry_price = 0.0
            entry_time = None
            
            # Run backtest
            for timestamp, data in historical_data.items():
                # Generate features for this point in time
                features = self.generate_features(token_address)
                
                # Get strategy signal
                signal = self._get_strategy_signal(strategy_name, features)
                
                # Process signal
                if signal > 0 and current_position is None:  # Buy signal
                    current_position = "long"
                    entry_price = data['price']
                    entry_time = timestamp
                elif signal < 0 and current_position == "long":  # Sell signal
                    exit_price = data['price']
                    profit_loss = (exit_price - entry_price) / entry_price * 100
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit_loss': profit_loss,
                        'duration': (timestamp - entry_time).total_seconds() / 3600  # hours
                    })
                    
                    current_position = None
            
            # Calculate strategy metrics
            metrics = self._calculate_strategy_metrics(trades)
            
            # Store results
            self.backtest_results[strategy_name] = {
                'trades': trades,
                'metrics': metrics,
                'token_address': token_address,
                'start_date': start_date,
                'end_date': end_date
            }
            
            return self.backtest_results[strategy_name]
            
        except Exception as e:
            logger.error(f"Error backtesting strategy {strategy_name}: {e}")
            return {"error": str(e)}
    
    def _calculate_strategy_metrics(self, trades: List[Dict[str, Any]]) -> StrategyMetrics:
        """Calculate strategy performance metrics."""
        if not trades:
            return StrategyMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
        
        # Calculate basic metrics
        total_trades = len(trades)
        profitable_trades = sum(1 for trade in trades if trade['profit_loss'] > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit factor
        total_profit = sum(trade['profit_loss'] for trade in trades if trade['profit_loss'] > 0)
        total_loss = abs(sum(trade['profit_loss'] for trade in trades if trade['profit_loss'] < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio
        returns = [trade['profit_loss'] for trade in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 else 0.0
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumsum(returns)
        max_drawdown = 0.0
        peak = cumulative_returns[0]
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate average trade duration
        avg_duration = np.mean([trade['duration'] for trade in trades])
        
        return StrategyMetrics(
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_trade_duration=avg_duration,
            total_trades=total_trades,
            profitable_trades=profitable_trades
        )
    
    def optimize_strategy(self, strategy_name: str, token_address: str,
                         optimization_params: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy_name: Name of the strategy
            token_address: Token to optimize for
            optimization_params: Dictionary of parameters to optimize
            
        Returns:
            Optimization results
        """
        try:
            # Get historical data for optimization
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Use last 30 days for optimization
            
            historical_data = token_analytics.get_historical_data(
                token_address, start_date, end_date
            )
            
            if not historical_data:
                return {"error": "No historical data available for optimization"}
            
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations(optimization_params)
            
            # Test each parameter combination
            best_metrics = None
            best_params = None
            best_sharpe = float('-inf')
            
            for params in param_combinations:
                # Update strategy parameters
                self._update_strategy_params(strategy_name, params)
                
                # Run backtest with these parameters
                results = self.backtest_strategy(strategy_name, token_address, start_date, end_date)
                
                if 'error' in results:
                    continue
                
                metrics = results['metrics']
                
                # Use Sharpe ratio as optimization metric
                if metrics.sharpe_ratio > best_sharpe:
                    best_sharpe = metrics.sharpe_ratio
                    best_metrics = metrics
                    best_params = params
            
            if best_params is None:
                return {"error": "No valid parameter combinations found"}
            
            # Update strategy with best parameters
            self._update_strategy_params(strategy_name, best_params)
            
            return {
                'best_parameters': best_params,
                'metrics': best_metrics,
                'optimization_date': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing strategy {strategy_name}: {e}")
            return {"error": str(e)}
    
    def _generate_param_combinations(self, params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for grid search."""
        import itertools
        
        keys = params.keys()
        values = params.values()
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _update_strategy_params(self, strategy_name: str, params: Dict[str, Any]) -> None:
        """Update strategy parameters."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update(params)
    
    def train_ml_model(self, strategy_name: str, training_data: List[Dict[str, Any]]) -> bool:
        """
        Train a new ML model for strategy prediction.
        
        Args:
            strategy_name: Name of the strategy
            training_data: List of training examples
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            # Prepare training data
            X = pd.DataFrame([example['features'] for example in training_data])
            y = np.array([example['label'] for example in training_data])
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_scaled, y)
            
            # Save model
            model_path = Path("models") / f"{strategy_name}.joblib"
            joblib.dump(model, model_path)
            
            # Update models dictionary
            self.ml_models[strategy_name] = model
            
            logger.info(f"Trained new ML model for strategy: {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error training ML model for {strategy_name}: {e}")
            return False

# Global instance
strategy_engine = StrategyEngine() 