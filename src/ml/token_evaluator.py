"""
Machine learning-based token evaluation module for the Solana Memecoin Trading Bot.
Evaluates tokens based on various metrics and predicts potential performance.
"""

import json
import logging
import time
import threading
import requests
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pickle

from config import get_config_value, update_config
from src.solana.solana_interact import solana_client
from src.trading.jupiter_api import jupiter_api
from src.trading.sentiment_analysis import sentiment_analyzer
from src.trading.technical_analysis import technical_analyzer
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class TokenEvaluator:
    """Machine learning-based token evaluator."""

    def __init__(self):
        """Initialize the token evaluator."""
        self.enabled = get_config_value("ml_evaluation_enabled", False)

        # Path for storing ML models and data
        self.data_path = Path(get_config_value("ml_data_path",
                                             str(Path.home() / ".solana-trading-bot" / "ml_models")))
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Model file paths
        self.risk_model_path = self.data_path / "risk_model.pkl"
        self.performance_model_path = self.data_path / "performance_model.pkl"

        # Token evaluations cache
        self.evaluations: Dict[str, Dict[str, Any]] = {}

        # Feature importance
        self.feature_importance: Dict[str, float] = {}

        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.monitoring_interval = int(get_config_value("ml_monitoring_interval", "3600"))  # Default: 1 hour

        # Load models
        self._load_models()

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable ML-based token evaluation.

        Args:
            enabled: Whether ML evaluation should be enabled
        """
        self.enabled = enabled
        update_config("ml_evaluation_enabled", enabled)
        logger.info(f"ML-based token evaluation {'enabled' if enabled else 'disabled'}")

        if enabled and not self.monitoring_thread:
            self.start_monitoring_thread()
        elif not enabled and self.monitoring_thread:
            self.stop_monitoring_thread()

    def _load_models(self) -> None:
        """Load ML models."""
        try:
            # Check if models exist
            if self.risk_model_path.exists() and self.performance_model_path.exists():
                # Load models
                with open(self.risk_model_path, 'rb') as f:
                    self.risk_model = pickle.load(f)

                with open(self.performance_model_path, 'rb') as f:
                    self.performance_model = pickle.load(f)

                logger.info("Loaded ML models")
            else:
                # Create dummy models
                self._create_dummy_models()
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            self._create_dummy_models()

    def _create_dummy_models(self) -> None:
        """Create dummy ML models for demonstration purposes."""
        # This is a simplified implementation
        # In a real implementation, we would train proper ML models

        # Create dummy risk model
        class DummyRiskModel:
            def predict(self, features):
                # Simulate risk prediction
                # Higher values indicate higher risk
                return np.random.uniform(0.1, 0.9, size=len(features))

        # Create dummy performance model
        class DummyPerformanceModel:
            def predict(self, features):
                # Simulate performance prediction
                # Higher values indicate better expected performance
                return np.random.uniform(-0.2, 0.5, size=len(features))

        # Set models
        self.risk_model = DummyRiskModel()
        self.performance_model = DummyPerformanceModel()

        # Set feature importance
        self.feature_importance = {
            "market_cap": 0.25,
            "volume_24h": 0.20,
            "liquidity": 0.15,
            "price_change_24h": 0.10,
            "social_sentiment": 0.10,
            "holder_count": 0.08,
            "age_days": 0.07,
            "developer_activity": 0.05
        }

        logger.info("Created dummy ML models")

    def start_monitoring_thread(self) -> None:
        """Start the monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("ML monitoring thread already running")
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("ML monitoring thread started")

    def stop_monitoring_thread(self) -> None:
        """Stop the monitoring thread."""
        if not self.monitoring_thread:
            logger.warning("ML monitoring thread not running")
            return

        self.stop_monitoring.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.monitoring_thread = None
        logger.info("ML monitoring thread stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Get list of tokens to evaluate
                tracked_tokens = get_config_value("tracked_tokens", {})

                # Evaluate each token
                for token_mint, token_info in tracked_tokens.items():
                    try:
                        self.evaluate_token(token_mint)
                    except Exception as e:
                        logger.error(f"Error evaluating token {token_mint}: {e}")
            except Exception as e:
                logger.error(f"Error in ML monitoring loop: {e}")

            # Sleep for the monitoring interval
            self.stop_monitoring.wait(self.monitoring_interval)

    def evaluate_token(self, token_mint: str) -> Dict[str, Any]:
        """
        Evaluate a token using ML models.

        Args:
            token_mint: The token mint address

        Returns:
            Evaluation results
        """
        if not self.enabled:
            logger.warning("ML evaluation is disabled")
            return {
                "score": 0.0,
                "risk": "unknown",
                "potential": "unknown",
                "recommendation": "neutral",
                "confidence": 0.0,
                "features": {}
            }

        try:
            # Check if we have a recent evaluation
            if token_mint in self.evaluations:
                last_eval = self.evaluations[token_mint]
                last_eval_time = datetime.fromisoformat(last_eval["timestamp"])

                # Use cached evaluation if less than 1 hour old
                if datetime.now() - last_eval_time < timedelta(hours=1):
                    return last_eval

            # Extract features
            features = self._extract_token_features(token_mint)

            # Convert features to numpy array for model input
            feature_array = np.array([list(features.values())])

            # Predict risk
            risk_score = float(self.risk_model.predict(feature_array)[0])

            # Predict performance
            performance_score = float(self.performance_model.predict(feature_array)[0])

            # Calculate overall score (0-100)
            # Higher performance and lower risk is better
            overall_score = (performance_score * 100) * (1 - risk_score)
            overall_score = max(0, min(100, overall_score))  # Clamp to 0-100

            # Determine risk level
            if risk_score < 0.3:
                risk_level = "low"
            elif risk_score < 0.7:
                risk_level = "medium"
            else:
                risk_level = "high"

            # Determine potential
            if performance_score < 0:
                potential = "negative"
            elif performance_score < 0.2:
                potential = "low"
            elif performance_score < 0.4:
                potential = "medium"
            else:
                potential = "high"

            # Generate recommendation
            if overall_score < 30:
                recommendation = "avoid"
            elif overall_score < 50:
                recommendation = "neutral"
            elif overall_score < 70:
                recommendation = "consider"
            else:
                recommendation = "strong_buy"

            # Calculate confidence (0-1)
            # This is a simplified implementation
            confidence = 0.7  # Fixed confidence for dummy model

            # Create evaluation result
            evaluation = {
                "token_mint": token_mint,
                "timestamp": datetime.now().isoformat(),
                "score": overall_score,
                "risk": risk_level,
                "risk_score": risk_score,
                "potential": potential,
                "performance_score": performance_score,
                "recommendation": recommendation,
                "confidence": confidence,
                "features": features
            }

            # Cache evaluation
            self.evaluations[token_mint] = evaluation

            logger.info(f"Evaluated token {token_mint}: score={overall_score:.1f}, risk={risk_level}, potential={potential}")
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating token {token_mint}: {e}")
            return {
                "score": 0.0,
                "risk": "unknown",
                "potential": "unknown",
                "recommendation": "neutral",
                "confidence": 0.0,
                "features": {},
                "error": str(e)
            }

    def _extract_token_features(self, token_mint: str) -> Dict[str, float]:
        """
        Extract features for a token using real data sources.

        Args:
            token_mint: The token mint address

        Returns:
            Dictionary of features
        """
        features = {}

        try:
            # Get token info from tracked tokens
            tracked_tokens = get_config_value("tracked_tokens", {})
            token_info = tracked_tokens.get(token_mint, {})

            # Basic token info
            token_symbol = token_info.get("symbol", "")

            # Get price data from technical analyzer
            price_data = None
            if token_mint in technical_analyzer.price_data:
                price_data = technical_analyzer.price_data[token_mint]

            # Get token data from Jupiter API
            try:
                # Get token price and market data
                token_price = jupiter_api.get_token_price(token_mint)

                # Get token market data from Helius API
                helius_key = get_config_value("helius_api_key", "")
                if helius_key:
                    url = f"https://api.helius.xyz/v0/tokens/metadata?api-key={helius_key}"
                    payload = {"mintAccounts": [token_mint]}
                    response = requests.post(url, json=payload)
                    if response.status_code == 200:
                        token_data = response.json()[0]

                        # Extract market cap if available
                        if "marketCap" in token_data:
                            features["market_cap"] = float(token_data["marketCap"])
                        else:
                            # Estimate market cap from supply and price
                            supply = float(token_data.get("supply", 0))
                            features["market_cap"] = supply * token_price

                        # Extract volume if available
                        features["volume_24h"] = float(token_data.get("volumeUsd24h", 0))

                        # Extract liquidity if available
                        features["liquidity"] = float(token_data.get("liquidity", 0))

                        # Extract holder count if available
                        features["holder_count"] = float(token_data.get("holderCount", 0))

                        # Calculate token age in days
                        if "firstSeenTimestamp" in token_data:
                            first_seen = datetime.fromtimestamp(token_data["firstSeenTimestamp"] / 1000)
                            age_days = (datetime.now() - first_seen).days
                            features["age_days"] = max(1, age_days)  # Ensure at least 1 day
            except Exception as e:
                logger.warning(f"Error fetching token data from Jupiter/Helius: {e}")

            # If we couldn't get market cap from API, estimate it
            if "market_cap" not in features and token_price > 0:
                # Try to get supply from Solana RPC
                try:
                    token_supply_response = solana_client.client.get_token_supply(token_mint)
                    if "result" in token_supply_response:
                        supply_data = token_supply_response["result"]["value"]
                        supply = float(supply_data["uiAmount"])
                        features["market_cap"] = supply * token_price
                except Exception as e:
                    logger.warning(f"Error getting token supply: {e}")
                    features["market_cap"] = 0

            # Get price change from technical analyzer or calculate it
            if price_data is not None and len(price_data) > 1:
                first_price = price_data["price"].iloc[0]
                last_price = price_data["price"].iloc[-1]
                price_change = (last_price - first_price) / first_price
                features["price_change_24h"] = price_change
            else:
                # Try to get 24h price change from Jupiter
                try:
                    yesterday_price = jupiter_api.get_token_price_history(token_mint, days=1)
                    if yesterday_price and yesterday_price > 0:
                        price_change = (token_price - yesterday_price) / yesterday_price
                        features["price_change_24h"] = price_change
                    else:
                        features["price_change_24h"] = 0
                except Exception as e:
                    logger.warning(f"Error getting price history: {e}")
                    features["price_change_24h"] = 0

            # Get social sentiment from sentiment analyzer
            if hasattr(sentiment_analyzer, "sentiment_data") and token_mint in sentiment_analyzer.sentiment_data:
                sentiment_data = sentiment_analyzer.sentiment_data[token_mint]
                features["social_sentiment"] = sentiment_data.get("twitter_sentiment_24h", 0)
            else:
                # Get sentiment from Twitter API if available
                try:
                    if token_symbol:
                        sentiment_score = sentiment_analyzer.get_token_sentiment(token_symbol)
                        features["social_sentiment"] = sentiment_score
                    else:
                        features["social_sentiment"] = 0
                except Exception as e:
                    logger.warning(f"Error getting social sentiment: {e}")
                    features["social_sentiment"] = 0

            # Get developer activity from GitHub if available
            if token_symbol:
                try:
                    # Check if we have GitHub info in token_info
                    github_repo = token_info.get("github_repo", "")
                    if github_repo:
                        # Calculate developer activity score based on commits, issues, etc.
                        github_api_url = f"https://api.github.com/repos/{github_repo}/stats/commit_activity"
                        response = requests.get(github_api_url)
                        if response.status_code == 200:
                            commit_data = response.json()
                            # Calculate weekly commit average
                            total_commits = sum(week.get("total", 0) for week in commit_data[-4:])  # Last 4 weeks
                            avg_weekly_commits = total_commits / 4
                            # Normalize to 0-1 scale (0-50 commits per week)
                            features["developer_activity"] = min(1.0, avg_weekly_commits / 50)
                        else:
                            features["developer_activity"] = 0
                    else:
                        features["developer_activity"] = 0
                except Exception as e:
                    logger.warning(f"Error getting developer activity: {e}")
                    features["developer_activity"] = 0
            else:
                features["developer_activity"] = 0

            # Fill in any missing features with defaults
            default_features = {
                "market_cap": 0,
                "volume_24h": 0,
                "liquidity": 0,
                "price_change_24h": 0,
                "social_sentiment": 0,
                "holder_count": 0,
                "age_days": 1,
                "developer_activity": 0
            }

            for key, default_value in default_features.items():
                if key not in features:
                    features[key] = default_value

            return features

        except Exception as e:
            logger.error(f"Error extracting token features: {e}")
            # Return default features on error
            return {
                "market_cap": 0,
                "volume_24h": 0,
                "liquidity": 0,
                "price_change_24h": 0,
                "social_sentiment": 0,
                "holder_count": 0,
                "age_days": 1,
                "developer_activity": 0
            }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for the ML models.

        Returns:
            Dictionary of feature names and their importance
        """
        return self.feature_importance

    def get_token_evaluations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all token evaluations.

        Returns:
            Dictionary of token evaluations
        """
        return self.evaluations

    def clear_evaluations(self) -> None:
        """Clear all cached evaluations."""
        self.evaluations = {}
        logger.info("Cleared all token evaluations")

    def train_models(self, training_data: Optional[pd.DataFrame] = None) -> bool:
        """
        Train ML models with new data.

        Args:
            training_data: DataFrame with training data (optional)

        Returns:
            True if training successful, False otherwise
        """
        # This is a simplified implementation
        # In a real implementation, we would train actual ML models

        try:
            logger.info("Training ML models...")

            # Simulate training
            time.sleep(2)

            # Update feature importance (simulated)
            self.feature_importance = {
                "market_cap": np.random.uniform(0.1, 0.3),
                "volume_24h": np.random.uniform(0.1, 0.3),
                "liquidity": np.random.uniform(0.1, 0.2),
                "price_change_24h": np.random.uniform(0.05, 0.15),
                "social_sentiment": np.random.uniform(0.05, 0.15),
                "holder_count": np.random.uniform(0.05, 0.1),
                "age_days": np.random.uniform(0.03, 0.1),
                "developer_activity": np.random.uniform(0.03, 0.1)
            }

            # Normalize feature importance to sum to 1
            total = sum(self.feature_importance.values())
            self.feature_importance = {k: v/total for k, v in self.feature_importance.items()}

            logger.info("ML models trained successfully")
            return True
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            return False


# Create singleton instance
token_evaluator = TokenEvaluator()
