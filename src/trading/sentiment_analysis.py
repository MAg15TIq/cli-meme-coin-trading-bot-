"""
Social sentiment analysis module for the Solana Memecoin Trading Bot.
Analyzes social media for token mentions and sentiment.
"""

import json
import logging
import time
import threading
import re
import requests
from typing import Dict, Any, Optional, List, Union, Set
from datetime import datetime, timedelta
import statistics

from config import get_config_value, update_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzer for social media sentiment."""

    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.enabled = get_config_value("sentiment_analysis_enabled", False)
        self.twitter_api_key = get_config_value("twitter_api_key", "")
        self.twitter_api_secret = get_config_value("twitter_api_secret", "")
        self.twitter_bearer_token = get_config_value("twitter_bearer_token", "")

        # Tokens to monitor
        self.monitored_tokens: Dict[str, Dict[str, Any]] = get_config_value("monitored_tokens", {})

        # Sentiment data: token_mint -> sentiment data
        self.sentiment_data: Dict[str, Dict[str, Any]] = {}

        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.monitoring_interval = int(get_config_value("sentiment_monitoring_interval", "3600"))  # Default: 1 hour

        # Load sentiment data
        self._load_sentiment_data()

    def _load_sentiment_data(self) -> None:
        """Load sentiment data from config."""
        self.sentiment_data = get_config_value("sentiment_data", {})
        logger.info(f"Loaded sentiment data for {len(self.sentiment_data)} tokens")

    def _save_sentiment_data(self) -> None:
        """Save sentiment data to config."""
        update_config("sentiment_data", self.sentiment_data)
        logger.info(f"Saved sentiment data for {len(self.sentiment_data)} tokens")

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable sentiment analysis.

        Args:
            enabled: Whether sentiment analysis should be enabled
        """
        self.enabled = enabled
        update_config("sentiment_analysis_enabled", enabled)
        logger.info(f"Sentiment analysis {'enabled' if enabled else 'disabled'}")

        # Start or stop monitoring
        if enabled:
            self.start_monitoring()
        else:
            self.stop_monitoring_thread()

    def add_monitored_token(self, token_mint: str, token_name: str, token_symbol: str, keywords: List[str]) -> None:
        """
        Add a token to monitor for sentiment.

        Args:
            token_mint: The token mint address
            token_name: The name of the token
            token_symbol: The symbol of the token
            keywords: Additional keywords to monitor
        """
        self.monitored_tokens[token_mint] = {
            "name": token_name,
            "symbol": token_symbol,
            "keywords": keywords,
            "added_at": datetime.now().isoformat()
        }

        update_config("monitored_tokens", self.monitored_tokens)
        logger.info(f"Added token {token_name} ({token_mint}) to sentiment monitoring")

        # Initialize sentiment data
        if token_mint not in self.sentiment_data:
            self.sentiment_data[token_mint] = {
                "last_updated": datetime.now().isoformat(),
                "twitter_mentions_24h": 0,
                "twitter_sentiment_24h": 0.0,
                "twitter_mentions_history": [],
                "twitter_sentiment_history": [],
                "alerts": []
            }
            self._save_sentiment_data()

    def remove_monitored_token(self, token_mint: str) -> bool:
        """
        Remove a token from sentiment monitoring.

        Args:
            token_mint: The token mint address

        Returns:
            True if the token was removed, False if it wasn't monitored
        """
        if token_mint in self.monitored_tokens:
            del self.monitored_tokens[token_mint]
            update_config("monitored_tokens", self.monitored_tokens)
            logger.info(f"Removed token {token_mint} from sentiment monitoring")
            return True
        return False

    def get_monitored_tokens(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all monitored tokens.

        Returns:
            Dictionary of monitored tokens
        """
        return self.monitored_tokens

    def get_token_sentiment(self, token_mint: str) -> Optional[Dict[str, Any]]:
        """
        Get sentiment data for a token.

        Args:
            token_mint: The token mint address

        Returns:
            Sentiment data if available, None otherwise
        """
        return self.sentiment_data.get(token_mint)

    def start_monitoring(self) -> bool:
        """
        Start sentiment monitoring.

        Returns:
            True if started successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Sentiment analysis is disabled")
            return False

        if not self.monitored_tokens:
            logger.warning("No tokens to monitor")
            return False

        if not self.twitter_bearer_token:
            logger.warning("Twitter API credentials not configured")
            return False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.info("Sentiment monitoring already running")
            return True

        # Clear stop event
        self.stop_monitoring.clear()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("Started sentiment monitoring thread")
        return True

    def stop_monitoring_thread(self) -> None:
        """Stop sentiment monitoring."""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            logger.info("Sentiment monitoring not running")
            return

        # Set stop event
        self.stop_monitoring.set()

        # Wait for thread to stop
        self.monitoring_thread.join(timeout=5)

        logger.info("Stopped sentiment monitoring")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Update sentiment for all monitored tokens
                for token_mint, token_info in self.monitored_tokens.items():
                    try:
                        self._update_token_sentiment(token_mint, token_info)
                    except Exception as e:
                        logger.error(f"Error updating sentiment for {token_mint}: {e}")

                # Save sentiment data
                self._save_sentiment_data()
            except Exception as e:
                logger.error(f"Error in sentiment monitoring loop: {e}")

            # Sleep for the monitoring interval
            self.stop_monitoring.wait(self.monitoring_interval)

    def _update_token_sentiment(self, token_mint: str, token_info: Dict[str, Any]) -> None:
        """
        Update sentiment data for a token.

        Args:
            token_mint: The token mint address
            token_info: Token information
        """
        # Get token keywords
        token_name = token_info["name"]
        token_symbol = token_info["symbol"]
        keywords = token_info.get("keywords", [])

        # Create search query
        search_terms = [token_name, token_symbol] + keywords
        search_query = " OR ".join([f'"{term}"' for term in search_terms if term])

        # Get Twitter mentions
        mentions, sentiment = self._get_twitter_sentiment(search_query)

        # Update sentiment data
        if token_mint not in self.sentiment_data:
            self.sentiment_data[token_mint] = {
                "last_updated": datetime.now().isoformat(),
                "twitter_mentions_24h": mentions,
                "twitter_sentiment_24h": sentiment,
                "twitter_mentions_history": [],
                "twitter_sentiment_history": [],
                "alerts": []
            }
        else:
            # Add to history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "mentions": mentions,
                "sentiment": sentiment
            }

            # Keep only last 30 days of history
            history = self.sentiment_data[token_mint]["twitter_mentions_history"]
            history.append(history_entry)

            # Trim history to last 30 entries
            if len(history) > 30:
                history = history[-30:]

            self.sentiment_data[token_mint]["twitter_mentions_history"] = history
            self.sentiment_data[token_mint]["twitter_sentiment_history"] = [
                entry["sentiment"] for entry in history
            ]

            # Update current values
            self.sentiment_data[token_mint]["last_updated"] = datetime.now().isoformat()
            self.sentiment_data[token_mint]["twitter_mentions_24h"] = mentions
            self.sentiment_data[token_mint]["twitter_sentiment_24h"] = sentiment

            # Check for significant changes
            self._check_sentiment_alerts(token_mint, token_name, mentions, sentiment)

        logger.info(f"Updated sentiment for {token_name}: {mentions} mentions, sentiment: {sentiment:.2f}")

    def _get_twitter_sentiment(self, query: str) -> tuple[int, float]:
        """
        Get Twitter mentions and sentiment for a query using Twitter API v2.

        Args:
            query: The search query

        Returns:
            Tuple of (mentions_count, sentiment_score)
        """
        if not self.twitter_bearer_token:
            logger.warning("Twitter API token not configured, cannot get real sentiment data")
            return 0, 0.0

        try:
            # Set up Twitter API request (Twitter API v2)
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                "Authorization": f"Bearer {self.twitter_bearer_token}"
            }
            params = {
                "query": query,
                "max_results": 100,  # Maximum allowed by API
                "tweet.fields": "created_at,public_metrics,lang,context_annotations,entities",
                "expansions": "author_id,referenced_tweets.id",
                "user.fields": "name,username,public_metrics,verified"
            }

            # Make request
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            # Parse response
            data = response.json()
            tweets = data.get("data", [])
            users = {user["id"]: user for user in data.get("includes", {}).get("users", [])}

            # Count mentions
            mentions = len(tweets)

            if mentions == 0:
                return 0, 0.0

            # Calculate sentiment using a more sophisticated approach
            # We'll use a combination of metrics:
            # 1. Engagement metrics (likes, retweets, etc.)
            # 2. User influence (follower count, verified status)
            # 3. Basic sentiment analysis using keyword matching

            # Positive and negative sentiment keywords
            positive_keywords = [
                "bullish", "moon", "mooning", "gem", "buy", "buying", "hodl", "hold", "holding",
                "good", "great", "amazing", "excellent", "profit", "profits", "win", "winning",
                "pump", "pumping", "up", "high", "higher", "rise", "rising", "gain", "gains",
                "potential", "opportunity", "promising", "excited", "exciting", "🚀", "💎", "🔥"
            ]

            negative_keywords = [
                "bearish", "dump", "dumping", "sell", "selling", "sold", "crash", "crashing",
                "bad", "terrible", "awful", "poor", "loss", "losses", "lose", "losing",
                "down", "low", "lower", "fall", "falling", "drop", "dropping", "decrease",
                "scam", "rug", "rugpull", "fake", "fraud", "ponzi", "avoid", "warning", "⚠️"
            ]

            total_sentiment = 0.0
            total_weight = 0.0

            for tweet in tweets:
                # Get tweet text and convert to lowercase
                text = tweet.get("text", "").lower()

                # Get author info
                author_id = tweet.get("author_id")
                author = users.get(author_id, {})

                # Calculate base sentiment score from text
                sentiment_score = 0.0

                # Count positive and negative keywords
                positive_count = sum(1 for keyword in positive_keywords if keyword.lower() in text)
                negative_count = sum(1 for keyword in negative_keywords if keyword.lower() in text)

                # Calculate raw sentiment from keyword counts
                if positive_count > 0 or negative_count > 0:
                    sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)

                # Get engagement metrics
                metrics = tweet.get("public_metrics", {})
                likes = metrics.get("like_count", 0)
                retweets = metrics.get("retweet_count", 0)
                replies = metrics.get("reply_count", 0)
                quotes = metrics.get("quote_count", 0)

                # Calculate engagement score (0-1)
                engagement = (likes + retweets * 2 + replies + quotes) / 100
                engagement = min(1.0, engagement)  # Cap at 1.0

                # Get user influence
                followers = author.get("public_metrics", {}).get("followers_count", 0)
                verified = author.get("verified", False)

                # Calculate influence score (0-1)
                influence = min(1.0, followers / 10000)  # Cap at 1.0
                if verified:
                    influence = min(1.0, influence * 1.5)  # Boost for verified users

                # Calculate weight for this tweet
                weight = 1.0 + engagement + influence

                # Add weighted sentiment to total
                total_sentiment += sentiment_score * weight
                total_weight += weight

            # Calculate final sentiment score (-1 to 1)
            if total_weight > 0:
                final_sentiment = total_sentiment / total_weight
            else:
                final_sentiment = 0.0

            logger.info(f"Twitter sentiment analysis: {mentions} mentions, sentiment: {final_sentiment:.2f}")
            return mentions, final_sentiment

        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {e}")
            return 0, 0.0

    def _check_sentiment_alerts(self, token_mint: str, token_name: str, mentions: int, sentiment: float) -> None:
        """
        Check for significant sentiment changes and create alerts.

        Args:
            token_mint: The token mint address
            token_name: The name of the token
            mentions: Current mentions count
            sentiment: Current sentiment score
        """
        # Get historical data
        history = self.sentiment_data[token_mint]["twitter_mentions_history"]

        # Need at least 2 data points
        if len(history) < 2:
            return

        # Get previous mentions
        prev_mentions = history[-2]["mentions"] if len(history) > 1 else 0

        # Calculate mention change percentage
        if prev_mentions > 0:
            mention_change_pct = ((mentions - prev_mentions) / prev_mentions) * 100
        else:
            mention_change_pct = 100 if mentions > 0 else 0

        # Get sentiment history
        sentiment_history = [entry["sentiment"] for entry in history[:-1]]  # Exclude current

        # Calculate sentiment statistics
        if sentiment_history:
            avg_sentiment = statistics.mean(sentiment_history)
            if len(sentiment_history) > 1:
                std_sentiment = statistics.stdev(sentiment_history)
            else:
                std_sentiment = 0.1  # Default if not enough data

            # Calculate z-score for current sentiment
            if std_sentiment > 0:
                sentiment_z_score = (sentiment - avg_sentiment) / std_sentiment
            else:
                sentiment_z_score = 0
        else:
            sentiment_z_score = 0

        # Check for alerts
        alerts = []

        # Mention spike alert (>100% increase)
        if mention_change_pct >= 100:
            alerts.append({
                "type": "mention_spike",
                "timestamp": datetime.now().isoformat(),
                "message": f"Mentions increased by {mention_change_pct:.1f}% (from {prev_mentions} to {mentions})",
                "severity": "high" if mention_change_pct >= 200 else "medium"
            })

        # Sentiment shift alert (z-score > 2)
        if abs(sentiment_z_score) >= 2:
            direction = "positive" if sentiment_z_score > 0 else "negative"
            alerts.append({
                "type": "sentiment_shift",
                "timestamp": datetime.now().isoformat(),
                "message": f"Sentiment shifted significantly in a {direction} direction (z-score: {sentiment_z_score:.1f})",
                "severity": "high" if abs(sentiment_z_score) >= 3 else "medium"
            })

        # Add alerts to sentiment data
        if alerts:
            # Add new alerts
            self.sentiment_data[token_mint]["alerts"] = alerts + self.sentiment_data[token_mint].get("alerts", [])

            # Keep only last 10 alerts
            self.sentiment_data[token_mint]["alerts"] = self.sentiment_data[token_mint]["alerts"][:10]

            # Log alerts
            for alert in alerts:
                logger.info(f"Sentiment alert for {token_name}: {alert['message']} (severity: {alert['severity']})")

    def get_sentiment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of sentiment data for all monitored tokens.

        Returns:
            Dictionary with sentiment summary
        """
        summary = {
            "tokens": {},
            "alerts": [],
            "last_updated": datetime.now().isoformat()
        }

        # Process each token
        for token_mint, token_data in self.sentiment_data.items():
            if token_mint in self.monitored_tokens:
                token_info = self.monitored_tokens[token_mint]
                token_name = token_info["name"]

                # Add token summary
                summary["tokens"][token_mint] = {
                    "name": token_name,
                    "symbol": token_info["symbol"],
                    "mentions_24h": token_data["twitter_mentions_24h"],
                    "sentiment_24h": token_data["twitter_sentiment_24h"],
                    "last_updated": token_data["last_updated"]
                }

                # Add alerts
                for alert in token_data.get("alerts", []):
                    alert_copy = alert.copy()
                    alert_copy["token_mint"] = token_mint
                    alert_copy["token_name"] = token_name
                    summary["alerts"].append(alert_copy)

        # Sort alerts by timestamp (newest first)
        summary["alerts"] = sorted(
            summary["alerts"],
            key=lambda x: x["timestamp"],
            reverse=True
        )

        return summary


# Create a singleton instance
sentiment_analyzer = SentimentAnalyzer()
