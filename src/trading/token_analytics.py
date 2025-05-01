"""
Token analytics module for the Solana Memecoin Trading Bot.
Provides comprehensive token analytics including social sentiment, holder metrics, market data,
risk assessment, and position sizing recommendations.
"""

import json
import logging
import time
import threading
import requests
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path

from config import get_config_value, update_config
from src.utils.logging_utils import get_logger
from src.trading.helius_api import helius_api
from src.trading.jupiter_api import jupiter_api
from src.trading.sentiment_analysis import sentiment_analyzer
from src.notifications.notification_service import notification_service, NotificationPriority

# Get logger for this module
logger = get_logger(__name__)


class TokenAnalytics:
    """Comprehensive token analytics provider."""

    def __init__(self):
        """Initialize the token analytics provider."""
        self.enabled = get_config_value("token_analytics_enabled", False)
        self.analytics_data: Dict[str, Dict[str, Any]] = {}
        self.analytics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history_entries = int(get_config_value("max_analytics_history_entries", "100"))

        # GitHub API for developer activity
        self.github_token = get_config_value("github_token", "")

        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.monitoring_interval = int(get_config_value("analytics_monitoring_interval", "3600"))  # Default: 1 hour

        # Load analytics data
        self._load_analytics_data()

        # Start monitoring if enabled
        if self.enabled:
            self.start_monitoring()

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable token analytics.

        Args:
            enabled: Whether token analytics should be enabled
        """
        self.enabled = enabled
        update_config("token_analytics_enabled", enabled)
        logger.info(f"Token analytics {'enabled' if enabled else 'disabled'}")

        if enabled and not self.monitoring_thread:
            self.start_monitoring()
        elif not enabled and self.monitoring_thread:
            self.stop_monitoring()

    def _load_analytics_data(self) -> None:
        """Load analytics data from disk."""
        try:
            analytics_file = Path(get_config_value("data_dir", "data")) / "token_analytics.json"
            if analytics_file.exists():
                with open(analytics_file, "r") as f:
                    data = json.load(f)
                    self.analytics_data = data.get("current", {})
                    self.analytics_history = data.get("history", {})
                logger.info(f"Loaded analytics data for {len(self.analytics_data)} tokens")
        except Exception as e:
            logger.error(f"Error loading analytics data: {e}")

    def _save_analytics_data(self) -> None:
        """Save analytics data to disk."""
        try:
            analytics_file = Path(get_config_value("data_dir", "data")) / "token_analytics.json"
            analytics_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "current": self.analytics_data,
                "history": self.analytics_history,
                "last_updated": datetime.now().isoformat()
            }

            with open(analytics_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved analytics data for {len(self.analytics_data)} tokens")
        except Exception as e:
            logger.error(f"Error saving analytics data: {e}")

    def start_monitoring(self) -> None:
        """Start the analytics monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Analytics monitoring thread is already running")
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started token analytics monitoring thread")

    def stop_monitoring(self) -> None:
        """Stop the analytics monitoring thread."""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            logger.warning("Analytics monitoring thread is not running")
            return

        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=5.0)
        logger.info("Stopped token analytics monitoring thread")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Get tracked tokens
                tracked_tokens = get_config_value("tracked_tokens", {})

                # Update analytics for each token
                for token_mint, token_info in tracked_tokens.items():
                    try:
                        self.update_token_analytics(token_mint)
                    except Exception as e:
                        logger.error(f"Error updating analytics for {token_mint}: {e}")

                # Save analytics data
                self._save_analytics_data()
            except Exception as e:
                logger.error(f"Error in analytics monitoring loop: {e}")

            # Sleep until next update
            self.stop_monitoring.wait(self.monitoring_interval)

    def update_token_analytics(self, token_mint: str) -> Dict[str, Any]:
        """
        Update analytics for a specific token.

        Args:
            token_mint: The token mint address

        Returns:
            The updated analytics data
        """
        if not self.enabled:
            logger.warning("Token analytics is disabled")
            return {}

        try:
            # Get token info
            token_info = self._get_token_info(token_mint)
            if not token_info:
                logger.warning(f"No token info found for {token_mint}")
                return {}

            # Create analytics entry if it doesn't exist
            if token_mint not in self.analytics_data:
                self.analytics_data[token_mint] = {
                    "token_mint": token_mint,
                    "token_name": token_info.get("name", ""),
                    "token_symbol": token_info.get("symbol", ""),
                    "first_analyzed": datetime.now().isoformat()
                }

            # Update token info
            self.analytics_data[token_mint]["token_name"] = token_info.get("name", self.analytics_data[token_mint]["token_name"])
            self.analytics_data[token_mint]["token_symbol"] = token_info.get("symbol", self.analytics_data[token_mint]["token_symbol"])

            # Get market data
            market_data = self._get_market_data(token_mint)
            self.analytics_data[token_mint].update(market_data)

            # Get holder metrics
            holder_metrics = self._get_holder_metrics(token_mint)
            self.analytics_data[token_mint].update(holder_metrics)

            # Get social sentiment
            social_sentiment = self._get_social_sentiment(token_mint)
            self.analytics_data[token_mint].update(social_sentiment)

            # Get developer activity
            developer_activity = self._get_developer_activity(token_mint, token_info)
            self.analytics_data[token_mint].update(developer_activity)

            # Update timestamp
            self.analytics_data[token_mint]["last_updated"] = datetime.now().isoformat()

            # Add to history
            self._add_to_history(token_mint, self.analytics_data[token_mint])

            logger.info(f"Updated analytics for {token_info.get('symbol', token_mint)}")
            return self.analytics_data[token_mint]
        except Exception as e:
            logger.error(f"Error updating analytics for {token_mint}: {e}")
            return {}

    def _get_token_info(self, token_mint: str) -> Dict[str, Any]:
        """
        Get basic token information.

        Args:
            token_mint: The token mint address

        Returns:
            Token information
        """
        # First check tracked tokens
        tracked_tokens = get_config_value("tracked_tokens", {})
        if token_mint in tracked_tokens:
            return tracked_tokens[token_mint]

        # Then try Jupiter API
        try:
            token_info = jupiter_api.get_token_info(token_mint)
            if token_info:
                return token_info
        except Exception as e:
            logger.warning(f"Error getting token info from Jupiter: {e}")

        # Finally try Helius API
        try:
            token_metadata = helius_api.get_token_metadata(token_mint)
            if token_metadata:
                return {
                    "name": token_metadata.get("name", ""),
                    "symbol": token_metadata.get("symbol", ""),
                    "decimals": token_metadata.get("decimals", 9),
                    "logo": token_metadata.get("image", "")
                }
        except Exception as e:
            logger.warning(f"Error getting token metadata from Helius: {e}")

        return {}

    def _get_market_data(self, token_mint: str) -> Dict[str, Any]:
        """
        Get market data for a token.

        Args:
            token_mint: The token mint address

        Returns:
            Market data
        """
        market_data = {}

        try:
            # Get current price
            price = jupiter_api.get_token_price(token_mint)
            if price is not None:
                market_data["price"] = price

            # Get token metadata from Helius
            token_metadata = helius_api.get_token_metadata(token_mint)
            if token_metadata:
                # Extract market cap
                if "marketCap" in token_metadata:
                    market_data["market_cap"] = float(token_metadata["marketCap"])
                elif "supply" in token_metadata and price is not None:
                    supply = float(token_metadata.get("supply", 0))
                    market_data["market_cap"] = supply * price

                # Extract volume
                if "volumeUsd24h" in token_metadata:
                    market_data["volume_24h"] = float(token_metadata["volumeUsd24h"])

                # Extract liquidity
                if "liquidity" in token_metadata:
                    market_data["liquidity"] = float(token_metadata["liquidity"])

                # Calculate token age in days
                if "firstSeenTimestamp" in token_metadata:
                    first_seen = datetime.fromtimestamp(token_metadata["firstSeenTimestamp"] / 1000)
                    age_days = (datetime.now() - first_seen).days
                    market_data["age_days"] = max(1, age_days)  # Ensure at least 1 day

            # Get price history for price change calculation
            try:
                yesterday_price = jupiter_api.get_token_price_history(token_mint, days=1)
                if yesterday_price and yesterday_price > 0 and price is not None:
                    price_change = (price - yesterday_price) / yesterday_price
                    market_data["price_change_24h"] = price_change
            except Exception as e:
                logger.warning(f"Error getting price history: {e}")

        except Exception as e:
            logger.error(f"Error getting market data for {token_mint}: {e}")

        return market_data

    def _get_holder_metrics(self, token_mint: str) -> Dict[str, Any]:
        """
        Get holder metrics for a token.

        Args:
            token_mint: The token mint address

        Returns:
            Holder metrics
        """
        holder_metrics = {}

        try:
            # Get token metadata from Helius
            token_metadata = helius_api.get_token_metadata(token_mint)
            if token_metadata:
                # Extract holder count
                if "holderCount" in token_metadata:
                    holder_metrics["holder_count"] = int(token_metadata["holderCount"])

                # Extract top holders if available
                if "topHolders" in token_metadata:
                    holder_metrics["top_holders"] = token_metadata["topHolders"]

                # Calculate holder concentration (% held by top 10 holders)
                if "topHolders" in token_metadata and "supply" in token_metadata:
                    top_holders = token_metadata["topHolders"]
                    total_supply = float(token_metadata["supply"])

                    if total_supply > 0 and top_holders:
                        top_10_amount = sum(float(holder.get("amount", 0)) for holder in top_holders[:10])
                        concentration = (top_10_amount / total_supply) * 100
                        holder_metrics["top10_concentration"] = concentration

            # Get additional holder metrics from Solscan API
            try:
                solscan_url = f"https://public-api.solscan.io/token/holders?tokenAddress={token_mint}&limit=10&offset=0"
                response = requests.get(solscan_url)
                if response.status_code == 200:
                    solscan_data = response.json()

                    # Extract holder count if not already available
                    if "holder_count" not in holder_metrics and "total" in solscan_data:
                        holder_metrics["holder_count"] = int(solscan_data["total"])

                    # Extract holder distribution
                    if "data" in solscan_data:
                        holders_data = solscan_data["data"]
                        if holders_data:
                            # Calculate average holding time if available
                            holding_times = []
                            for holder in holders_data:
                                if "owner" in holder and "amount" in holder:
                                    # Try to get first transaction time for this holder
                                    try:
                                        owner = holder["owner"]
                                        first_tx_url = f"https://public-api.solscan.io/account/transactions?account={owner}&limit=1&tokenAddress={token_mint}"
                                        tx_response = requests.get(first_tx_url)
                                        if tx_response.status_code == 200:
                                            tx_data = tx_response.json()
                                            if tx_data and len(tx_data) > 0:
                                                first_tx_time = tx_data[0].get("blockTime", 0)
                                                if first_tx_time > 0:
                                                    holding_time = (datetime.now().timestamp() - first_tx_time) / 86400  # Convert to days
                                                    holding_times.append(holding_time)
                                    except Exception as e:
                                        logger.debug(f"Error getting holding time: {e}")

                            if holding_times:
                                holder_metrics["avg_holding_time_days"] = sum(holding_times) / len(holding_times)
            except Exception as e:
                logger.warning(f"Error getting holder metrics from Solscan: {e}")

        except Exception as e:
            logger.error(f"Error getting holder metrics for {token_mint}: {e}")

        return holder_metrics

    def _get_social_sentiment(self, token_mint: str) -> Dict[str, Any]:
        """
        Get social sentiment for a token.

        Args:
            token_mint: The token mint address

        Returns:
            Social sentiment data
        """
        social_sentiment = {}

        try:
            # Get sentiment data from sentiment analyzer
            sentiment_data = sentiment_analyzer.get_token_sentiment(token_mint)
            if sentiment_data:
                social_sentiment["twitter_mentions_24h"] = sentiment_data.get("twitter_mentions_24h", 0)
                social_sentiment["twitter_sentiment_24h"] = sentiment_data.get("twitter_sentiment_24h", 0.0)

                # Calculate sentiment trend
                history = sentiment_data.get("twitter_sentiment_history", [])
                if len(history) >= 2:
                    current = history[-1]
                    previous = history[-2]
                    trend = current - previous
                    social_sentiment["sentiment_trend"] = trend

            # Get additional social metrics from external APIs
            try:
                # Get token info for search terms
                token_info = self._get_token_info(token_mint)
                token_name = token_info.get("name", "")
                token_symbol = token_info.get("symbol", "")

                if token_name or token_symbol:
                    # Try to get Reddit mentions
                    search_term = token_symbol if token_symbol else token_name
                    reddit_url = f"https://www.reddit.com/search.json?q={search_term}&sort=new&t=day&limit=100"
                    headers = {"User-Agent": "Mozilla/5.0"}

                    response = requests.get(reddit_url, headers=headers)
                    if response.status_code == 200:
                        reddit_data = response.json()
                        posts = reddit_data.get("data", {}).get("children", [])
                        social_sentiment["reddit_mentions_24h"] = len(posts)

                        # Calculate Reddit sentiment (simplified)
                        if posts:
                            upvotes = sum(post.get("data", {}).get("ups", 0) for post in posts)
                            downvotes = sum(post.get("data", {}).get("downs", 0) for post in posts)
                            total_votes = upvotes + downvotes
                            if total_votes > 0:
                                reddit_sentiment = (upvotes - downvotes) / total_votes
                                social_sentiment["reddit_sentiment_24h"] = reddit_sentiment

            except Exception as e:
                logger.warning(f"Error getting additional social metrics: {e}")

        except Exception as e:
            logger.error(f"Error getting social sentiment for {token_mint}: {e}")

        return social_sentiment

    def _get_developer_activity(self, token_mint: str, token_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get developer activity for a token.

        Args:
            token_mint: The token mint address
            token_info: Token information

        Returns:
            Developer activity data
        """
        developer_activity = {}

        try:
            # Check if we have GitHub repository information
            github_repo = token_info.get("github_repo", "")
            if not github_repo:
                # Try to find GitHub repo from token website
                website = token_info.get("website", "")
                if "github.com" in website:
                    github_repo = website.split("github.com/")[-1].strip("/")

            if github_repo:
                # Get repository information
                headers = {}
                if self.github_token:
                    headers["Authorization"] = f"token {self.github_token}"

                repo_url = f"https://api.github.com/repos/{github_repo}"
                response = requests.get(repo_url, headers=headers)

                if response.status_code == 200:
                    repo_data = response.json()

                    # Extract repository stats
                    developer_activity["github_stars"] = repo_data.get("stargazers_count", 0)
                    developer_activity["github_forks"] = repo_data.get("forks_count", 0)
                    developer_activity["github_open_issues"] = repo_data.get("open_issues_count", 0)

                    # Get commit activity
                    commits_url = f"https://api.github.com/repos/{github_repo}/commits"
                    commits_response = requests.get(commits_url, headers=headers)

                    if commits_response.status_code == 200:
                        commits_data = commits_response.json()

                        # Count recent commits (last 30 days)
                        thirty_days_ago = datetime.now() - timedelta(days=30)
                        recent_commits = 0

                        for commit in commits_data:
                            commit_date_str = commit.get("commit", {}).get("committer", {}).get("date", "")
                            if commit_date_str:
                                try:
                                    commit_date = datetime.fromisoformat(commit_date_str.replace("Z", "+00:00"))
                                    if commit_date >= thirty_days_ago:
                                        recent_commits += 1
                                except Exception:
                                    pass

                        developer_activity["github_commits_30d"] = recent_commits

                        # Calculate development activity score (0-100)
                        stars = developer_activity.get("github_stars", 0)
                        forks = developer_activity.get("github_forks", 0)
                        commits = developer_activity.get("github_commits_30d", 0)

                        # Simple weighted score
                        activity_score = min(100, (stars * 0.3 + forks * 0.2 + commits * 5) / 10)
                        developer_activity["dev_activity_score"] = activity_score

        except Exception as e:
            logger.error(f"Error getting developer activity for {token_mint}: {e}")

        return developer_activity

    def _add_to_history(self, token_mint: str, analytics_data: Dict[str, Any]) -> None:
        """
        Add analytics data to history.

        Args:
            token_mint: The token mint address
            analytics_data: The analytics data to add
        """
        # Create history entry if it doesn't exist
        if token_mint not in self.analytics_history:
            self.analytics_history[token_mint] = []

        # Create history entry
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "price": analytics_data.get("price", 0),
            "market_cap": analytics_data.get("market_cap", 0),
            "volume_24h": analytics_data.get("volume_24h", 0),
            "holder_count": analytics_data.get("holder_count", 0),
            "twitter_mentions_24h": analytics_data.get("twitter_mentions_24h", 0),
            "twitter_sentiment_24h": analytics_data.get("twitter_sentiment_24h", 0),
            "dev_activity_score": analytics_data.get("dev_activity_score", 0)
        }

        # Add to history
        self.analytics_history[token_mint].append(history_entry)

        # Trim history if needed
        if len(self.analytics_history[token_mint]) > self.max_history_entries:
            self.analytics_history[token_mint] = self.analytics_history[token_mint][-self.max_history_entries:]

    def get_token_analytics(self, token_mint: str, force_update: bool = False) -> Dict[str, Any]:
        """
        Get analytics for a token.

        Args:
            token_mint: The token mint address
            force_update: Whether to force an update

        Returns:
            Token analytics data
        """
        if not self.enabled:
            logger.warning("Token analytics is disabled")
            return {}

        # Check if we have analytics data
        if token_mint in self.analytics_data:
            # Check if data is recent enough
            last_updated = self.analytics_data[token_mint].get("last_updated", "")
            if last_updated:
                try:
                    last_updated_dt = datetime.fromisoformat(last_updated)
                    if datetime.now() - last_updated_dt < timedelta(hours=1) and not force_update:
                        # Data is recent, return it
                        return self.analytics_data[token_mint]
                except Exception:
                    pass

        # Update analytics
        return self.update_token_analytics(token_mint)

    def get_analytics_history(self, token_mint: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get analytics history for a token.

        Args:
            token_mint: The token mint address
            days: Number of days of history to return

        Returns:
            List of historical analytics data
        """
        if not self.enabled:
            logger.warning("Token analytics is disabled")
            return []

        if token_mint not in self.analytics_history:
            return []

        # Filter history by date
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_history = []

        for entry in self.analytics_history[token_mint]:
            try:
                entry_date = datetime.fromisoformat(entry["timestamp"])
                if entry_date >= cutoff_date:
                    filtered_history.append(entry)
            except Exception:
                pass

        return filtered_history


    def calculate_risk_score(self, token_mint: str) -> int:
        """
        Calculate a comprehensive risk score for a token.

        Args:
            token_mint: The token mint address

        Returns:
            Risk score (0-100, higher is riskier)
        """
        if not self.enabled:
            logger.warning("Token analytics is disabled")
            return 50  # Default to medium risk

        try:
            # Get analytics data
            analytics = self.get_token_analytics(token_mint)
            if not analytics:
                return 50  # Default to medium risk

            risk_score = 0

            # Market risk factors (0-30 points)

            # Age risk - newer tokens are riskier
            age_days = analytics.get("age_days", 0)
            if age_days < 1:
                risk_score += 15
            elif age_days < 7:
                risk_score += 10
            elif age_days < 30:
                risk_score += 5

            # Market cap risk - lower market cap is riskier
            market_cap = analytics.get("market_cap", 0)
            if market_cap < 100000:  # Less than $100k
                risk_score += 15
            elif market_cap < 1000000:  # Less than $1M
                risk_score += 10
            elif market_cap < 10000000:  # Less than $10M
                risk_score += 5

            # Holder risk factors (0-30 points)

            # Holder count risk - fewer holders is riskier
            holder_count = analytics.get("holder_count", 0)
            if holder_count < 100:
                risk_score += 15
            elif holder_count < 500:
                risk_score += 10
            elif holder_count < 1000:
                risk_score += 5

            # Concentration risk - higher concentration is riskier
            top10_concentration = analytics.get("top10_concentration", 0)
            if top10_concentration > 80:  # Top 10 holders own >80%
                risk_score += 15
            elif top10_concentration > 60:  # Top 10 holders own >60%
                risk_score += 10
            elif top10_concentration > 40:  # Top 10 holders own >40%
                risk_score += 5

            # Social and development risk factors (0-40 points)

            # Social sentiment risk - negative sentiment is riskier
            twitter_sentiment = analytics.get("twitter_sentiment_24h", 0)
            if twitter_sentiment < -0.5:  # Very negative
                risk_score += 20
            elif twitter_sentiment < 0:  # Negative
                risk_score += 10
            elif twitter_sentiment < 0.2:  # Slightly positive
                risk_score += 5

            # Development activity risk - low activity is riskier
            dev_activity_score = analytics.get("dev_activity_score", 0)
            if dev_activity_score < 10:  # Very low activity
                risk_score += 20
            elif dev_activity_score < 30:  # Low activity
                risk_score += 10
            elif dev_activity_score < 50:  # Moderate activity
                risk_score += 5

            # Cap at 100
            return min(risk_score, 100)
        except Exception as e:
            logger.error(f"Error calculating risk score for {token_mint}: {e}")
            return 50  # Default to medium risk

    def get_risk_level(self, token_mint: str) -> str:
        """
        Get risk level for a token.

        Args:
            token_mint: The token mint address

        Returns:
            Risk level (low, medium, high, very_high)
        """
        risk_score = self.calculate_risk_score(token_mint)

        if risk_score < 25:
            return "low"
        elif risk_score < 50:
            return "medium"
        elif risk_score < 75:
            return "high"
        else:
            return "very_high"

    def get_position_size_recommendation(self, token_mint: str, portfolio_value: float) -> Dict[str, Any]:
        """
        Get position size recommendation for a token.

        Args:
            token_mint: The token mint address
            portfolio_value: Total portfolio value in SOL

        Returns:
            Position size recommendation
        """
        try:
            # Get risk level
            risk_level = self.get_risk_level(token_mint)

            # Define position size percentages based on risk level
            position_sizes = {
                "low": 0.10,  # 10% of portfolio
                "medium": 0.05,  # 5% of portfolio
                "high": 0.02,  # 2% of portfolio
                "very_high": 0.01  # 1% of portfolio
            }

            # Get position size percentage
            position_size_pct = position_sizes.get(risk_level, 0.05)

            # Calculate position size in SOL
            position_size_sol = portfolio_value * position_size_pct

            # Calculate max loss
            max_loss_pct = {
                "low": 0.05,  # 5% of position
                "medium": 0.10,  # 10% of position
                "high": 0.15,  # 15% of position
                "very_high": 0.20  # 20% of position
            }

            max_loss_sol = position_size_sol * max_loss_pct.get(risk_level, 0.10)

            return {
                "risk_level": risk_level,
                "position_size_percentage": position_size_pct * 100,
                "position_size_sol": position_size_sol,
                "max_loss_percentage": max_loss_pct.get(risk_level, 0.10) * 100,
                "max_loss_sol": max_loss_sol
            }
        except Exception as e:
            logger.error(f"Error getting position size recommendation for {token_mint}: {e}")
            return {
                "risk_level": "medium",
                "position_size_percentage": 5.0,
                "position_size_sol": portfolio_value * 0.05,
                "max_loss_percentage": 10.0,
                "max_loss_sol": portfolio_value * 0.05 * 0.10,
                "error": str(e)
            }

    def get_exit_strategy_recommendation(self, token_mint: str) -> Dict[str, Any]:
        """
        Get exit strategy recommendation for a token.

        Args:
            token_mint: The token mint address

        Returns:
            Exit strategy recommendation
        """
        try:
            # Get risk level
            risk_level = self.get_risk_level(token_mint)

            # Define exit strategies based on risk level
            exit_strategies = {
                "low": {
                    "strategy_type": "tiered",
                    "take_profit_levels": [
                        {"percentage": 50, "sell_percentage": 25},
                        {"percentage": 100, "sell_percentage": 50},
                        {"percentage": 200, "sell_percentage": 25}
                    ],
                    "stop_loss_percentage": 10,
                    "trailing_stop": {
                        "enabled": True,
                        "trail_percentage": 15,
                        "activation_percentage": 20
                    },
                    "time_based_exit": {
                        "enabled": True,
                        "hold_hours": 72,
                        "min_profit_percentage": 10
                    }
                },
                "medium": {
                    "strategy_type": "tiered",
                    "take_profit_levels": [
                        {"percentage": 30, "sell_percentage": 30},
                        {"percentage": 60, "sell_percentage": 40},
                        {"percentage": 100, "sell_percentage": 30}
                    ],
                    "stop_loss_percentage": 15,
                    "trailing_stop": {
                        "enabled": True,
                        "trail_percentage": 20,
                        "activation_percentage": 15
                    },
                    "time_based_exit": {
                        "enabled": True,
                        "hold_hours": 48,
                        "min_profit_percentage": 5
                    }
                },
                "high": {
                    "strategy_type": "tiered",
                    "take_profit_levels": [
                        {"percentage": 20, "sell_percentage": 40},
                        {"percentage": 40, "sell_percentage": 40},
                        {"percentage": 60, "sell_percentage": 20}
                    ],
                    "stop_loss_percentage": 20,
                    "trailing_stop": {
                        "enabled": True,
                        "trail_percentage": 25,
                        "activation_percentage": 10
                    },
                    "time_based_exit": {
                        "enabled": True,
                        "hold_hours": 24,
                        "min_profit_percentage": 0
                    }
                },
                "very_high": {
                    "strategy_type": "tiered",
                    "take_profit_levels": [
                        {"percentage": 10, "sell_percentage": 50},
                        {"percentage": 20, "sell_percentage": 30},
                        {"percentage": 30, "sell_percentage": 20}
                    ],
                    "stop_loss_percentage": 25,
                    "trailing_stop": {
                        "enabled": True,
                        "trail_percentage": 30,
                        "activation_percentage": 5
                    },
                    "time_based_exit": {
                        "enabled": True,
                        "hold_hours": 12,
                        "min_profit_percentage": 0
                    }
                }
            }

            # Get exit strategy
            exit_strategy = exit_strategies.get(risk_level, exit_strategies["medium"])

            # Add risk level to the result
            exit_strategy["risk_level"] = risk_level

            return exit_strategy
        except Exception as e:
            logger.error(f"Error getting exit strategy recommendation for {token_mint}: {e}")
            return {
                "risk_level": "medium",
                "strategy_type": "tiered",
                "take_profit_levels": [
                    {"percentage": 30, "sell_percentage": 30},
                    {"percentage": 60, "sell_percentage": 40},
                    {"percentage": 100, "sell_percentage": 30}
                ],
                "stop_loss_percentage": 15,
                "trailing_stop": {
                    "enabled": True,
                    "trail_percentage": 20,
                    "activation_percentage": 15
                },
                "time_based_exit": {
                    "enabled": True,
                    "hold_hours": 48,
                    "min_profit_percentage": 5
                },
                "error": str(e)
            }


# Create a singleton instance
token_analytics = TokenAnalytics()
