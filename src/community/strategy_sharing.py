"""
Community strategy sharing module for the Solana Memecoin Trading Bot.
Allows users to share, import, and rate trading strategies.
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

from config import get_config_value, update_config
from src.trading.strategy_generator import strategy_generator
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class StrategySharing:
    """Manager for community strategy sharing."""
    
    def __init__(self):
        """Initialize the strategy sharing manager."""
        self.enabled = get_config_value("strategy_sharing_enabled", False)
        
        # Path for storing community strategies
        self.data_path = Path(get_config_value("community_strategies_path", 
                                             str(Path.home() / ".solana-trading-bot" / "community_strategies")))
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Community strategies database
        self.strategies_db_path = self.data_path / "strategies_db.json"
        
        # Community strategies: id -> strategy data
        self.community_strategies: Dict[str, Dict[str, Any]] = {}
        
        # Load community strategies
        self._load_strategies()
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable strategy sharing.
        
        Args:
            enabled: Whether strategy sharing should be enabled
        """
        self.enabled = enabled
        update_config("strategy_sharing_enabled", enabled)
        logger.info(f"Strategy sharing {'enabled' if enabled else 'disabled'}")
    
    def _load_strategies(self) -> None:
        """Load community strategies from database."""
        try:
            if self.strategies_db_path.exists():
                with open(self.strategies_db_path, 'r') as f:
                    self.community_strategies = json.load(f)
                logger.info(f"Loaded {len(self.community_strategies)} community strategies")
            else:
                # Create empty database
                self._save_strategies()
        except Exception as e:
            logger.error(f"Error loading community strategies: {e}")
            self.community_strategies = {}
    
    def _save_strategies(self) -> None:
        """Save community strategies to database."""
        try:
            with open(self.strategies_db_path, 'w') as f:
                json.dump(self.community_strategies, f, indent=2)
            logger.debug("Saved community strategies database")
        except Exception as e:
            logger.error(f"Error saving community strategies: {e}")
    
    def share_strategy(self, strategy_id: str, author_name: str, description: str = "") -> Optional[str]:
        """
        Share a strategy with the community.
        
        Args:
            strategy_id: ID of the strategy to share
            author_name: Name of the strategy author
            description: Additional description for the shared strategy
            
        Returns:
            Community strategy ID if successful, None otherwise
        """
        if not self.enabled:
            logger.warning("Strategy sharing is disabled")
            return None
        
        try:
            # Get strategy from strategy generator
            strategies = strategy_generator.list_strategies()
            strategy = None
            
            for s in strategies:
                if s["id"] == strategy_id:
                    strategy = s
                    break
            
            if not strategy:
                logger.warning(f"Strategy not found: {strategy_id}")
                return None
            
            # Generate community strategy ID
            community_id = str(uuid.uuid4())
            
            # Create community strategy
            community_strategy = {
                "id": community_id,
                "original_id": strategy_id,
                "name": strategy["name"],
                "description": description or strategy["description"],
                "author": author_name,
                "risk_level": strategy["risk_level"],
                "created_at": datetime.now().isoformat(),
                "shared_at": datetime.now().isoformat(),
                "performance": strategy["performance"],
                "ratings": [],
                "average_rating": 0.0,
                "downloads": 0,
                "strategy_data": strategy_generator.get_strategy_data(strategy_id)
            }
            
            # Add to community strategies
            self.community_strategies[community_id] = community_strategy
            
            # Save database
            self._save_strategies()
            
            logger.info(f"Shared strategy {strategy['name']} (ID: {community_id})")
            return community_id
        except Exception as e:
            logger.error(f"Error sharing strategy: {e}")
            return None
    
    def import_strategy(self, community_id: str) -> Optional[str]:
        """
        Import a community strategy.
        
        Args:
            community_id: ID of the community strategy to import
            
        Returns:
            Local strategy ID if successful, None otherwise
        """
        if not self.enabled:
            logger.warning("Strategy sharing is disabled")
            return None
        
        try:
            # Check if strategy exists
            if community_id not in self.community_strategies:
                logger.warning(f"Community strategy not found: {community_id}")
                return None
            
            # Get community strategy
            community_strategy = self.community_strategies[community_id]
            
            # Import strategy
            strategy_data = community_strategy["strategy_data"]
            local_id = strategy_generator.import_strategy(strategy_data)
            
            if not local_id:
                logger.warning(f"Failed to import strategy: {community_id}")
                return None
            
            # Update download count
            self.community_strategies[community_id]["downloads"] += 1
            self._save_strategies()
            
            logger.info(f"Imported community strategy {community_strategy['name']} (ID: {community_id})")
            return local_id
        except Exception as e:
            logger.error(f"Error importing strategy: {e}")
            return None
    
    def rate_strategy(self, community_id: str, rating: float, comment: str = "") -> bool:
        """
        Rate a community strategy.
        
        Args:
            community_id: ID of the community strategy to rate
            rating: Rating (1-5)
            comment: Optional comment
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Strategy sharing is disabled")
            return False
        
        try:
            # Check if strategy exists
            if community_id not in self.community_strategies:
                logger.warning(f"Community strategy not found: {community_id}")
                return False
            
            # Validate rating
            rating = max(1.0, min(5.0, rating))
            
            # Add rating
            self.community_strategies[community_id]["ratings"].append({
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update average rating
            ratings = [r["rating"] for r in self.community_strategies[community_id]["ratings"]]
            self.community_strategies[community_id]["average_rating"] = sum(ratings) / len(ratings)
            
            # Save database
            self._save_strategies()
            
            logger.info(f"Rated strategy {community_id}: {rating}/5")
            return True
        except Exception as e:
            logger.error(f"Error rating strategy: {e}")
            return False
    
    def list_community_strategies(self, sort_by: str = "average_rating", 
                                 filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List community strategies.
        
        Args:
            sort_by: Field to sort by (average_rating, downloads, created_at)
            filter_by: Filter criteria (optional)
            
        Returns:
            List of community strategies
        """
        if not self.enabled:
            logger.warning("Strategy sharing is disabled")
            return []
        
        try:
            # Get strategies
            strategies = list(self.community_strategies.values())
            
            # Apply filters
            if filter_by:
                for key, value in filter_by.items():
                    if key == "risk_level":
                        strategies = [s for s in strategies if s["risk_level"] == value]
                    elif key == "min_rating":
                        strategies = [s for s in strategies if s["average_rating"] >= value]
                    elif key == "author":
                        strategies = [s for s in strategies if s["author"] == value]
            
            # Sort strategies
            if sort_by == "average_rating":
                strategies.sort(key=lambda s: s["average_rating"], reverse=True)
            elif sort_by == "downloads":
                strategies.sort(key=lambda s: s["downloads"], reverse=True)
            elif sort_by == "created_at":
                strategies.sort(key=lambda s: s["created_at"], reverse=True)
            
            return strategies
        except Exception as e:
            logger.error(f"Error listing community strategies: {e}")
            return []
    
    def get_community_strategy(self, community_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a community strategy by ID.
        
        Args:
            community_id: ID of the community strategy
            
        Returns:
            Strategy data if found, None otherwise
        """
        if not self.enabled:
            logger.warning("Strategy sharing is disabled")
            return None
        
        if community_id not in self.community_strategies:
            logger.warning(f"Community strategy not found: {community_id}")
            return None
        
        return self.community_strategies[community_id]
    
    def export_strategy_file(self, community_id: str) -> Optional[str]:
        """
        Export a community strategy to a file.
        
        Args:
            community_id: ID of the community strategy
            
        Returns:
            Path to the exported file if successful, None otherwise
        """
        if not self.enabled:
            logger.warning("Strategy sharing is disabled")
            return None
        
        try:
            # Check if strategy exists
            if community_id not in self.community_strategies:
                logger.warning(f"Community strategy not found: {community_id}")
                return None
            
            # Get community strategy
            community_strategy = self.community_strategies[community_id]
            
            # Create export file
            export_path = self.data_path / f"strategy_{community_id}.json"
            
            with open(export_path, 'w') as f:
                json.dump(community_strategy, f, indent=2)
            
            logger.info(f"Exported strategy to {export_path}")
            return str(export_path)
        except Exception as e:
            logger.error(f"Error exporting strategy: {e}")
            return None
    
    def import_strategy_file(self, file_path: str) -> Optional[str]:
        """
        Import a strategy from a file.
        
        Args:
            file_path: Path to the strategy file
            
        Returns:
            Community strategy ID if successful, None otherwise
        """
        if not self.enabled:
            logger.warning("Strategy sharing is disabled")
            return None
        
        try:
            # Load strategy from file
            with open(file_path, 'r') as f:
                strategy_data = json.load(f)
            
            # Validate strategy data
            required_fields = ["name", "description", "author", "risk_level", "strategy_data"]
            for field in required_fields:
                if field not in strategy_data:
                    logger.warning(f"Invalid strategy file: missing {field}")
                    return None
            
            # Generate new community ID
            community_id = str(uuid.uuid4())
            
            # Update strategy data
            strategy_data["id"] = community_id
            strategy_data["imported_at"] = datetime.now().isoformat()
            
            if "ratings" not in strategy_data:
                strategy_data["ratings"] = []
            
            if "average_rating" not in strategy_data:
                strategy_data["average_rating"] = 0.0
            
            if "downloads" not in strategy_data:
                strategy_data["downloads"] = 0
            
            # Add to community strategies
            self.community_strategies[community_id] = strategy_data
            
            # Save database
            self._save_strategies()
            
            logger.info(f"Imported strategy from file: {strategy_data['name']} (ID: {community_id})")
            return community_id
        except Exception as e:
            logger.error(f"Error importing strategy from file: {e}")
            return None


# Create singleton instance
strategy_sharing = StrategySharing()
