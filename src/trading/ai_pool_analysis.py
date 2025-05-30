"""
AI-Powered Pool Analysis for the Solana Memecoin Trading Bot.
Implements machine learning models for pool quality scoring and sustainability prediction.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import get_config_value
from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


@dataclass
class PoolFeatures:
    """Pool feature set for ML analysis."""
    # Liquidity metrics
    total_liquidity: float
    liquidity_depth_1pct: float
    liquidity_depth_5pct: float
    liquidity_concentration: float

    # Volume metrics
    volume_24h: float
    volume_7d: float
    volume_to_liquidity_ratio: float
    trade_count_24h: int

    # Price metrics
    price_volatility_24h: float
    price_change_24h: float
    price_impact_1sol: float

    # Holder metrics
    holder_count: int
    top_10_holder_percentage: float
    holder_distribution_score: float

    # Technical metrics
    age_days: int
    dex_name: str
    fee_tier: float

    # Market metrics
    market_cap: float
    fdv: float
    circulating_supply: float


@dataclass
class PoolAnalysisResult:
    """Pool analysis result with ML predictions."""
    pool_address: str
    quality_score: float
    sustainability_score: float
    risk_score: float
    recommendation: str
    confidence: float
    features: PoolFeatures
    analysis_timestamp: datetime


class PoolQualityAnalyzer:
    """AI-powered analysis of pool quality and sustainability."""

    def __init__(self):
        """Initialize the pool quality analyzer."""
        self.quality_model = None
        self.sustainability_model = None
        self.risk_model = None
        self.feature_scaler = StandardScaler()

        self.feature_extractors = {
            'liquidity_metrics': self._extract_liquidity_features,
            'volume_metrics': self._extract_volume_features,
            'holder_metrics': self._extract_holder_features,
            'technical_metrics': self._extract_technical_features,
            'market_metrics': self._extract_market_features
        }

        # Initialize models
        self._initialize_models()

        logger.info("AI pool quality analyzer initialized")

    def _initialize_models(self):
        """Initialize ML models for pool analysis."""
        try:
            # Quality scoring model (Random Forest Regressor)
            self.quality_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            # Sustainability prediction model (Gradient Boosting Classifier)
            self.sustainability_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )

            # Risk assessment model (Random Forest Regressor)
            self.risk_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )

            # Train models with synthetic data (in production, use real historical data)
            self._train_models_with_synthetic_data()

        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")

    def _train_models_with_synthetic_data(self):
        """Train models with synthetic data (placeholder for real training data)."""
        try:
            # Generate synthetic training data
            n_samples = 1000
            X_synthetic = self._generate_synthetic_features(n_samples)

            # Generate synthetic labels based on feature relationships
            y_quality = self._generate_quality_labels(X_synthetic)
            y_sustainability = self._generate_sustainability_labels(X_synthetic)
            y_risk = self._generate_risk_labels(X_synthetic)

            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X_synthetic)

            # Train models
            self.quality_model.fit(X_scaled, y_quality)
            self.sustainability_model.fit(X_scaled, y_sustainability)
            self.risk_model.fit(X_scaled, y_risk)

            logger.info("ML models trained with synthetic data")

        except Exception as e:
            logger.error(f"Error training models: {e}")

    def analyze_pool(self, pool_data: Dict[str, Any]) -> PoolAnalysisResult:
        """
        Comprehensive pool analysis with ML scoring.

        Args:
            pool_data: Pool data dictionary

        Returns:
            Pool analysis result with ML predictions
        """
        try:
            # Extract features
            features = self._extract_all_features(pool_data)

            # Convert features to array for ML models
            feature_array = self._features_to_array(features)
            feature_array_scaled = self.feature_scaler.transform([feature_array])

            # Get ML predictions
            quality_score = self.quality_model.predict(feature_array_scaled)[0]
            sustainability_prob = self.sustainability_model.predict_proba(feature_array_scaled)[0]
            risk_score = self.risk_model.predict(feature_array_scaled)[0]

            # Calculate sustainability score from probability
            sustainability_score = sustainability_prob[1] if len(sustainability_prob) > 1 else 0.5

            # Calculate confidence based on model uncertainty
            confidence = self._calculate_prediction_confidence(feature_array_scaled)

            # Generate recommendation
            recommendation = self._generate_recommendation(quality_score, sustainability_score, risk_score)

            return PoolAnalysisResult(
                pool_address=pool_data.get('address', ''),
                quality_score=max(0, min(100, quality_score)),
                sustainability_score=max(0, min(1, sustainability_score)),
                risk_score=max(0, min(100, risk_score)),
                recommendation=recommendation,
                confidence=confidence,
                features=features,
                analysis_timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error analyzing pool: {e}")
            return PoolAnalysisResult(
                pool_address=pool_data.get('address', ''),
                quality_score=50.0,
                sustainability_score=0.5,
                risk_score=50.0,
                recommendation="NEUTRAL",
                confidence=0.0,
                features=None,
                analysis_timestamp=datetime.now()
            )

    def _extract_all_features(self, pool_data: Dict[str, Any]) -> PoolFeatures:
        """Extract all features from pool data."""
        try:
            # Extract features using different extractors
            liquidity_features = self._extract_liquidity_features(pool_data)
            volume_features = self._extract_volume_features(pool_data)
            holder_features = self._extract_holder_features(pool_data)
            technical_features = self._extract_technical_features(pool_data)
            market_features = self._extract_market_features(pool_data)

            return PoolFeatures(
                # Liquidity metrics
                total_liquidity=liquidity_features.get('total_liquidity', 0.0),
                liquidity_depth_1pct=liquidity_features.get('liquidity_depth_1pct', 0.0),
                liquidity_depth_5pct=liquidity_features.get('liquidity_depth_5pct', 0.0),
                liquidity_concentration=liquidity_features.get('liquidity_concentration', 0.0),

                # Volume metrics
                volume_24h=volume_features.get('volume_24h', 0.0),
                volume_7d=volume_features.get('volume_7d', 0.0),
                volume_to_liquidity_ratio=volume_features.get('volume_to_liquidity_ratio', 0.0),
                trade_count_24h=volume_features.get('trade_count_24h', 0),

                # Price metrics
                price_volatility_24h=volume_features.get('price_volatility_24h', 0.0),
                price_change_24h=volume_features.get('price_change_24h', 0.0),
                price_impact_1sol=liquidity_features.get('price_impact_1sol', 0.0),

                # Holder metrics
                holder_count=holder_features.get('holder_count', 0),
                top_10_holder_percentage=holder_features.get('top_10_holder_percentage', 0.0),
                holder_distribution_score=holder_features.get('holder_distribution_score', 0.0),

                # Technical metrics
                age_days=technical_features.get('age_days', 0),
                dex_name=technical_features.get('dex_name', 'unknown'),
                fee_tier=technical_features.get('fee_tier', 0.0),

                # Market metrics
                market_cap=market_features.get('market_cap', 0.0),
                fdv=market_features.get('fdv', 0.0),
                circulating_supply=market_features.get('circulating_supply', 0.0)
            )

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return PoolFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'unknown', 0, 0, 0, 0)

    def _extract_liquidity_features(self, pool_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract liquidity-related features."""
        try:
            total_liquidity = pool_data.get('liquidity', 0.0)

            # Calculate liquidity depth at different price levels
            liquidity_depth_1pct = self._calculate_depth(pool_data, 0.01)
            liquidity_depth_5pct = self._calculate_depth(pool_data, 0.05)

            # Calculate liquidity concentration (Herfindahl index)
            liquidity_concentration = self._calculate_concentration(pool_data)

            # Calculate price impact for 1 SOL trade
            price_impact_1sol = self._calculate_price_impact(pool_data, 1.0)

            return {
                'total_liquidity': total_liquidity,
                'liquidity_depth_1pct': liquidity_depth_1pct,
                'liquidity_depth_5pct': liquidity_depth_5pct,
                'liquidity_concentration': liquidity_concentration,
                'price_impact_1sol': price_impact_1sol
            }

        except Exception as e:
            logger.error(f"Error extracting liquidity features: {e}")
            return {'total_liquidity': 0, 'liquidity_depth_1pct': 0, 'liquidity_depth_5pct': 0,
                   'liquidity_concentration': 0, 'price_impact_1sol': 0}

    def _extract_volume_features(self, pool_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract volume-related features."""
        try:
            volume_24h = pool_data.get('volume_24h', 0.0)
            volume_7d = pool_data.get('volume_7d', 0.0)
            trade_count_24h = pool_data.get('trade_count_24h', 0)

            # Calculate volume to liquidity ratio
            liquidity = pool_data.get('liquidity', 1.0)
            volume_to_liquidity_ratio = volume_24h / liquidity if liquidity > 0 else 0

            # Calculate price volatility and change
            price_volatility_24h = pool_data.get('price_volatility_24h', 0.0)
            price_change_24h = pool_data.get('price_change_24h', 0.0)

            return {
                'volume_24h': volume_24h,
                'volume_7d': volume_7d,
                'volume_to_liquidity_ratio': volume_to_liquidity_ratio,
                'trade_count_24h': trade_count_24h,
                'price_volatility_24h': price_volatility_24h,
                'price_change_24h': price_change_24h
            }

        except Exception as e:
            logger.error(f"Error extracting volume features: {e}")
            return {'volume_24h': 0, 'volume_7d': 0, 'volume_to_liquidity_ratio': 0,
                   'trade_count_24h': 0, 'price_volatility_24h': 0, 'price_change_24h': 0}

    def _extract_holder_features(self, pool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract holder-related features."""
        try:
            holder_count = pool_data.get('holder_count', 0)
            top_10_holder_percentage = pool_data.get('top_10_holder_percentage', 0.0)

            # Calculate holder distribution score (lower concentration = higher score)
            holder_distribution_score = max(0, 100 - top_10_holder_percentage)

            return {
                'holder_count': holder_count,
                'top_10_holder_percentage': top_10_holder_percentage,
                'holder_distribution_score': holder_distribution_score
            }

        except Exception as e:
            logger.error(f"Error extracting holder features: {e}")
            return {'holder_count': 0, 'top_10_holder_percentage': 0, 'holder_distribution_score': 0}

    def _extract_technical_features(self, pool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical features."""
        try:
            created_at = pool_data.get('created_at')
            age_days = 0
            if created_at:
                if isinstance(created_at, str):
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    created_date = created_at
                age_days = (datetime.now() - created_date.replace(tzinfo=None)).days

            dex_name = pool_data.get('dex', 'unknown')
            fee_tier = pool_data.get('fee_tier', 0.0)

            return {
                'age_days': age_days,
                'dex_name': dex_name,
                'fee_tier': fee_tier
            }

        except Exception as e:
            logger.error(f"Error extracting technical features: {e}")
            return {'age_days': 0, 'dex_name': 'unknown', 'fee_tier': 0}

    def _extract_market_features(self, pool_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract market-related features."""
        try:
            market_cap = pool_data.get('market_cap', 0.0)
            fdv = pool_data.get('fdv', 0.0)
            circulating_supply = pool_data.get('circulating_supply', 0.0)

            return {
                'market_cap': market_cap,
                'fdv': fdv,
                'circulating_supply': circulating_supply
            }

        except Exception as e:
            logger.error(f"Error extracting market features: {e}")
            return {'market_cap': 0, 'fdv': 0, 'circulating_supply': 0}


    def _calculate_depth(self, pool_data: Dict[str, Any], price_impact: float) -> float:
        """Calculate liquidity depth at a given price impact level."""
        try:
            # Simplified calculation - would need real order book data
            total_liquidity = pool_data.get('liquidity', 0.0)
            return total_liquidity * (1 - price_impact)
        except:
            return 0.0

    def _calculate_concentration(self, pool_data: Dict[str, Any]) -> float:
        """Calculate liquidity concentration using Herfindahl index."""
        try:
            # Simplified - would analyze actual liquidity distribution
            return 0.5  # Default moderate concentration
        except:
            return 0.0

    def _calculate_price_impact(self, pool_data: Dict[str, Any], trade_size: float) -> float:
        """Calculate price impact for a given trade size."""
        try:
            liquidity = pool_data.get('liquidity', 1.0)
            # Simplified price impact calculation
            return min(trade_size / liquidity * 100, 50.0)  # Cap at 50%
        except:
            return 0.0

    def _features_to_array(self, features: PoolFeatures) -> np.array:
        """Convert PoolFeatures to numpy array for ML models."""
        return np.array([
            features.total_liquidity,
            features.liquidity_depth_1pct,
            features.liquidity_depth_5pct,
            features.liquidity_concentration,
            features.volume_24h,
            features.volume_7d,
            features.volume_to_liquidity_ratio,
            features.trade_count_24h,
            features.price_volatility_24h,
            features.price_change_24h,
            features.price_impact_1sol,
            features.holder_count,
            features.top_10_holder_percentage,
            features.holder_distribution_score,
            features.age_days,
            1.0 if features.dex_name == 'raydium' else 0.0,  # One-hot encoding
            features.fee_tier,
            features.market_cap,
            features.fdv,
            features.circulating_supply
        ])

    def _generate_synthetic_features(self, n_samples: int) -> np.array:
        """Generate synthetic feature data for model training."""
        np.random.seed(42)
        return np.random.rand(n_samples, 20) * 100  # 20 features

    def _generate_quality_labels(self, X: np.array) -> np.array:
        """Generate quality score labels based on feature relationships."""
        # Higher liquidity, volume, and holder count = higher quality
        quality = (X[:, 0] * 0.3 +  # liquidity
                  X[:, 4] * 0.2 +   # volume_24h
                  X[:, 11] * 0.2 +  # holder_count
                  X[:, 13] * 0.3)   # holder_distribution_score
        return np.clip(quality, 0, 100)

    def _generate_sustainability_labels(self, X: np.array) -> np.array:
        """Generate sustainability labels (binary classification)."""
        # Sustainable if good liquidity, volume, and age
        sustainability_score = (X[:, 0] * 0.4 +  # liquidity
                               X[:, 6] * 0.3 +   # volume_to_liquidity_ratio
                               X[:, 14] * 0.3)   # age_days
        return (sustainability_score > 50).astype(int)

    def _generate_risk_labels(self, X: np.array) -> np.array:
        """Generate risk score labels."""
        # Higher volatility and concentration = higher risk
        risk = (X[:, 8] * 0.4 +   # price_volatility_24h
                X[:, 3] * 0.3 +   # liquidity_concentration
                X[:, 12] * 0.3)   # top_10_holder_percentage
        return np.clip(risk, 0, 100)

    def _calculate_prediction_confidence(self, feature_array_scaled: np.array) -> float:
        """Calculate prediction confidence based on model uncertainty."""
        try:
            # Use ensemble variance as confidence measure
            quality_preds = []
            risk_preds = []

            # Get predictions from multiple trees (simplified)
            for estimator in self.quality_model.estimators_[:10]:
                quality_preds.append(estimator.predict(feature_array_scaled)[0])

            for estimator in self.risk_model.estimators_[:10]:
                risk_preds.append(estimator.predict(feature_array_scaled)[0])

            # Calculate variance (lower variance = higher confidence)
            quality_var = np.var(quality_preds)
            risk_var = np.var(risk_preds)

            # Convert variance to confidence (0-1 scale)
            avg_var = (quality_var + risk_var) / 2
            confidence = max(0, min(1, 1 - (avg_var / 100)))

            return confidence

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _generate_recommendation(self, quality_score: float, sustainability_score: float,
                               risk_score: float) -> str:
        """Generate trading recommendation based on scores."""
        try:
            # Weighted scoring
            composite_score = (quality_score * 0.4 +
                             sustainability_score * 100 * 0.3 +
                             (100 - risk_score) * 0.3)

            if composite_score >= 75:
                return "STRONG_BUY"
            elif composite_score >= 60:
                return "BUY"
            elif composite_score >= 40:
                return "HOLD"
            elif composite_score >= 25:
                return "SELL"
            else:
                return "STRONG_SELL"

        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "NEUTRAL"


# Create a singleton instance
ai_pool_analyzer = PoolQualityAnalyzer()
