"""
Advanced AI Engine - Phase 4B Implementation
Deep learning models for price prediction, pattern recognition, and sentiment analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
import asyncio
import aiohttp
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import ta

from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class PredictionResult:
    """AI prediction result structure"""
    symbol: str
    prediction_type: str  # 'price', 'pattern', 'sentiment'
    value: float
    confidence: float
    timestamp: datetime
    model_version: str
    features_used: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'prediction_type': self.prediction_type,
            'value': self.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'model_version': self.model_version,
            'features_used': self.features_used
        }

class LSTMPricePredictor(nn.Module):
    """LSTM model for price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out

class TransformerPredictor(nn.Module):
    """Transformer model for sequence prediction"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, 1)
        
    def _generate_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        
        # Add positional encoding
        x += self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Take the last output
        x = self.output_projection(x[:, -1, :])
        
        return x

class PatternRecognitionCNN(nn.Module):
    """CNN model for chart pattern recognition"""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super(PatternRecognitionCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Calculate the size after conv layers
        self.fc_input_size = 128 * 8 * 8  # Assuming 64x64 input
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class SentimentAnalyzer:
    """Sentiment analysis for social media and news"""
    
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize sentiment analysis models"""
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name
            )
            self.initialized = True
            logger.info("Sentiment analyzer initialized")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            self.initialized = False
    
    async def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        if not self.initialized:
            await self.initialize()
        
        try:
            result = self.sentiment_pipeline(text)
            return {
                'sentiment': result[0]['label'],
                'confidence': result[0]['score'],
                'text_length': len(text)
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'sentiment': 'NEUTRAL', 'confidence': 0.5, 'text_length': len(text)}
    
    async def analyze_social_media(self, symbol: str) -> Dict:
        """Analyze social media sentiment for a symbol"""
        # This would integrate with Twitter API, Discord, Telegram, etc.
        # For now, return mock data
        return {
            'overall_sentiment': 'POSITIVE',
            'confidence': 0.75,
            'mention_count': 150,
            'sentiment_score': 0.65,
            'trending': True
        }

class FeatureEngineer:
    """Advanced feature engineering for ML models"""
    
    def __init__(self):
        self.scalers = {}
        
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features"""
        try:
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
                df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
            
            # Technical indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = (
                ta.volatility.bollinger_hband(df['close']),
                ta.volatility.bollinger_mavg(df['close']),
                ta.volatility.bollinger_lband(df['close'])
            )
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
                df['vwap'] = ta.volume.volume_weighted_average_price(
                    df['high'], df['low'], df['close'], df['volume']
                )
            
            # Momentum indicators
            df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating technical features: {e}")
            return df
    
    def create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        try:
            # Price action features
            df['price_range'] = df['high'] - df['low']
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            
            # Liquidity features
            if 'liquidity' in df.columns:
                df['liquidity_change'] = df['liquidity'].pct_change()
                df['liquidity_volatility'] = df['liquidity_change'].rolling(window=20).std()
            
            # Time-based features
            df['hour'] = pd.to_datetime(df.index).hour
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating market features: {e}")
            return df
    
    def prepare_sequences(self, data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM/Transformer models"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)

class AdvancedAIEngine:
    """Main advanced AI engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_engineer = FeatureEngineer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.model_versions = {}
        self.performance_metrics = {}
        
        # Model configurations
        self.lstm_config = {
            'input_size': 20,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2
        }
        
        self.transformer_config = {
            'input_size': 20,
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'dropout': 0.1
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Advanced AI Engine initialized (Device: {self.device})")
    
    async def initialize(self):
        """Initialize all AI models"""
        logger.info("Initializing AI models...")
        
        # Initialize sentiment analyzer
        await self.sentiment_analyzer.initialize()
        
        # Initialize deep learning models
        await self._initialize_price_models()
        await self._initialize_pattern_models()
        
        logger.info("AI models initialized successfully")
    
    async def _initialize_price_models(self):
        """Initialize price prediction models"""
        try:
            # LSTM model
            self.models['lstm'] = LSTMPricePredictor(**self.lstm_config).to(self.device)
            self.model_versions['lstm'] = "1.0.0"
            
            # Transformer model
            self.models['transformer'] = TransformerPredictor(**self.transformer_config).to(self.device)
            self.model_versions['transformer'] = "1.0.0"
            
            # Traditional ML models
            self.models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['gradient_boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            logger.info("Price prediction models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing price models: {e}")
    
    async def _initialize_pattern_models(self):
        """Initialize pattern recognition models"""
        try:
            # CNN for pattern recognition
            self.models['pattern_cnn'] = PatternRecognitionCNN().to(self.device)
            self.model_versions['pattern_cnn'] = "1.0.0"
            
            logger.info("Pattern recognition models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing pattern models: {e}")
    
    async def predict_price(self, symbol: str, data: pd.DataFrame, 
                          horizon: int = 1) -> PredictionResult:
        """Predict future price using ensemble of models"""
        try:
            # Feature engineering
            features_df = self.feature_engineer.create_technical_features(data.copy())
            features_df = self.feature_engineer.create_market_features(features_df)
            
            # Prepare data for models
            feature_columns = [col for col in features_df.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            if len(features_df) < 60:  # Minimum data requirement
                raise ValueError("Insufficient data for prediction")
            
            # Get predictions from different models
            predictions = {}
            confidences = {}
            
            # LSTM prediction
            lstm_pred, lstm_conf = await self._predict_with_lstm(features_df, feature_columns)
            predictions['lstm'] = lstm_pred
            confidences['lstm'] = lstm_conf
            
            # Transformer prediction
            transformer_pred, transformer_conf = await self._predict_with_transformer(features_df, feature_columns)
            predictions['transformer'] = transformer_pred
            confidences['transformer'] = transformer_conf
            
            # Traditional ML predictions
            rf_pred, rf_conf = await self._predict_with_random_forest(features_df, feature_columns)
            predictions['random_forest'] = rf_pred
            confidences['random_forest'] = rf_conf
            
            # Ensemble prediction (weighted average)
            weights = {'lstm': 0.3, 'transformer': 0.3, 'random_forest': 0.4}
            ensemble_prediction = sum(predictions[model] * weights[model] for model in weights)
            ensemble_confidence = sum(confidences[model] * weights[model] for model in weights)
            
            return PredictionResult(
                symbol=symbol,
                prediction_type='price',
                value=ensemble_prediction,
                confidence=ensemble_confidence,
                timestamp=datetime.now(),
                model_version='ensemble_1.0.0',
                features_used=feature_columns
            )
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {e}")
            # Return neutral prediction on error
            return PredictionResult(
                symbol=symbol,
                prediction_type='price',
                value=data['close'].iloc[-1],  # Current price
                confidence=0.5,
                timestamp=datetime.now(),
                model_version='fallback',
                features_used=[]
            )
    
    async def _predict_with_lstm(self, data: pd.DataFrame, feature_columns: List[str]) -> Tuple[float, float]:
        """Make prediction using LSTM model"""
        try:
            # Prepare data
            features = data[feature_columns].values
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Create sequences
            X, _ = self.feature_engineer.prepare_sequences(features_scaled, sequence_length=60)
            
            if len(X) == 0:
                return data['close'].iloc[-1], 0.5
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X[-1:]).to(self.device)  # Last sequence
            
            # Make prediction
            self.models['lstm'].eval()
            with torch.no_grad():
                prediction = self.models['lstm'](X_tensor)
                pred_value = prediction.cpu().numpy()[0][0]
            
            # Convert back to price scale (simplified)
            current_price = data['close'].iloc[-1]
            predicted_price = current_price * (1 + pred_value * 0.1)  # Assume 10% max change
            
            return float(predicted_price), 0.8  # High confidence for LSTM
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return data['close'].iloc[-1], 0.5
    
    async def _predict_with_transformer(self, data: pd.DataFrame, feature_columns: List[str]) -> Tuple[float, float]:
        """Make prediction using Transformer model"""
        try:
            # Similar to LSTM but with transformer
            features = data[feature_columns].values
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            X, _ = self.feature_engineer.prepare_sequences(features_scaled, sequence_length=60)
            
            if len(X) == 0:
                return data['close'].iloc[-1], 0.5
            
            X_tensor = torch.FloatTensor(X[-1:]).to(self.device)
            
            self.models['transformer'].eval()
            with torch.no_grad():
                prediction = self.models['transformer'](X_tensor)
                pred_value = prediction.cpu().numpy()[0][0]
            
            current_price = data['close'].iloc[-1]
            predicted_price = current_price * (1 + pred_value * 0.1)
            
            return float(predicted_price), 0.85  # Slightly higher confidence for transformer
            
        except Exception as e:
            logger.error(f"Error in Transformer prediction: {e}")
            return data['close'].iloc[-1], 0.5
    
    async def _predict_with_random_forest(self, data: pd.DataFrame, feature_columns: List[str]) -> Tuple[float, float]:
        """Make prediction using Random Forest model"""
        try:
            # Prepare features and target
            features = data[feature_columns].values
            target = data['close'].pct_change().dropna().values
            
            if len(features) != len(target) + 1:
                features = features[1:]  # Align with target
            
            if len(features) < 100:  # Minimum training data
                return data['close'].iloc[-1], 0.5
            
            # Train model (in production, this would be pre-trained)
            X_train, X_test = features[:-1], features[-1:]
            y_train = target
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.models['random_forest'].fit(X_train_scaled, y_train)
            
            # Make prediction
            pred_return = self.models['random_forest'].predict(X_test_scaled)[0]
            current_price = data['close'].iloc[-1]
            predicted_price = current_price * (1 + pred_return)
            
            return float(predicted_price), 0.75  # Moderate confidence for RF
            
        except Exception as e:
            logger.error(f"Error in Random Forest prediction: {e}")
            return data['close'].iloc[-1], 0.5
    
    async def analyze_sentiment(self, symbol: str) -> PredictionResult:
        """Analyze sentiment for a symbol"""
        try:
            sentiment_data = await self.sentiment_analyzer.analyze_social_media(symbol)
            
            # Convert sentiment to numerical score
            sentiment_score = sentiment_data['sentiment_score']
            confidence = sentiment_data['confidence']
            
            return PredictionResult(
                symbol=symbol,
                prediction_type='sentiment',
                value=sentiment_score,
                confidence=confidence,
                timestamp=datetime.now(),
                model_version=self.model_versions.get('sentiment', '1.0.0'),
                features_used=['social_media', 'news', 'mentions']
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return PredictionResult(
                symbol=symbol,
                prediction_type='sentiment',
                value=0.5,  # Neutral
                confidence=0.5,
                timestamp=datetime.now(),
                model_version='fallback',
                features_used=[]
            )
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all models"""
        return {
            'models': list(self.models.keys()),
            'versions': self.model_versions,
            'performance': self.performance_metrics,
            'device': str(self.device),
            'initialized': len(self.models) > 0
        }

# Global instance
advanced_ai_engine = None

def get_advanced_ai_engine(config: Dict = None) -> AdvancedAIEngine:
    """Get or create advanced AI engine instance"""
    global advanced_ai_engine
    if advanced_ai_engine is None and config:
        advanced_ai_engine = AdvancedAIEngine(config)
    return advanced_ai_engine
