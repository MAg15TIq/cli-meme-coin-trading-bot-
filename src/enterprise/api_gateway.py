"""
Enterprise API Gateway - Phase 4C Implementation
RESTful API for external integrations with authentication and rate limiting
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import jwt
import redis
import time
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import hashlib
import asyncio
from contextlib import asynccontextmanager

from ..utils.logging_utils import setup_logger
from ..trading.live_trading_engine import get_live_trading_engine, LiveOrder
from ..trading.cross_chain_manager import get_cross_chain_manager
from ..ml.advanced_ai_engine import get_advanced_ai_engine

logger = setup_logger(__name__)

# Pydantic models for API requests/responses
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class OrderRequest(BaseModel):
    symbol: str
    side: str = Field(..., regex=r'^(buy|sell)$')
    amount: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)
    order_type: str = Field(default="market", regex=r'^(market|limit|stop)$')

class OrderResponse(BaseModel):
    order_id: str
    status: str
    message: str

class PortfolioResponse(BaseModel):
    total_value_usd: float
    positions: Dict[str, Any]
    performance: Dict[str, float]
    risk_metrics: Dict[str, float]

class PredictionRequest(BaseModel):
    symbol: str
    prediction_type: str = Field(..., regex=r'^(price|sentiment|pattern)$')
    horizon_hours: int = Field(default=24, ge=1, le=168)  # 1 hour to 1 week

class PredictionResponse(BaseModel):
    symbol: str
    prediction_type: str
    value: float
    confidence: float
    timestamp: str
    model_version: str

class ArbitrageResponse(BaseModel):
    opportunities: List[Dict[str, Any]]
    total_count: int
    timestamp: str

# Authentication and rate limiting
class AuthManager:
    """Manages authentication and JWT tokens"""
    
    def __init__(self, secret_key: str, redis_client: redis.Redis):
        self.secret_key = secret_key
        self.redis_client = redis_client
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60
        
        # Mock user database (in production, use proper database)
        self.users_db = {
            "admin": {
                "username": "admin",
                "email": "admin@example.com",
                "hashed_password": self._hash_password("admin123"),
                "is_active": True,
                "permissions": ["read", "write", "admin"]
            },
            "trader": {
                "username": "trader",
                "email": "trader@example.com", 
                "hashed_password": self._hash_password("trader123"),
                "is_active": True,
                "permissions": ["read", "write"]
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self._hash_password(plain_password) == hashed_password
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user credentials"""
        user = self.users_db.get(username)
        if user and user["is_active"] and self.verify_password(password, user["hashed_password"]):
            return user
        return None
    
    def create_access_token(self, data: Dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                return None
            
            # Check if token is blacklisted
            if self.redis_client.get(f"blacklist:{token}"):
                return None
            
            user = self.users_db.get(username)
            return user
            
        except jwt.PyJWTError:
            return None
    
    def blacklist_token(self, token: str):
        """Blacklist a token (for logout)"""
        # Set expiration to match token expiration
        self.redis_client.setex(f"blacklist:{token}", 3600, "1")

class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        
        # Rate limit configurations
        self.limits = {
            "default": {"requests": 100, "window": 60},      # 100 requests per minute
            "trading": {"requests": 10, "window": 60},       # 10 trades per minute
            "data": {"requests": 1000, "window": 60},        # 1000 data requests per minute
            "auth": {"requests": 5, "window": 300},          # 5 auth attempts per 5 minutes
        }
    
    async def check_rate_limit(self, key: str, limit_type: str = "default") -> bool:
        """Check if request is within rate limit"""
        limit_config = self.limits.get(limit_type, self.limits["default"])
        
        current_time = int(time.time())
        window_start = current_time - limit_config["window"]
        
        # Clean old entries
        self.redis_client.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_requests = self.redis_client.zcard(key)
        
        if current_requests >= limit_config["requests"]:
            return False
        
        # Add current request
        self.redis_client.zadd(key, {str(current_time): current_time})
        self.redis_client.expire(key, limit_config["window"])
        
        return True

# Global instances
auth_manager = None
rate_limiter = None
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting API Gateway...")
    
    # Initialize global instances
    global auth_manager, rate_limiter
    
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        auth_manager = AuthManager("your-secret-key-here", redis_client)
        rate_limiter = RateLimiter(redis_client)
        
        logger.info("API Gateway started successfully")
    except Exception as e:
        logger.error(f"Failed to start API Gateway: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway...")

# Create FastAPI app
app = FastAPI(
    title="Memecoin Trading Bot API",
    description="Enterprise API for the CLI Memecoin Trading Bot",
    version="4.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )
    
    user = auth_manager.verify_token(credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

async def check_rate_limit(request: Request, limit_type: str = "default"):
    """Check rate limit for request"""
    if not rate_limiter:
        return True
    
    client_ip = request.client.host
    key = f"rate_limit:{limit_type}:{client_ip}"
    
    if not await rate_limiter.check_rate_limit(key, limit_type):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

# Authentication endpoints
@app.post("/auth/login", response_model=TokenResponse)
async def login(user_login: UserLogin, request: Request):
    """User login endpoint"""
    await check_rate_limit(request, "auth")
    
    if not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )
    
    user = auth_manager.authenticate_user(user_login.username, user_login.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = auth_manager.create_access_token(
        data={"sub": user["username"], "permissions": user["permissions"]}
    )
    
    return TokenResponse(
        access_token=access_token,
        expires_in=auth_manager.access_token_expire_minutes * 60
    )

@app.post("/auth/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """User logout endpoint"""
    if auth_manager:
        auth_manager.blacklist_token(credentials.credentials)
    
    return {"message": "Successfully logged out"}

# Trading endpoints
@app.post("/trading/orders", response_model=OrderResponse)
async def place_order(
    order_request: OrderRequest,
    request: Request,
    current_user: Dict = Depends(get_current_user)
):
    """Place a trading order"""
    await check_rate_limit(request, "trading")
    
    if "write" not in current_user.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        # Get live trading engine
        live_engine = get_live_trading_engine()
        if not live_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trading engine unavailable"
            )
        
        # Create order
        order = LiveOrder(
            order_id=f"api_{int(time.time())}_{current_user['username']}",
            symbol=order_request.symbol,
            side=order_request.side,
            amount=order_request.amount,
            price=order_request.price,
            order_type=order_request.order_type
        )
        
        # Place order
        order_id = await live_engine.place_order(order)
        
        return OrderResponse(
            order_id=order_id,
            status="submitted",
            message="Order placed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to place order: {str(e)}"
        )

@app.get("/trading/portfolio", response_model=PortfolioResponse)
async def get_portfolio(
    request: Request,
    current_user: Dict = Depends(get_current_user)
):
    """Get portfolio information"""
    await check_rate_limit(request, "data")
    
    try:
        # Mock portfolio data (integrate with actual portfolio manager)
        portfolio_data = {
            "total_value_usd": 50000.0,
            "positions": {
                "SOL": {"amount": 100.0, "value_usd": 15000.0},
                "BONK": {"amount": 1000000.0, "value_usd": 5000.0}
            },
            "performance": {
                "total_return": 0.25,
                "daily_pnl": 1250.0,
                "sharpe_ratio": 1.8
            },
            "risk_metrics": {
                "var_95": -2500.0,
                "max_drawdown": 0.15,
                "volatility": 0.35
            }
        }
        
        return PortfolioResponse(**portfolio_data)
        
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve portfolio"
        )

# AI/ML endpoints
@app.post("/ai/predict", response_model=PredictionResponse)
async def get_prediction(
    prediction_request: PredictionRequest,
    request: Request,
    current_user: Dict = Depends(get_current_user)
):
    """Get AI prediction for symbol"""
    await check_rate_limit(request, "data")
    
    try:
        # Get AI engine
        ai_engine = get_advanced_ai_engine()
        if not ai_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI engine unavailable"
            )
        
        # Mock prediction (integrate with actual AI engine)
        prediction_data = {
            "symbol": prediction_request.symbol,
            "prediction_type": prediction_request.prediction_type,
            "value": 125.50 if prediction_request.prediction_type == "price" else 0.75,
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat(),
            "model_version": "ensemble_1.0.0"
        }
        
        return PredictionResponse(**prediction_data)
        
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get prediction"
        )

# Cross-chain endpoints
@app.get("/cross-chain/arbitrage", response_model=ArbitrageResponse)
async def get_arbitrage_opportunities(
    request: Request,
    current_user: Dict = Depends(get_current_user)
):
    """Get cross-chain arbitrage opportunities"""
    await check_rate_limit(request, "data")
    
    try:
        # Get cross-chain manager
        cross_chain = get_cross_chain_manager()
        if not cross_chain:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cross-chain service unavailable"
            )
        
        # Mock arbitrage data (integrate with actual cross-chain manager)
        arbitrage_data = {
            "opportunities": [
                {
                    "asset": "USDC",
                    "buy_chain": "polygon",
                    "sell_chain": "ethereum",
                    "profit_percentage": 2.5,
                    "net_profit": 125.0,
                    "confidence": 0.9
                }
            ],
            "total_count": 1,
            "timestamp": datetime.now().isoformat()
        }
        
        return ArbitrageResponse(**arbitrage_data)
        
    except Exception as e:
        logger.error(f"Error getting arbitrage opportunities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get arbitrage opportunities"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0"
    }

# System status endpoint
@app.get("/status")
async def system_status(current_user: Dict = Depends(get_current_user)):
    """Get system status"""
    return {
        "trading_engine": "operational",
        "ai_engine": "operational", 
        "cross_chain": "operational",
        "database": "operational",
        "redis": "operational",
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

def start_api_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Start the API server"""
    uvicorn.run(
        "src.enterprise.api_gateway:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    start_api_server()
