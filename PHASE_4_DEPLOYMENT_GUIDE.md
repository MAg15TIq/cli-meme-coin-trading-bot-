# Phase 4 Deployment Guide: Production-Ready Trading Bot

## 🚀 Overview

This guide covers the deployment of Phase 4 features, transforming your CLI memecoin trading bot into a production-ready, enterprise-grade platform with advanced AI capabilities, cross-chain support, and comprehensive monitoring.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **CPU**: 4+ cores (8+ recommended for AI features)
- **RAM**: 8GB minimum (16GB+ recommended for deep learning)
- **Storage**: 50GB+ available space
- **Network**: Stable internet connection with low latency

### Software Dependencies
- **Python**: 3.11+
- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+
- **Git**: Latest version

### Optional (for advanced features)
- **CUDA**: For GPU-accelerated AI models
- **PostgreSQL**: 15+ (if not using Docker)
- **Redis**: 7+ (if not using Docker)
- **Elasticsearch**: 8.8+ (for log aggregation)

## 🔧 Installation Steps

### 1. Clone and Setup Repository

```bash
# Clone the repository
git clone <repository-url>
cd cli-meme-coin-trading-bot

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration Setup

```bash
# Copy example configuration
cp config.example.py config.py

# Edit configuration for Phase 4
nano config.py
```

**Key Phase 4 Configuration Settings:**

```python
# Phase 4: Live Trading & Real Data
"live_trading_enabled": False,  # Start with False for safety
"paper_trading_mode": True,     # Always start with paper trading
"real_data_feeds_enabled": True,
"websocket_feeds_enabled": True,

# Phase 4: Advanced AI & Machine Learning
"deep_learning_enabled": True,
"sentiment_analysis_enabled": True,
"pattern_recognition_enabled": True,
"ai_confidence_threshold": 0.7,

# Phase 4: Cross-Chain Integration
"cross_chain_enabled": True,
"ethereum_enabled": True,
"bsc_enabled": True,
"polygon_enabled": True,

# Phase 4: Enterprise Features
"api_enabled": True,
"api_port": 8000,
"authentication_required": True,
"rate_limiting_enabled": True,

# Phase 4: Production Monitoring
"metrics_enabled": True,
"health_checks_enabled": True,
"alerting_enabled": True,
"structured_logging_enabled": True,
```

### 3. Environment Variables Setup

Create `.env` file:

```bash
# Security
JWT_SECRET_KEY=your-super-secret-jwt-key-here
ENCRYPTION_KEY=your-32-byte-encryption-key

# Database
DATABASE_URL=postgresql://user:password@localhost/trading_bot
REDIS_HOST=localhost
REDIS_PORT=6379

# API Keys (add your actual keys)
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID
BSC_RPC_URL=https://bsc-dataseed1.binance.org/
POLYGON_RPC_URL=https://polygon-rpc.com/

# Monitoring & Alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Trading Configuration
MAX_DAILY_LOSS_LIMIT=1000.0
EXECUTION_LATENCY_TARGET_MS=100
MIN_ARBITRAGE_PROFIT_PERCENTAGE=1.0
```

## 🐳 Docker Deployment (Recommended)

### 1. Production Deployment with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f trading-bot

# Stop services
docker-compose down
```

### 2. Individual Service Management

```bash
# Start only the trading bot
docker-compose up -d trading-bot

# Start with monitoring stack
docker-compose up -d trading-bot prometheus grafana

# Scale API workers
docker-compose up -d --scale trading-bot=3
```

### 3. Production Configuration

Update `docker-compose.yml` for production:

```yaml
# Production overrides
environment:
  - LIVE_TRADING_ENABLED=true  # Enable after testing
  - PAPER_TRADING_MODE=false   # Disable after thorough testing
  - DEEP_LEARNING_ENABLED=true
  - CROSS_CHAIN_ENABLED=true
  - API_WORKERS=4              # Scale based on load
  - LOGGING_LEVEL=WARNING      # Reduce log verbosity
```

## 🔧 Manual Installation (Alternative)

### 1. Database Setup

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb trading_bot
sudo -u postgres createuser trading
sudo -u postgres psql -c "ALTER USER trading PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading;"
```

### 2. Redis Setup

```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### 3. Application Setup

```bash
# Initialize database
python -c "from src.enterprise.database import init_db; init_db()"

# Start the application
python main.py
```

## 🌐 Service Access

### Web Interfaces

| Service | URL | Default Credentials |
|---------|-----|-------------------|
| **Trading Bot API** | http://localhost:8000 | admin/admin123 |
| **API Documentation** | http://localhost:8000/docs | - |
| **Grafana Dashboards** | http://localhost:3000 | admin/admin123 |
| **Kibana Logs** | http://localhost:5601 | - |
| **Prometheus Metrics** | http://localhost:9091 | - |

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Authentication
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Get portfolio
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/trading/portfolio

# Place order
curl -X POST http://localhost:8000/trading/orders \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SOL", "side": "buy", "amount": 1.0}'
```

## 📊 Monitoring Setup

### 1. Grafana Dashboards

Access Grafana at http://localhost:3000 and import dashboards:

- **Trading Performance**: Order execution, P&L, portfolio metrics
- **System Health**: CPU, memory, disk usage
- **API Metrics**: Request rates, response times, error rates
- **AI Models**: Prediction accuracy, model performance

### 2. Alerting Configuration

Configure alerts in `config.py`:

```python
# Alert thresholds
"alert_recipients": ["admin@yourcompany.com"],
"slack_webhook_url": "https://hooks.slack.com/services/...",

# Alert rules
"high_cpu_threshold": 80,
"high_memory_threshold": 85,
"daily_loss_alert_threshold": 500,
"api_error_rate_threshold": 0.1,
```

### 3. Log Aggregation

Logs are automatically sent to Elasticsearch and viewable in Kibana:

- **Application Logs**: Trading activities, errors, performance
- **System Logs**: Resource usage, health checks
- **API Logs**: Request/response logs, authentication events

## 🔒 Security Configuration

### 1. Authentication Setup

```python
# Strong JWT secret
JWT_SECRET_KEY = "your-256-bit-secret-key"

# User management
USERS = {
    "admin": {
        "password": "hashed_password",
        "permissions": ["read", "write", "admin"]
    },
    "trader": {
        "password": "hashed_password", 
        "permissions": ["read", "write"]
    }
}
```

### 2. Network Security

```bash
# Firewall configuration
sudo ufw allow 22    # SSH
sudo ufw allow 8000  # API
sudo ufw allow 3000  # Grafana
sudo ufw enable
```

### 3. SSL/TLS Setup

```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/private.key -out ssl/certificate.crt

# Update nginx.conf for HTTPS
```

## 🧪 Testing & Validation

### 1. Health Checks

```bash
# System health
curl http://localhost:8000/health

# Service status
curl http://localhost:8000/status

# Metrics endpoint
curl http://localhost:9090/metrics
```

### 2. Paper Trading Validation

```bash
# Start with paper trading
python main.py

# CLI: Select "T" for Live Trading Engine
# Place test orders and verify execution
```

### 3. Load Testing

```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

## 🚀 Production Deployment

### 1. Pre-Production Checklist

- [ ] All tests passing
- [ ] Paper trading validated
- [ ] Security configuration reviewed
- [ ] Monitoring and alerting configured
- [ ] Backup procedures established
- [ ] Disaster recovery plan documented

### 2. Go-Live Process

```bash
# 1. Final configuration update
# Set live_trading_enabled=true in production config

# 2. Deploy with zero downtime
docker-compose up -d --no-deps trading-bot

# 3. Monitor closely
docker-compose logs -f trading-bot

# 4. Verify all systems
curl http://localhost:8000/health
```

### 3. Post-Deployment Monitoring

- Monitor system metrics in Grafana
- Check application logs in Kibana
- Verify trading performance
- Monitor alert channels

## 🔧 Maintenance & Operations

### 1. Regular Maintenance

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Database maintenance
python -c "from src.enterprise.database import maintenance; maintenance()"

# Log rotation
docker-compose exec trading-bot logrotate /etc/logrotate.conf
```

### 2. Backup Procedures

```bash
# Database backup
docker-compose exec postgres pg_dump -U trading trading_bot > backup.sql

# Configuration backup
tar -czf config_backup.tar.gz config.py .env

# Model backup
tar -czf models_backup.tar.gz models/
```

### 3. Scaling

```bash
# Horizontal scaling
docker-compose up -d --scale trading-bot=3

# Load balancer configuration
# Update nginx.conf for multiple instances
```

## 🆘 Troubleshooting

### Common Issues

1. **High Memory Usage**: Reduce AI model complexity or add more RAM
2. **API Timeouts**: Increase worker count or optimize database queries
3. **WebSocket Disconnections**: Check network stability and implement reconnection logic
4. **Database Locks**: Optimize queries and add connection pooling

### Debug Commands

```bash
# Check container logs
docker-compose logs trading-bot

# Access container shell
docker-compose exec trading-bot bash

# Check system resources
docker stats

# Validate configuration
python -c "from config import validate_config; validate_config()"
```

## 📞 Support

For issues and support:

1. Check logs in Kibana dashboard
2. Review Grafana metrics for system health
3. Consult troubleshooting section
4. Create GitHub issue with logs and configuration

## 🎉 Success Metrics

Your Phase 4 deployment is successful when:

- ✅ All health checks passing
- ✅ API responding within SLA (< 100ms)
- ✅ AI predictions generating with >70% confidence
- ✅ Cross-chain arbitrage opportunities detected
- ✅ Monitoring and alerting operational
- ✅ Zero critical security vulnerabilities
- ✅ Paper trading showing positive performance

**Congratulations! Your CLI memecoin trading bot is now a production-ready, enterprise-grade trading platform! 🚀**
