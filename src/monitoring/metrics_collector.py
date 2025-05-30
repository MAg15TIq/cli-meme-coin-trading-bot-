"""
Metrics Collector - Phase 4D Implementation
Production monitoring with Prometheus metrics and health checks
"""

import time
import psutil
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.exposition import MetricsHandler
from http.server import HTTPServer
import threading
import json
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class HealthCheck:
    """Health check result"""
    service: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    response_time_ms: float
    message: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Alert:
    """Alert definition"""
    name: str
    severity: str  # 'critical', 'warning', 'info'
    message: str
    timestamp: datetime
    resolved: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)

class PrometheusMetrics:
    """Prometheus metrics collection"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Trading metrics
        self.orders_total = Counter(
            'trading_orders_total',
            'Total number of orders placed',
            ['status', 'order_type'],
            registry=self.registry
        )
        
        self.order_execution_time = Histogram(
            'trading_order_execution_seconds',
            'Order execution time in seconds',
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        self.daily_pnl = Gauge(
            'trading_daily_pnl_usd',
            'Daily profit and loss in USD',
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
        
        # API metrics
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # AI/ML metrics
        self.ml_predictions_total = Counter(
            'ml_predictions_total',
            'Total ML predictions made',
            ['model', 'prediction_type'],
            registry=self.registry
        )
        
        self.ml_model_accuracy = Gauge(
            'ml_model_accuracy',
            'ML model accuracy score',
            ['model'],
            registry=self.registry
        )
        
        # Cross-chain metrics
        self.arbitrage_opportunities = Gauge(
            'arbitrage_opportunities_count',
            'Number of arbitrage opportunities detected',
            registry=self.registry
        )
        
        self.bridge_transactions = Counter(
            'bridge_transactions_total',
            'Total bridge transactions',
            ['from_chain', 'to_chain', 'status'],
            registry=self.registry
        )
    
    def record_order(self, status: str, order_type: str, execution_time: float):
        """Record order metrics"""
        self.orders_total.labels(status=status, order_type=order_type).inc()
        self.order_execution_time.observe(execution_time)
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value metric"""
        self.portfolio_value.set(value)
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L metric"""
        self.daily_pnl.set(pnl)
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        self.cpu_usage.set(psutil.cpu_percent())
        self.memory_usage.set(psutil.virtual_memory().used)
        self.disk_usage.set(psutil.disk_usage('/').percent)
    
    def record_api_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record API request metrics"""
        self.api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        self.api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_ml_prediction(self, model: str, prediction_type: str):
        """Record ML prediction metrics"""
        self.ml_predictions_total.labels(model=model, prediction_type=prediction_type).inc()
    
    def update_model_accuracy(self, model: str, accuracy: float):
        """Update ML model accuracy"""
        self.ml_model_accuracy.labels(model=model).set(accuracy)
    
    def update_arbitrage_opportunities(self, count: int):
        """Update arbitrage opportunities count"""
        self.arbitrage_opportunities.set(count)
    
    def record_bridge_transaction(self, from_chain: str, to_chain: str, status: str):
        """Record bridge transaction"""
        self.bridge_transactions.labels(from_chain=from_chain, to_chain=to_chain, status=status).inc()
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

class HealthChecker:
    """Health check system"""
    
    def __init__(self):
        self.checks = {}
        self.check_interval = 30  # seconds
        self.running = False
        
    def register_check(self, name: str, check_func, timeout: float = 5.0):
        """Register a health check"""
        self.checks[name] = {
            'func': check_func,
            'timeout': timeout,
            'last_result': None
        }
        logger.info(f"Registered health check: {name}")
    
    async def start(self):
        """Start health checking"""
        self.running = True
        logger.info("Starting health checks...")
        
        while self.running:
            await self._run_checks()
            await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop health checking"""
        self.running = False
        logger.info("Health checks stopped")
    
    async def _run_checks(self):
        """Run all health checks"""
        for name, check_config in self.checks.items():
            try:
                start_time = time.time()
                
                # Run check with timeout
                result = await asyncio.wait_for(
                    check_config['func'](),
                    timeout=check_config['timeout']
                )
                
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                health_check = HealthCheck(
                    service=name,
                    status='healthy' if result else 'unhealthy',
                    response_time_ms=response_time,
                    message='OK' if result else 'Check failed',
                    timestamp=datetime.now()
                )
                
                check_config['last_result'] = health_check
                
            except asyncio.TimeoutError:
                health_check = HealthCheck(
                    service=name,
                    status='unhealthy',
                    response_time_ms=check_config['timeout'] * 1000,
                    message='Timeout',
                    timestamp=datetime.now()
                )
                check_config['last_result'] = health_check
                
            except Exception as e:
                health_check = HealthCheck(
                    service=name,
                    status='unhealthy',
                    response_time_ms=0.0,
                    message=f'Error: {str(e)}',
                    timestamp=datetime.now()
                )
                check_config['last_result'] = health_check
    
    def get_health_status(self) -> Dict[str, HealthCheck]:
        """Get current health status"""
        return {name: check['last_result'] for name, check in self.checks.items() 
                if check['last_result'] is not None}
    
    def is_healthy(self) -> bool:
        """Check if all services are healthy"""
        for check in self.checks.values():
            if check['last_result'] and check['last_result'].status != 'healthy':
                return False
        return True

class AlertManager:
    """Alert management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alerts = []
        self.alert_rules = {}
        self.notification_channels = {}
        
        # Initialize notification channels
        self._setup_notification_channels()
        
        # Default alert rules
        self._setup_default_rules()
    
    def _setup_notification_channels(self):
        """Setup notification channels"""
        # Email notifications
        if self.config.get('email_notifications_enabled'):
            self.notification_channels['email'] = {
                'type': 'email',
                'smtp_server': self.config.get('smtp_server', 'localhost'),
                'smtp_port': self.config.get('smtp_port', 587),
                'username': self.config.get('smtp_username'),
                'password': self.config.get('smtp_password'),
                'recipients': self.config.get('alert_recipients', [])
            }
        
        # Slack notifications (webhook)
        if self.config.get('slack_webhook_url'):
            self.notification_channels['slack'] = {
                'type': 'slack',
                'webhook_url': self.config.get('slack_webhook_url')
            }
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        self.alert_rules = {
            'high_cpu_usage': {
                'condition': lambda metrics: metrics.get('cpu_usage', 0) > 80,
                'severity': 'warning',
                'message': 'High CPU usage detected: {cpu_usage}%'
            },
            'high_memory_usage': {
                'condition': lambda metrics: metrics.get('memory_usage_percent', 0) > 85,
                'severity': 'warning',
                'message': 'High memory usage detected: {memory_usage_percent}%'
            },
            'trading_engine_down': {
                'condition': lambda health: health.get('trading_engine', {}).get('status') != 'healthy',
                'severity': 'critical',
                'message': 'Trading engine is unhealthy'
            },
            'daily_loss_limit': {
                'condition': lambda metrics: metrics.get('daily_pnl', 0) < -1000,
                'severity': 'critical',
                'message': 'Daily loss limit exceeded: ${daily_pnl}'
            },
            'api_high_error_rate': {
                'condition': lambda metrics: metrics.get('api_error_rate', 0) > 0.1,
                'severity': 'warning',
                'message': 'High API error rate: {api_error_rate}%'
            }
        }
    
    async def check_alerts(self, metrics: Dict, health_status: Dict):
        """Check for alert conditions"""
        current_time = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Check condition
                if rule['condition'](metrics) or rule['condition'](health_status):
                    # Create alert
                    alert = Alert(
                        name=rule_name,
                        severity=rule['severity'],
                        message=rule['message'].format(**metrics, **health_status),
                        timestamp=current_time
                    )
                    
                    # Check if this is a new alert (not duplicate)
                    if not self._is_duplicate_alert(alert):
                        self.alerts.append(alert)
                        await self._send_alert(alert)
                        
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _is_duplicate_alert(self, new_alert: Alert) -> bool:
        """Check if alert is duplicate of recent alert"""
        cutoff_time = datetime.now() - timedelta(minutes=15)  # 15 minute window
        
        for alert in self.alerts:
            if (alert.name == new_alert.name and 
                alert.timestamp > cutoff_time and 
                not alert.resolved):
                return True
        
        return False
    
    async def _send_alert(self, alert: Alert):
        """Send alert through notification channels"""
        logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")
        
        # Send email notification
        if 'email' in self.notification_channels:
            await self._send_email_alert(alert)
        
        # Send Slack notification
        if 'slack' in self.notification_channels:
            await self._send_slack_alert(alert)
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            email_config = self.notification_channels['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"[{alert.severity.upper()}] Trading Bot Alert: {alert.name}"
            
            body = f"""
            Alert: {alert.name}
            Severity: {alert.severity}
            Message: {alert.message}
            Timestamp: {alert.timestamp}
            
            Please check the trading bot system.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        try:
            slack_config = self.notification_channels['slack']
            
            color = {
                'critical': '#FF0000',
                'warning': '#FFA500',
                'info': '#00FF00'
            }.get(alert.severity, '#808080')
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Trading Bot Alert: {alert.name}",
                        "fields": [
                            {"title": "Severity", "value": alert.severity, "short": True},
                            {"title": "Message", "value": alert.message, "short": False},
                            {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": True}
                        ]
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(slack_config['webhook_url'], json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent for {alert.name}")
                    else:
                        logger.error(f"Failed to send Slack alert: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_name: str):
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.name == alert_name and not alert.resolved:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_name}")

class MetricsCollector:
    """Main metrics collection system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.prometheus_metrics = PrometheusMetrics()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager(config)
        self.running = False
        
        # Metrics server
        self.metrics_server = None
        self.metrics_port = config.get('metrics_port', 9090)
        
        # Setup health checks
        self._setup_health_checks()
    
    def _setup_health_checks(self):
        """Setup default health checks"""
        # Database health check
        async def check_database():
            # Mock database check
            return True
        
        # Redis health check
        async def check_redis():
            # Mock Redis check
            return True
        
        # Trading engine health check
        async def check_trading_engine():
            # Mock trading engine check
            return True
        
        # API health check
        async def check_api():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('http://localhost:8000/health', timeout=5) as response:
                        return response.status == 200
            except:
                return False
        
        self.health_checker.register_check('database', check_database)
        self.health_checker.register_check('redis', check_redis)
        self.health_checker.register_check('trading_engine', check_trading_engine)
        self.health_checker.register_check('api', check_api)
    
    async def start(self):
        """Start metrics collection"""
        self.running = True
        logger.info("Starting metrics collection...")
        
        # Start metrics server
        self._start_metrics_server()
        
        # Start health checker
        health_task = asyncio.create_task(self.health_checker.start())
        
        # Start monitoring loop
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        await asyncio.gather(health_task, monitoring_task, return_exceptions=True)
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False
        self.health_checker.stop()
        
        if self.metrics_server:
            self.metrics_server.shutdown()
        
        logger.info("Metrics collection stopped")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            class MetricsHTTPHandler(MetricsHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.registry = self.prometheus_metrics.registry
            
            self.metrics_server = HTTPServer(('0.0.0.0', self.metrics_port), MetricsHTTPHandler)
            
            # Start server in separate thread
            server_thread = threading.Thread(target=self.metrics_server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            logger.info(f"Metrics server started on port {self.metrics_port}")
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Update system metrics
                self.prometheus_metrics.update_system_metrics()
                
                # Get current metrics and health status
                metrics = self._collect_current_metrics()
                health_status = self.health_checker.get_health_status()
                
                # Check for alerts
                await self.alert_manager.check_alerts(metrics, health_status)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _collect_current_metrics(self) -> Dict:
        """Collect current system metrics"""
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics_summary(self) -> Dict:
        """Get metrics summary"""
        return {
            'system_metrics': self._collect_current_metrics(),
            'health_status': self.health_checker.get_health_status(),
            'active_alerts': [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
            'is_healthy': self.health_checker.is_healthy()
        }

# Global instance
metrics_collector = None

def get_metrics_collector(config: Dict = None) -> MetricsCollector:
    """Get or create metrics collector instance"""
    global metrics_collector
    if metrics_collector is None and config:
        metrics_collector = MetricsCollector(config)
    return metrics_collector
