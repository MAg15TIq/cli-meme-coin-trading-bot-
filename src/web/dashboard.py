"""
Web Dashboard for Solana Memecoin Trading Bot.
Provides a modern, responsive interface for monitoring and controlling the bot.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO
import pandas as pd
import numpy as np

from src.utils.logging_utils import get_logger
from src.trading.position_manager import position_manager
from src.trading.token_analytics import token_analytics
from src.trading.technical_analysis import technical_analyzer
from src.trading.sentiment_analysis import sentiment_analyzer
from src.utils.performance_tracker import performance_tracker
from src.trading.strategy_engine import strategy_engine
from src.security.security_manager import security_manager
from src.analytics.analytics_engine import analytics_engine
from src.alerts.alert_manager import alert_manager, Alert

logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

@dataclass
class DashboardConfig:
    """Dashboard configuration settings."""
    port: int = 5000
    debug: bool = False
    host: str = '127.0.0.1'
    refresh_interval: int = 5  # seconds
    max_data_points: int = 1000
    enable_websocket: bool = True

class DashboardManager:
    def __init__(self):
        self.config = DashboardConfig()
        self.active_sessions: Dict[str, Dict] = {}
        self.data_cache: Dict[str, Any] = {}
        self.last_update = datetime.now()
        
        # Load templates
        self.template_dir = Path(__file__).parent / 'templates'
        self.static_dir = Path(__file__).parent / 'static'
        
        # Ensure directories exist
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize routes
        self._init_routes()
        
    def _init_routes(self):
        """Initialize Flask routes."""
        
        @app.route('/')
        def index():
            """Render main dashboard page."""
            return render_template('index.html')
            
        @app.route('/api/positions')
        def get_positions():
            """Get current positions data."""
            try:
                positions = position_manager.get_all_positions()
                return jsonify({
                    'success': True,
                    'data': positions
                })
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
                
        @app.route('/api/performance')
        def get_performance():
            """Get performance metrics."""
            try:
                metrics = performance_tracker.get_performance_metrics()
                return jsonify({
                    'success': True,
                    'data': metrics
                })
            except Exception as e:
                logger.error(f"Error getting performance metrics: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
                
        @app.route('/api/portfolio')
        def get_portfolio():
            """Get portfolio analytics."""
            try:
                analytics = analytics_engine.get_portfolio_analytics()
                return jsonify({
                    'success': True,
                    'data': analytics
                })
            except Exception as e:
                logger.error(f"Error getting portfolio analytics: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
                
        @app.route('/api/token/<token_address>')
        def get_token_data(token_address: str):
            """Get token analytics and metrics."""
            try:
                analytics = analytics_engine.get_token_analytics(token_address)
                return jsonify({
                    'success': True,
                    'data': analytics
                })
            except Exception as e:
                logger.error(f"Error getting token data: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
                
        @app.route('/api/reports/<report_type>')
        def get_report(report_type: str):
            """Generate trading report."""
            try:
                report = analytics_engine.generate_report(report_type)
                return jsonify({
                    'success': True,
                    'data': report
                })
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
                
        @app.route('/api/strategies')
        def get_strategies():
            """Get available trading strategies."""
            try:
                strategies = strategy_engine.list_strategies()
                return jsonify({
                    'success': True,
                    'data': strategies
                })
            except Exception as e:
                logger.error(f"Error getting strategies: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
                
        @app.route('/api/alerts', methods=['GET', 'POST'])
        def handle_alerts():
            """Get all alerts or create a new alert."""
            if request.method == 'GET':
                try:
                    alerts = alert_manager.get_all_alerts()
                    return jsonify({
                        'success': True,
                        'data': [vars(alert) for alert in alerts]
                    })
                except Exception as e:
                    logger.error(f"Error getting alerts: {e}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500
            else:
                try:
                    alert_data = request.json
                    alert = Alert(**alert_data)
                    alert_id = alert_manager.create_alert(alert)
                    return jsonify({
                        'success': True,
                        'data': {'id': alert_id}
                    })
                except Exception as e:
                    logger.error(f"Error creating alert: {e}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500
                    
        @app.route('/api/alerts/<alert_id>', methods=['DELETE'])
        def delete_alert(alert_id: str):
            """Delete an alert."""
            try:
                alert_manager.delete_alert(alert_id)
                return jsonify({
                    'success': True,
                    'message': 'Alert deleted successfully'
                })
            except Exception as e:
                logger.error(f"Error deleting alert: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
                
        @app.route('/api/alerts/<alert_id>/toggle', methods=['POST'])
        def toggle_alert(alert_id: str):
            """Toggle alert enabled state."""
            try:
                alert = alert_manager.get_alert(alert_id)
                if not alert:
                    raise ValueError(f"Alert not found: {alert_id}")
                    
                alert_manager.update_alert(alert_id, {'enabled': not alert.enabled})
                return jsonify({
                    'success': True,
                    'message': 'Alert updated successfully'
                })
            except Exception as e:
                logger.error(f"Error toggling alert: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
                
        @app.route('/api/settings', methods=['GET', 'POST'])
        def handle_settings():
            """Get or update bot settings."""
            if request.method == 'GET':
                try:
                    settings = self._get_current_settings()
                    return jsonify({
                        'success': True,
                        'data': settings
                    })
                except Exception as e:
                    logger.error(f"Error getting settings: {e}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500
            else:
                try:
                    settings = request.json
                    self._update_settings(settings)
                    return jsonify({
                        'success': True,
                        'message': 'Settings updated successfully'
                    })
                except Exception as e:
                    logger.error(f"Error updating settings: {e}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500
                    
        @socketio.on('connect')
        def handle_connect():
            """Handle WebSocket connection."""
            session_id = request.sid
            self.active_sessions[session_id] = {
                'connected_at': datetime.now(),
                'last_update': datetime.now()
            }
            logger.info(f"New WebSocket connection: {session_id}")
            
        @socketio.on('disconnect')
        def handle_disconnect():
            """Handle WebSocket disconnection."""
            session_id = request.sid
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")
            
    def _get_current_settings(self) -> Dict[str, Any]:
        """Get current bot settings."""
        return {
            'trading': {
                'enabled': get_config_value('trading_enabled', False),
                'max_position_size': get_config_value('max_position_size', 1000.0),
                'stop_loss_percentage': get_config_value('stop_loss_percentage', 5.0),
                'take_profit_percentage': get_config_value('take_profit_percentage', 10.0)
            },
            'risk': {
                'max_daily_trades': get_config_value('max_daily_trades', 50),
                'max_daily_loss': get_config_value('max_daily_loss', 100.0),
                'position_sizing': get_config_value('position_sizing', 'fixed')
            },
            'monitoring': {
                'price_alerts': get_config_value('price_alerts_enabled', False),
                'wallet_monitoring': get_config_value('wallet_monitoring_enabled', False),
                'sentiment_analysis': get_config_value('sentiment_analysis_enabled', False)
            },
            'notifications': {
                'email': {
                    'enabled': alert_manager.config.email_enabled,
                    'smtp_server': alert_manager.config.smtp_server,
                    'smtp_port': alert_manager.config.smtp_port,
                    'smtp_username': alert_manager.config.smtp_username
                },
                'webhook': {
                    'enabled': alert_manager.config.webhook_enabled,
                    'url': alert_manager.config.webhook_url
                },
                'telegram': {
                    'enabled': alert_manager.config.telegram_enabled,
                    'bot_token': alert_manager.config.telegram_bot_token,
                    'chat_id': alert_manager.config.telegram_chat_id
                }
            }
        }
        
    def _update_settings(self, settings: Dict[str, Any]) -> None:
        """Update bot settings."""
        # Update trading settings
        if 'trading' in settings:
            update_config('trading_enabled', settings['trading'].get('enabled'))
            update_config('max_position_size', settings['trading'].get('max_position_size'))
            update_config('stop_loss_percentage', settings['trading'].get('stop_loss_percentage'))
            update_config('take_profit_percentage', settings['trading'].get('take_profit_percentage'))
            
        # Update risk settings
        if 'risk' in settings:
            update_config('max_daily_trades', settings['risk'].get('max_daily_trades'))
            update_config('max_daily_loss', settings['risk'].get('max_daily_loss'))
            update_config('position_sizing', settings['risk'].get('position_sizing'))
            
        # Update monitoring settings
        if 'monitoring' in settings:
            update_config('price_alerts_enabled', settings['monitoring'].get('price_alerts'))
            update_config('wallet_monitoring_enabled', settings['monitoring'].get('wallet_monitoring'))
            update_config('sentiment_analysis_enabled', settings['monitoring'].get('sentiment_analysis'))
            
        # Update notification settings
        if 'notifications' in settings:
            if 'email' in settings['notifications']:
                alert_manager.config.email_enabled = settings['notifications']['email'].get('enabled', False)
                alert_manager.config.smtp_server = settings['notifications']['email'].get('smtp_server', '')
                alert_manager.config.smtp_port = settings['notifications']['email'].get('smtp_port', 587)
                alert_manager.config.smtp_username = settings['notifications']['email'].get('smtp_username', '')
                if 'smtp_password' in settings['notifications']['email']:
                    alert_manager.config.smtp_password = settings['notifications']['email']['smtp_password']
                    
            if 'webhook' in settings['notifications']:
                alert_manager.config.webhook_enabled = settings['notifications']['webhook'].get('enabled', False)
                alert_manager.config.webhook_url = settings['notifications']['webhook'].get('url', '')
                
            if 'telegram' in settings['notifications']:
                alert_manager.config.telegram_enabled = settings['notifications']['telegram'].get('enabled', False)
                alert_manager.config.telegram_bot_token = settings['notifications']['telegram'].get('bot_token', '')
                alert_manager.config.telegram_chat_id = settings['notifications']['telegram'].get('chat_id', '')
            
    def start(self):
        """Start the dashboard server."""
        try:
            logger.info(f"Starting dashboard server on {self.config.host}:{self.config.port}")
            socketio.run(
                app,
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug
            )
        except Exception as e:
            logger.error(f"Error starting dashboard server: {e}")
            raise
            
    def stop(self):
        """Stop the dashboard server."""
        try:
            logger.info("Stopping dashboard server...")
            # Clean up resources
            self.active_sessions.clear()
            self.data_cache.clear()
        except Exception as e:
            logger.error(f"Error stopping dashboard server: {e}")
            raise

# Global instance
dashboard_manager = DashboardManager() 