"""
Logging utilities for the Solana Memecoin Trading Bot.
Provides enhanced logging functionality with different levels and formats.
"""

import os
import logging
import logging.handlers
import traceback
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from config import get_config_value

# Log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


class JsonFormatter(logging.Formatter):
    """Formatter for JSON-formatted logs."""
    
    def format(self, record):
        """Format the log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if available
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        return json.dumps(log_data)


def setup_logging(log_dir: Optional[str] = None, log_level: str = "INFO", 
                  enable_console: bool = True, enable_file: bool = True,
                  max_file_size: int = 10 * 1024 * 1024, backup_count: int = 5,
                  json_format: bool = False) -> None:
    """
    Set up logging for the application.
    
    Args:
        log_dir: Directory for log files (default: ~/.solana-trading-bot/logs)
        log_level: Log level (default: INFO)
        enable_console: Whether to enable console logging (default: True)
        enable_file: Whether to enable file logging (default: True)
        max_file_size: Maximum log file size in bytes (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)
        json_format: Whether to use JSON format for logs (default: False)
    """
    # Get log level
    level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Add console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if enable_file:
        if log_dir is None:
            log_dir = Path.home() / ".solana-trading-bot" / "logs"
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        log_file = os.path.join(log_dir, "trading_bot.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log setup info
    logging.info(f"Logging initialized: level={log_level}, console={enable_console}, file={enable_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, level: str, message: str, context: Dict[str, Any]) -> None:
    """
    Log a message with additional context.
    
    Args:
        logger: Logger instance
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Log message
        context: Additional context as a dictionary
    """
    level_method = getattr(logger, level.lower())
    
    # Create a log record with extra context
    extra = {"extra": context}
    level_method(message, extra=extra)


def log_exception(logger: logging.Logger, message: str, exc: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an exception with additional context.
    
    Args:
        logger: Logger instance
        message: Log message
        exc: Exception to log
        context: Additional context as a dictionary
    """
    if context is None:
        context = {}
    
    # Add exception details to context
    context.update({
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "exception_traceback": traceback.format_exc()
    })
    
    # Log with context
    log_with_context(logger, "ERROR", message, context)


# Initialize logging
def init_logging():
    """Initialize logging based on configuration."""
    log_level = get_config_value("log_level", "INFO")
    log_dir = get_config_value("log_dir", str(Path.home() / ".solana-trading-bot" / "logs"))
    enable_console = get_config_value("enable_console_logging", True)
    enable_file = get_config_value("enable_file_logging", True)
    json_format = get_config_value("json_log_format", False)
    
    setup_logging(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file,
        json_format=json_format
    )
