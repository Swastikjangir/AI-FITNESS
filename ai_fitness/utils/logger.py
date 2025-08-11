"""
Logging utilities for AI Fitness Coach.

This module provides centralized logging configuration and utilities
for consistent logging across the application.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import os
from datetime import datetime

from ai_fitness.config.settings import get_settings

# Global logger instance
_logger = None

def setup_logging(
    log_level: str = None,
    log_file: str = None,
    log_format: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """Setup application logging configuration"""
    global _logger
    
    settings = get_settings()
    
    # Use settings if not provided
    log_level = log_level or settings.log_level
    log_file = log_file or settings.app_log_file
    log_format = log_format or settings.log_format
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger('ai_fitness')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set global logger
    _logger = logger
    
    # Log initial setup
    logger.info("Logging system initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file}")

def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance"""
    global _logger
    
    if _logger is None:
        setup_logging()
    
    if name:
        return logging.getLogger(f'ai_fitness.{name}')
    
    return _logger

def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed with error: {str(e)}")
                raise
        return wrapper
    return decorator

def log_performance(func_name: str):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"{func_name} completed in {duration:.3f} seconds")
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.error(f"{func_name} failed after {duration:.3f} seconds with error: {str(e)}")
                raise
        return wrapper
    return decorator

class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def log_exception(self, message: str, exc_info: bool = True, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, exc_info=exc_info, **kwargs)

def log_system_info():
    """Log system information for debugging"""
    logger = get_logger()
    
    import platform
    import sys
    
    system_info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'working_directory': os.getcwd(),
        'user': os.getenv('USER', 'unknown'),
        'home': os.getenv('HOME', 'unknown')
    }
    
    logger.info("System Information:")
    for key, value in system_info.items():
        logger.info(f"  {key}: {value}")

def log_memory_usage():
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        logger = get_logger()
        logger.info(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        
    except ImportError:
        # psutil not available
        pass

def create_log_summary(log_file: str = None, max_lines: int = 100) -> Dict[str, Any]:
    """Create a summary of log file contents"""
    if log_file is None:
        settings = get_settings()
        log_file = settings.app_log_file
    
    log_path = Path(log_file)
    if not log_path.exists():
        return {'error': 'Log file not found'}
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Get last N lines
        recent_lines = lines[-max_lines:] if len(lines) > max_lines else lines
        
        # Count log levels
        level_counts = {}
        error_lines = []
        warning_lines = []
        
        for line in recent_lines:
            if 'ERROR' in line:
                level_counts['ERROR'] = level_counts.get('ERROR', 0) + 1
                error_lines.append(line.strip())
            elif 'WARNING' in line:
                level_counts['WARNING'] = level_counts.get('WARNING', 0) + 1
                warning_lines.append(line.strip())
            elif 'INFO' in line:
                level_counts['INFO'] = level_counts.get('INFO', 0) + 1
            elif 'DEBUG' in line:
                level_counts['DEBUG'] = level_counts.get('DEBUG', 0) + 1
        
        summary = {
            'total_lines': len(lines),
            'recent_lines': len(recent_lines),
            'level_counts': level_counts,
            'recent_errors': error_lines[-5:],  # Last 5 errors
            'recent_warnings': warning_lines[-5:],  # Last 5 warnings
            'file_size_mb': log_path.stat().st_size / (1024 * 1024),
            'last_modified': datetime.fromtimestamp(log_path.stat().st_mtime).isoformat()
        }
        
        return summary
        
    except Exception as e:
        return {'error': f'Error reading log file: {str(e)}'}

def cleanup_old_logs(log_dir: str = None, days_to_keep: int = 30):
    """Clean up old log files"""
    if log_dir is None:
        settings = get_settings()
        log_dir = settings.logs_dir
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    deleted_count = 0
    
    for log_file in log_path.glob('*.log*'):
        if log_file.is_file():
            file_age = log_file.stat().st_mtime
            if file_age < cutoff_time:
                try:
                    log_file.unlink()
                    deleted_count += 1
                except Exception:
                    pass
    
    if deleted_count > 0:
        logger = get_logger()
        logger.info(f"Cleaned up {deleted_count} old log files")

if __name__ == "__main__":
    # Test logging setup
    setup_logging(log_level="DEBUG")
    
    logger = get_logger("test")
    
    logger.info("Testing logging system")
    logger.debug("Debug message")
    logger.warning("Warning message")
    
    # Test system info logging
    log_system_info()
    
    # Test log summary
    summary = create_log_summary()
    print("Log summary:", summary)
