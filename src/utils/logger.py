import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Unified colored formatter for consistent logging across the processor module."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    # Custom color mapping for specific messages
    CUSTOM_COLORS = {
        'red': '\033[31m',
        'green': '\033[32m',
        'blue': '\033[34m',
        'yellow': '\033[33m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'reset': '\033[0m'
    }
    
    def __init__(self, fmt: str = None, datefmt: str = None, style: str = '%'):
        if fmt is None:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if datefmt is None:
            datefmt = '%Y-%m-%d %H:%M:%S'
        super().__init__(fmt, datefmt, style)
    
    def format(self, record):
        # Get the original formatted message
        formatted = super().format(record)
        
        # Add color based on log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        return f"{color}{formatted}{reset}"


def setup_logger(
    name: str = None,
    level: str = "INFO",
    format_string: str = None,
    date_format: str = None,
    force: bool = False
) -> logging.Logger:
    """Setup a logger with colored output.
    
    Args:
        name: Logger name (defaults to __name__ if None)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
        date_format: Custom date format string
        force: Force recreation of logger handlers
    
    Returns:
        Configured logger instance
    """
    if name is None:
        name = __name__
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Always remove existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create colored formatter
    formatter = ColoredFormatter(format_string, date_format)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance. If not configured, it will be set up automatically.
    
    Args:
        name: Logger name (defaults to __name__ if None)
    
    Returns:
        Logger instance
    """
    if name is None:
        name = __name__
    
    logger = logging.getLogger(name)
    
    # Always set up the logger to ensure consistent configuration
    logger = setup_logger(name)
    
    return logger


def log_with_color(
    logger: logging.Logger,
    level: str,
    message: str,
    color: Optional[str] = None
) -> None:
    """Log a message with optional custom color.
    
    Args:
        logger: Logger instance
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Message to log
        color: Optional custom color (red, green, blue, yellow, magenta, cyan)
    """
    if color and color in ColoredFormatter.CUSTOM_COLORS:
        colored_message = f"{ColoredFormatter.CUSTOM_COLORS[color]}{message}{ColoredFormatter.CUSTOM_COLORS['reset']}"
        getattr(logger, level.lower())(colored_message)
    else:
        getattr(logger, level.lower())(message)


def log_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    message: str = "Progress",
    level: str = "INFO"
) -> None:
    """Log progress with percentage.
    
    Args:
        logger: Logger instance
        current: Current progress value
        total: Total value
        message: Progress message
        level: Log level
    """
    percentage = (current / total) * 100 if total > 0 else 0
    progress_message = f"{message}: {current}/{total} ({percentage:.1f}%)"
    getattr(logger, level.lower())(progress_message)


def log_timing(
    logger: logging.Logger,
    start_time: datetime,
    operation: str,
    level: str = "INFO"
) -> None:
    """Log timing information for an operation.
    
    Args:
        logger: Logger instance
        start_time: Start time of the operation
        operation: Description of the operation
        level: Log level
    """
    end_time = datetime.now()
    duration = end_time - start_time
    timing_message = f"{operation} completed in {duration.total_seconds():.2f} seconds"
    getattr(logger, level.lower())(timing_message)


# Convenience functions for common logging patterns
def log_file_operation(logger: logging.Logger, operation: str, file_path: str, level: str = "INFO") -> None:
    """Log file operations with consistent formatting."""
    getattr(logger, level.lower())(f"{operation}: {file_path}")


def log_data_stats(logger: logging.Logger, data_name: str, stats: Dict[str, Any], level: str = "INFO") -> None:
    """Log data statistics with consistent formatting."""
    stats_str = ", ".join([f"{k}={v}" for k, v in stats.items()])
    getattr(logger, level.lower())(f"{data_name} statistics: {stats_str}")


def log_config(logger: logging.Logger, config: Dict[str, Any], level: str = "INFO") -> None:
    """Log configuration parameters."""
    config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
    getattr(logger, level.lower())(f"Configuration: {config_str}")


# Initialize default logger
default_logger = get_logger("processor") 