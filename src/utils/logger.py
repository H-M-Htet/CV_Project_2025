"""
Logging utilities
"""
from loguru import logger
import sys
from pathlib import Path

def setup_logger(log_file: str = None, level: str = "INFO"):
    """
    Setup project logger
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level=level,
            rotation="10 MB",
            retention="1 week",
            compression="zip"
        )
    
    return logger

# Initialize default logger
log = setup_logger(
    log_file="../../results/logs/project.log",
    level="INFO"
)