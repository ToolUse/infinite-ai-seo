from pathlib import Path
import sys
from typing import Dict
from loguru import logger

def setup_logger(config: Dict) -> logger:
    """Configure loguru logger with user-friendly formatting and outputs."""
    logging_config = config["logging"]
    
    # Remove default handler
    logger.remove()
    
    # Format for console output - colorful and readable
    console_format = (
        "<white>{time:YYYY-MM-DD HH:mm:ss}</white> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Add console handler with emojis for better visibility
    logger.add(
        sys.stderr,
        colorize=True,
        format=console_format,
        level=logging_config["log_level"],
        filter=lambda record: record["level"].name != "DEBUG"  # Hide debug from console
    )
    
    # Detailed format for file logging
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "process:{process} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # Add file handler
    log_file = Path(logging_config["log_file"])
    logger.add(
        str(log_file),
        rotation=logging_config["rotation_size"],
        retention=f"{logging_config['retention_days']} days",
        compression=logging_config["compression"],
        format=file_format,
        level="DEBUG",  # Always log debug to file
        enqueue=True  # Thread-safe logging
    )
    
    # Add success, error and other level configurations
    logger.level("SUCCESS", color="<green>")
    logger.level("ERROR", color="<red>")
    logger.level("WARNING", color="<yellow>")
    
    # Log startup message
    logger.success("✨ Logger initialized successfully")
    logger.info(f"📁 Log file: {log_file}")
    logger.info(f"🔍 Log level: {logging_config['log_level']}")
    
    return logger

# Export logger for use in other modules
log = logger 