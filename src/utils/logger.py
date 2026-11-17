import logging
import sys
from pathlib import Path
import config

def get_logger(name: str) -> logging.Logger:
    """Configures structured logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)
        
        # File Handler
        log_file = Path(config.LOGS_DIR) / "pipeline.log"
        f_handler = logging.FileHandler(log_file, encoding='utf-8')
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
        
    return logger
