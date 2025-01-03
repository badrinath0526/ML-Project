import logging
import os
from logging.handlers import RotatingFileHandler

# Create a log directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Log file name
log_file = os.path.join(log_dir, 'app.log')

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3),  # Log rotation every 5MB, keep 3 backups
        logging.StreamHandler()  # Print logs to console
    ]
)

# Create logger
logger = logging.getLogger(__name__)

def log_info(message):
    logger.info(message)

def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)

def log_debug(message):
    logger.debug(message)

def log_critical(message):
    logger.critical(message)
