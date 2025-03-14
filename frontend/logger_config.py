import logging
from logging.handlers import RotatingFileHandler
import os

log_path = os.path.join("logs", "frontend.log")

if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("frontend_logger")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_path,
                              maxBytes=5_000_000,
                              backupCount=5,
                              encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(name)s - '
                              '%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
