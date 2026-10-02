import sys

from loguru import logger


logger.remove()

logger.add(
    sys.stdout,
    level="INFO"
)

logger.add(
    "logs/app.log",
    level="INFO",
    rotation="10 MB",
    retention="7 days"
)