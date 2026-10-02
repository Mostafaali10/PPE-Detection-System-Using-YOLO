from src.core.config import settings
from src.core.logger import logger


logger.info("Testing Core Variables")

print(f"model_path: {settings.model_path}")
print("Device:", settings.device)