from src.core.config import settings
from src.core.logger import logger


def test_core_settings():
    logger.info("Testing Core Variables")
    assert settings.model_path is not None
    assert settings.conf_threshold > 0
    assert settings.iob_association_threshold > 0
    assert settings.device in ["0", "cuda:0", "cpu"]