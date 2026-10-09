from pathlib import Path
import torch
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Base paths
    project_root: Path = PROJECT_ROOT
    model_path: Path = PROJECT_ROOT / "models" / "best.pt"
    output_dir: Path = PROJECT_ROOT / "data" / "outputs"
    violations_dir: Path = PROJECT_ROOT / "data" / "outputs" / "violations"
    log_dir: Path = PROJECT_ROOT / "logs"

    # Detection
    conf_threshold: float = 0.40
    iou_threshold: float = 0.50

    # PPE association
    iob_association_threshold: float = 0.15

    # Runtime
    device: str = "0" if torch.cuda.is_available() else "cpu"


settings = Settings()