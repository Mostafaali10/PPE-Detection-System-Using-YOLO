from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    # Model
    model_path: str = "models/ppe_detection.pt"

      # Detection
    conf_threshold: float = 0.4
    iou_threshold: float = 0.5

    # PPE association
    iob_association_threshold: float = 0.5

    # Runtime
    device: str = "cuda"

    # Directories
    output_dir: str = "data/outputs"
    log_dir: str = "logs"


settings = Settings()


# أرجعله تاني لما تبني الوديل كامل