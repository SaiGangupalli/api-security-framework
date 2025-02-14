from pathlib import Path


# Base paths
BASE_DIR = Path("/app")
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = SRC_DIR / "ml" / "models"

# Model file paths
SECURITY_MODEL_PATH = MODELS_DIR / "security_models.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.joblib"
PERFORMANCE_PATH = MODELS_DIR / "model_performance.joblib"