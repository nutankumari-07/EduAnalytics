"""Application configuration for EduAnalytics."""
import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _secret_key():
    """Use SECRET_KEY from the environment; otherwise persist a random one locally."""
    env = os.environ.get("SECRET_KEY")
    if env:
        return env
    key_file = os.path.join(BASE_DIR, "database", ".secret_key")
    if os.path.exists(key_file):
        with open(key_file) as fh:
            return fh.read().strip()
    os.makedirs(os.path.dirname(key_file), exist_ok=True)
    key = secrets.token_hex(32)
    with open(key_file, "w") as fh:
        fh.write(key)
    return key


class Config:
    SECRET_KEY = _secret_key()
    DATABASE = os.path.join(BASE_DIR, "database", "eduanalytics.db")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "performance_model.pkl")
    METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
    DATASET_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    MAX_CONTENT_LENGTH = 4 * 1024 * 1024          # 4 MB upload limit
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 8      # 8 hours
    DEFAULT_COURSE_IMAGE = "images/courses/default.jpg"
    PER_PAGE = 20
    TESTING = False
