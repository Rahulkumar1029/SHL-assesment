from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings
from pydantic import Field


load_dotenv()


class Settings(BaseSettings):

    # API KEYS
    GOOGLE_API_KEY1: str
    GOOGLE_API_KEY2: str

    # MODELS
    GOOGLE_MODEL_NAME: str = "gemini-2.5-flash"

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # PATHS
    RAW_DATA_PATH: str = "./data/raw_data.json"

    CHROMA_DB_DIR: str = "./chroma_db"

    # RETRIEVAL
    TOP_K_RESULTS: int = 10

    class Config:
        env_file = ".env"
        extra = "ignore"   # allow LANGSMITH_* and other unknown env vars


settings = Settings()