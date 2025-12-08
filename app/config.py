import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "local")
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "blazel")

# Inference URL based on environment
LOCAL_INFERENCE_URL = os.getenv("LOCAL_INFERENCE_URL", "http://localhost:8001")
PRODUCTION_INFERENCE_URL = os.getenv("PRODUCTION_INFERENCE_URL", "http://localhost:8001")
INFERENCE_URL = PRODUCTION_INFERENCE_URL if ENV == "production" else LOCAL_INFERENCE_URL

TRAINER_URL = os.getenv("TRAINER_URL", "http://localhost:8002")

# GCP Infrastructure
GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_ZONE = os.getenv("GCP_ZONE", "us-central1-a")
GCP_INFERENCE_INSTANCE = os.getenv("GCP_INFERENCE_INSTANCE", "blazel-inference")
GCP_CREDENTIALS_JSON = os.getenv("GCP_CREDENTIALS_JSON")  # JSON content for Heroku
