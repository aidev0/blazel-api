import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "local")
INFERENCE_ENV = os.getenv("INFERENCE_ENV", ENV)  # Defaults to ENV if not set
TRAINER_ENV = os.getenv("TRAINER_ENV", ENV)  # Defaults to ENV if not set

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "blazel")

# Inference URL based on INFERENCE_ENV
LOCAL_INFERENCE_URL = os.getenv("LOCAL_INFERENCE_URL", "http://localhost:8001")
PRODUCTION_INFERENCE_URL = os.getenv("PRODUCTION_INFERENCE_URL", "http://35.229.82.124:8001")
INFERENCE_URL = PRODUCTION_INFERENCE_URL if INFERENCE_ENV == "production" else LOCAL_INFERENCE_URL

# Trainer URL based on TRAINER_ENV
LOCAL_TRAINER_URL = os.getenv("LOCAL_TRAINER_URL", "http://localhost:8002")
PRODUCTION_TRAINER_URL = os.getenv("PRODUCTION_TRAINER_URL", "http://localhost:8002")
TRAINER_URL = PRODUCTION_TRAINER_URL if TRAINER_ENV == "production" else LOCAL_TRAINER_URL

# GCP Infrastructure
GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_ZONE = os.getenv("GCP_ZONE", "us-central1-a")
GCP_INFERENCE_INSTANCE = os.getenv("GCP_INFERENCE_INSTANCE", "blazel-inference")
GCP_CREDENTIALS_JSON = os.getenv("GCP_CREDENTIALS_JSON")  # JSON content for Heroku
