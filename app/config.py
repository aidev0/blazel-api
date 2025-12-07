import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "local")
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "blazel")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8001")
TRAINER_URL = os.getenv("TRAINER_URL", "http://localhost:8002")
