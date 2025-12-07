from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId

class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, info):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)

class GenerateRequest(BaseModel):
    topic: str
    context: Optional[str] = None
    variations: int = 1  # Number of variations to generate (1-5)

class GenerateResponse(BaseModel):
    drafts: List[dict]  # List of {draft_id, text, temperature}

class FeedbackRequest(BaseModel):
    draft_id: str
    original: str
    edited: str
    comments: List[str] = []
    rating: Optional[str] = None  # "like", "dislike", or None

class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str

class TrainingDataItem(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    workos_user_id: str

class TrainRequest(BaseModel):
    min_samples: int = 5

class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str

class Draft(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    workos_user_id: str
    topic: str
    text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Feedback(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    workos_user_id: str
    draft_id: str
    prompt: str
    original: str
    edited: str
    comments: List[str] = []
    used_in_training: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
