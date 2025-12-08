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
    customer_id: Optional[str] = None  # For marketing managers to generate for specific customer

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
    customer_id: str

class TrainRequest(BaseModel):
    min_samples: int = 5

class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str

class Draft(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    customer_id: str  # Identifies the customer
    topic: str
    text: str
    temperature: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Feedback(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    customer_id: str  # Identifies the customer
    draft_id: str
    prompt: str
    original: str
    edited: str
    comments: List[str] = []
    rating: Optional[str] = None  # "like", "dislike", or None
    used_in_training: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DraftEvent(BaseModel):
    """SSE event for streaming draft creation"""
    event: str  # "draft", "error", "done"
    draft_id: Optional[str] = None
    text: Optional[str] = None
    temperature: Optional[float] = None
    error: Optional[str] = None
    index: Optional[int] = None
    total: Optional[int] = None
