from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from datetime import datetime
from bson import ObjectId
import httpx
import os

from app.database import connect_db, close_db, get_db
from app.config import INFERENCE_URL, TRAINER_URL
from app.models import (
    GenerateRequest, GenerateResponse,
    FeedbackRequest, FeedbackResponse,
    TrainingDataItem, TrainRequest, TrainResponse
)
from app.auth import (
    get_authorization_url,
    authenticate_with_code,
    get_current_user,
    require_auth,
    User,
    TokenResponse,
    FRONTEND_URL
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()

app = FastAPI(
    title="Blazel API",
    description="LinkedIn Post Feedback Loop API",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Auth Routes ==============

@app.get("/auth/login")
async def login():
    """Redirect to WorkOS login"""
    auth_url = get_authorization_url()
    return RedirectResponse(url=auth_url)

@app.get("/auth/callback")
async def auth_callback(code: str = Query(...)):
    """Handle WorkOS callback and issue JWT"""
    try:
        token_response = await authenticate_with_code(code)
        # Redirect to frontend with token
        redirect_url = f"{FRONTEND_URL}?token={token_response.access_token}"
        return RedirectResponse(url=redirect_url)
    except Exception as e:
        return RedirectResponse(url=f"{FRONTEND_URL}?error={str(e)}")

@app.get("/auth/me")
async def get_me(user: User = Depends(require_auth)):
    """Get current user info"""
    return user

# ============== Public Routes ==============

@app.get("/")
async def root():
    return {"status": "ok", "service": "blazel-api"}

@app.get("/health")
async def health():
    db = get_db()
    try:
        await db.command("ping")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e)}

# ============== Protected Routes ==============

@app.post("/generate", response_model=GenerateResponse)
async def generate_post(request: GenerateRequest, user: User = Depends(require_auth)):
    db = get_db()

    prompt = f"Write a LinkedIn post about: {request.topic}"
    if request.context:
        prompt += f"\nContext: {request.context}"

    # Get style hints from user's previous feedback
    user_rules = await db.feedback.find(
        {"workos_user_id": user.workos_user_id}
    ).sort("created_at", -1).limit(5).to_list(5)

    style_hints = []
    for rule in user_rules:
        if rule.get("comments"):
            style_hints.extend(rule["comments"])

    if style_hints:
        prompt += f"\nStyle guidelines based on past feedback: {'; '.join(style_hints[:3])}"

    # Limit variations to 1-5
    num_variations = max(1, min(5, request.variations))

    # Temperature range: 0.3 (conservative) to 1.0 (creative)
    temperatures = [0.3 + (0.7 * i / max(1, num_variations - 1)) for i in range(num_variations)]
    if num_variations == 1:
        temperatures = [0.7]  # Default temperature for single draft

    drafts_result = []

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for temp in temperatures:
                response = await client.post(
                    f"{INFERENCE_URL}/generate",
                    json={
                        "prompt": prompt,
                        "customer_id": user.workos_user_id,
                        "temperature": temp
                    }
                )
                response.raise_for_status()
                generated_text = response.json().get("text", "")

                draft = {
                    "workos_user_id": user.workos_user_id,
                    "topic": request.topic,
                    "context": request.context,
                    "prompt": prompt,
                    "text": generated_text,
                    "temperature": temp,
                    "created_at": datetime.utcnow()
                }
                result = await db.drafts.insert_one(draft)

                drafts_result.append({
                    "draft_id": str(result.inserted_id),
                    "text": generated_text,
                    "temperature": round(temp, 2)
                })

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Inference service unavailable: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="Inference failed")

    return GenerateResponse(drafts=drafts_result)

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, user: User = Depends(require_auth)):
    db = get_db()

    draft = await db.drafts.find_one({"_id": ObjectId(request.draft_id)})
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    # Verify user owns this draft
    if draft.get("workos_user_id") != user.workos_user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Check if feedback already exists for this draft
    existing_feedback = await db.feedback.find_one({"draft_id": request.draft_id})

    now = datetime.utcnow()

    if existing_feedback:
        # Update existing feedback
        await db.feedback.update_one(
            {"_id": existing_feedback["_id"]},
            {
                "$set": {
                    "edited": request.edited,
                    "comments": request.comments,
                    "rating": request.rating,
                    "updated_at": now
                }
            }
        )
        return FeedbackResponse(
            feedback_id=str(existing_feedback["_id"]),
            message="Feedback updated successfully"
        )
    else:
        # Create new feedback
        feedback = {
            "workos_user_id": user.workos_user_id,
            "draft_id": request.draft_id,
            "prompt": draft.get("prompt", ""),
            "original": request.original,
            "edited": request.edited,
            "comments": request.comments,
            "rating": request.rating,
            "used_in_training": False,
            "created_at": now
        }
        result = await db.feedback.insert_one(feedback)

        return FeedbackResponse(
            feedback_id=str(result.inserted_id),
            message="Feedback recorded successfully"
        )

@app.get("/training-data")
async def get_training_data(
    limit: int = 100,
    user: User = Depends(require_auth)
):
    db = get_db()

    query = {"workos_user_id": user.workos_user_id, "used_in_training": False}

    samples = await db.feedback.find(query).limit(limit).to_list(limit)

    training_data = []
    for sample in samples:
        training_data.append(TrainingDataItem(
            prompt=sample["prompt"],
            chosen=sample["edited"],
            rejected=sample["original"],
            workos_user_id=sample["workos_user_id"]
        ))

    return {"count": len(training_data), "data": training_data}

@app.post("/train", response_model=TrainResponse)
async def trigger_training(request: TrainRequest, user: User = Depends(require_auth)):
    db = get_db()

    count = await db.feedback.count_documents({
        "workos_user_id": user.workos_user_id,
        "used_in_training": False
    })

    if count < request.min_samples:
        return TrainResponse(
            job_id="",
            status="skipped",
            message=f"Not enough samples. Have {count}, need {request.min_samples}"
        )

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{TRAINER_URL}/train",
                json={"customer_id": user.workos_user_id}
            )
            response.raise_for_status()
            result = response.json()

            await db.feedback.update_many(
                {"workos_user_id": user.workos_user_id, "used_in_training": False},
                {"$set": {"used_in_training": True}}
            )

            return TrainResponse(
                job_id=result.get("job_id", "unknown"),
                status="started",
                message=f"Training started with {count} samples"
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Trainer service unavailable: {str(e)}")

@app.get("/drafts")
async def list_drafts(
    limit: int = 50,
    user: User = Depends(require_auth)
):
    """Get all drafts for the current user"""
    db = get_db()

    drafts = await db.drafts.find(
        {"workos_user_id": user.workos_user_id}
    ).sort("created_at", -1).limit(limit).to_list(limit)

    result = []
    for d in drafts:
        # Check if feedback exists for this draft
        feedback = await db.feedback.find_one({"draft_id": str(d["_id"])})
        result.append({
            "id": str(d["_id"]),
            "topic": d.get("topic", ""),
            "text": d.get("text", ""),
            "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
            "has_feedback": feedback is not None,
            "temperature": d.get("temperature")
        })

    return {"drafts": result}

@app.get("/drafts/{draft_id}")
async def get_draft(
    draft_id: str,
    user: User = Depends(require_auth)
):
    """Get a single draft by ID"""
    db = get_db()

    draft = await db.drafts.find_one({"_id": ObjectId(draft_id), "workos_user_id": user.workos_user_id})
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    # Get feedback if exists
    feedback = await db.feedback.find_one({"draft_id": draft_id})

    return {
        "id": str(draft["_id"]),
        "topic": draft.get("topic", ""),
        "context": draft.get("context", ""),
        "text": draft.get("text", ""),
        "created_at": draft.get("created_at").isoformat() if draft.get("created_at") else None,
        "feedback": {
            "edited": feedback.get("edited"),
            "comments": feedback.get("comments", []),
            "rating": feedback.get("rating")
        } if feedback else None
    }

@app.get("/feedback/history")
async def get_feedback_history(
    limit: int = 20,
    user: User = Depends(require_auth)
):
    """Get feedback history for the current user"""
    db = get_db()

    samples = await db.feedback.find(
        {"workos_user_id": user.workos_user_id}
    ).sort("created_at", -1).limit(limit).to_list(limit)

    for s in samples:
        s["_id"] = str(s["_id"])

    return {"feedback": samples}
