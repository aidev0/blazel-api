from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from contextlib import asynccontextmanager
from datetime import datetime
from bson import ObjectId
import httpx
import os
import json
import asyncio

from app.database import connect_db, close_db, get_db
from app.config import INFERENCE_URL, TRAINER_URL, GCP_PROJECT, GCP_ZONE, GCP_INFERENCE_INSTANCE, GCP_CREDENTIALS_JSON
from app.models import (
    GenerateRequest, GenerateResponse,
    FeedbackRequest, FeedbackResponse,
    TrainingDataItem, TrainRequest, TrainResponse,
    DraftEvent,
    AdapterTrainRequest, AdapterTrainResponse,
    AdapterRecordRequest, AdapterActivateRequest
)
from app.auth import (
    get_authorization_url,
    authenticate_with_code,
    get_current_user,
    require_auth,
    require_admin,
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

@app.get("/generate/stream")
async def generate_post_stream(
    topic: str,
    context: str = None,
    variations: int = 1,
    customer_id: str = None,  # For marketing managers
    user: User = Depends(require_auth)
):
    """
    Stream draft generation using Server-Sent Events.
    Each draft is saved to DB and streamed to client immediately after generation.
    """
    db = get_db()

    # Determine which customer to generate for
    # Admins can generate for any customer, regular users can only generate for themselves
    target_customer_id = customer_id if (user.is_admin and customer_id) else user.customer_id

    if not target_customer_id:
        raise HTTPException(status_code=400, detail="No customer_id available")

    prompt = f"Write a LinkedIn post about: {topic}"
    if context:
        prompt += f"\nContext: {context}"

    # Get style hints from customer's previous feedback
    feedback_query = {"customer_id": target_customer_id}
    user_rules = await db.feedback.find(feedback_query).sort("created_at", -1).limit(5).to_list(5)

    style_hints = []
    for rule in user_rules:
        if rule.get("comments"):
            style_hints.extend(rule["comments"])

    if style_hints:
        prompt += f"\nStyle guidelines based on past feedback: {'; '.join(style_hints[:3])}"

    # Limit variations to 1-5
    num_variations = max(1, min(5, variations))

    # Temperature range: 0.3 (conservative) to 1.0 (creative)
    temperatures = [0.3 + (0.7 * i / max(1, num_variations - 1)) for i in range(num_variations)]
    if num_variations == 1:
        temperatures = [0.7]

    async def event_generator():
        async with httpx.AsyncClient(timeout=120.0) as client:
            for idx, temp in enumerate(temperatures):
                try:
                    response = await client.post(
                        f"{INFERENCE_URL}/generate",
                        json={
                            "prompt": prompt,
                            "customer_id": target_customer_id,
                            "temperature": temp
                        }
                    )
                    response.raise_for_status()
                    generated_text = response.json().get("text", "")

                    # Save draft to database immediately
                    draft = {
                        "customer_id": target_customer_id,
                        "topic": topic,
                        "context": context,
                        "prompt": prompt,
                        "text": generated_text,
                        "temperature": temp,
                        "created_at": datetime.utcnow()
                    }
                    result = await db.drafts.insert_one(draft)

                    # Send SSE event with the new draft
                    event = DraftEvent(
                        event="draft",
                        draft_id=str(result.inserted_id),
                        text=generated_text,
                        temperature=round(temp, 2),
                        index=idx + 1,
                        total=num_variations
                    )
                    yield f"data: {event.model_dump_json()}\n\n"

                except httpx.RequestError as e:
                    event = DraftEvent(event="error", error=f"Inference service unavailable: {str(e)}")
                    yield f"data: {event.model_dump_json()}\n\n"
                    break
                except httpx.HTTPStatusError as e:
                    event = DraftEvent(event="error", error=f"Inference failed: {e.response.status_code}")
                    yield f"data: {event.model_dump_json()}\n\n"
                    break

        # Send completion event
        event = DraftEvent(event="done", total=num_variations)
        yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_post(request: GenerateRequest, user: User = Depends(require_auth)):
    """
    Non-streaming draft generation (backwards compatible).
    For streaming, use GET /generate/stream
    """
    db = get_db()

    # Determine which customer to generate for
    # Admins can generate for any customer, regular users can only generate for themselves
    target_customer_id = request.customer_id if (user.is_admin and request.customer_id) else user.customer_id

    if not target_customer_id:
        raise HTTPException(status_code=400, detail="No customer_id available")

    prompt = f"Write a LinkedIn post about: {request.topic}"
    if request.context:
        prompt += f"\nContext: {request.context}"

    # Get style hints from customer's previous feedback
    feedback_query = {"customer_id": target_customer_id}
    user_rules = await db.feedback.find(feedback_query).sort("created_at", -1).limit(5).to_list(5)

    style_hints = []
    for rule in user_rules:
        if rule.get("comments"):
            style_hints.extend(rule["comments"])

    if style_hints:
        prompt += f"\nStyle guidelines based on past feedback: {'; '.join(style_hints[:3])}"

    num_variations = max(1, min(5, request.variations))
    temperatures = [0.3 + (0.7 * i / max(1, num_variations - 1)) for i in range(num_variations)]
    if num_variations == 1:
        temperatures = [0.7]

    drafts_result = []

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for temp in temperatures:
                response = await client.post(
                    f"{INFERENCE_URL}/generate",
                    json={
                        "prompt": prompt,
                        "customer_id": target_customer_id,
                        "temperature": temp
                    }
                )
                response.raise_for_status()
                generated_text = response.json().get("text", "")

                draft = {
                    "customer_id": target_customer_id,
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

    # Authorization: regular users can only access their own drafts, admins can access all
    draft_customer_id = draft.get("customer_id")
    user_customer_id = user.customer_id

    if not user.is_admin:
        if draft_customer_id != user_customer_id:
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
            "customer_id": draft_customer_id,
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

    query = {"customer_id": user.customer_id, "used_in_training": False}

    samples = await db.feedback.find(query).limit(limit).to_list(limit)

    training_data = []
    for sample in samples:
        training_data.append(TrainingDataItem(
            prompt=sample["prompt"],
            chosen=sample["edited"],
            rejected=sample["original"],
            customer_id=sample["customer_id"]
        ))

    return {"count": len(training_data), "data": training_data}

@app.post("/train", response_model=TrainResponse)
async def trigger_training(request: TrainRequest, user: User = Depends(require_auth)):
    db = get_db()

    count = await db.feedback.count_documents({
        "customer_id": user.customer_id,
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
                json={"customer_id": user.customer_id}
            )
            response.raise_for_status()
            result = response.json()

            # NOTE: Don't mark feedback as used here - it will be marked in /adapters/record
            # only after training completes successfully

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
    customer_id: str = None,  # For marketing managers to filter by customer
    user: User = Depends(require_auth)
):
    """Get drafts. Admins can view all customers, others see only their own."""
    db = get_db()

    # Build query based on admin status
    if user.is_admin and customer_id:
        # Admin filtering by specific customer
        query = {"customer_id": customer_id}
    elif user.is_admin:
        # Admin sees all drafts
        query = {}
    else:
        # Regular user sees only their drafts
        # Use customer_id if available, otherwise fall back to workos_user_id
        user_customer_id = user.customer_id or user.workos_user_id
        query = {"customer_id": user_customer_id}

    drafts = await db.drafts.find(query).sort("created_at", -1).limit(limit).to_list(limit)

    result = []
    for d in drafts:
        # Check if feedback exists for this draft
        feedback = await db.feedback.find_one({"draft_id": str(d["_id"])})
        result.append({
            "id": str(d["_id"]),
            "customer_id": d.get("customer_id"),
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

    draft = await db.drafts.find_one({"_id": ObjectId(draft_id)})
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    # Authorization check: admins can view all, regular users only their own
    draft_customer_id = draft.get("customer_id")

    if not user.is_admin:
        if draft_customer_id != user.customer_id:
            raise HTTPException(status_code=403, detail="Not authorized")

    # Get feedback if exists
    feedback = await db.feedback.find_one({"draft_id": draft_id})

    return {
        "id": str(draft["_id"]),
        "customer_id": draft.get("customer_id"),
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


@app.delete("/drafts/{draft_id}")
async def delete_draft(
    draft_id: str,
    user: User = Depends(require_auth)
):
    """Delete a draft and its associated feedback"""
    db = get_db()

    draft = await db.drafts.find_one({"_id": ObjectId(draft_id)})
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    # Authorization check: admins can delete all, regular users only their own
    draft_customer_id = draft.get("customer_id")

    if not user.is_admin:
        if draft_customer_id != user.customer_id:
            raise HTTPException(status_code=403, detail="Not authorized")

    # Delete associated feedback first
    feedback_result = await db.feedback.delete_many({"draft_id": draft_id})

    # Delete the draft
    await db.drafts.delete_one({"_id": ObjectId(draft_id)})

    return {
        "status": "deleted",
        "draft_id": draft_id,
        "feedback_deleted": feedback_result.deleted_count
    }


@app.get("/feedback/history")
async def get_feedback_history(
    limit: int = 20,
    user: User = Depends(require_auth)
):
    """Get feedback history for the current user/customer"""
    db = get_db()

    samples = await db.feedback.find(
        {"customer_id": user.customer_id}
    ).sort("created_at", -1).limit(limit).to_list(limit)

    for s in samples:
        s["_id"] = str(s["_id"])

    return {"feedback": samples}

@app.get("/customers")
async def list_customers(user: User = Depends(require_auth)):
    """List all customers. Admins see all, regular users see only themselves."""
    db = get_db()

    if user.is_admin:
        # Get all users as potential customers
        all_users = await db.users.find({"workos_user_id": {"$exists": True}}).to_list(100)

        # Also get customer_ids from drafts (in case there are orphaned drafts)
        draft_customer_ids = set(await db.drafts.distinct("customer_id"))

        customers = []
        seen_ids = set()

        # Add all users (using workos_user_id as customer_id)
        for u in all_users:
            cid = u.get("workos_user_id")
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)

            draft_count = await db.drafts.count_documents({"customer_id": cid})
            customers.append({
                "customer_id": cid,
                "email": u.get("email"),
                "first_name": u.get("first_name"),
                "last_name": u.get("last_name"),
                "draft_count": draft_count
            })

        # Add any orphaned customer_ids from drafts not in users
        for cid in draft_customer_ids:
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            draft_count = await db.drafts.count_documents({"customer_id": cid})
            customers.append({
                "customer_id": cid,
                "email": None,
                "first_name": None,
                "last_name": None,
                "draft_count": draft_count
            })

        # Sort by email or customer_id
        customers.sort(key=lambda c: c.get("email") or c.get("customer_id") or "")

        return {"customers": customers}
    else:
        # Regular user sees only themselves
        # Use workos_user_id as customer_id (consistent with how drafts are created)
        customer_id = user.customer_id or user.workos_user_id
        if not customer_id:
            return {"customers": []}
        draft_count = await db.drafts.count_documents({"customer_id": customer_id})
        return {
            "customers": [{
                "customer_id": customer_id,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "draft_count": draft_count
            }]
        }

# ============== Infrastructure Routes ==============

def get_compute_client():
    """Get GCP Compute client
    - GCP_CREDENTIALS_JSON env var (JSON content) if set
    - Otherwise: GOOGLE_APPLICATION_CREDENTIALS env var (file path) or Workload Identity
    """
    from google.cloud import compute_v1

    if GCP_CREDENTIALS_JSON:
        import json
        from google.oauth2 import service_account
        credentials_info = json.loads(GCP_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        return compute_v1.InstancesClient(credentials=credentials)

    # Falls back to GOOGLE_APPLICATION_CREDENTIALS or Workload Identity
    return compute_v1.InstancesClient()

@app.get("/infra/status")
async def get_inference_status(user: User = Depends(require_auth)):
    """Get the status of the inference VM"""
    if not GCP_PROJECT:
        return {"status": "disabled", "message": "GCP not configured"}

    try:
        client = get_compute_client()
        instance = client.get(project=GCP_PROJECT, zone=GCP_ZONE, instance=GCP_INFERENCE_INSTANCE)
        return {
            "status": instance.status,
            "instance": GCP_INFERENCE_INSTANCE,
            "zone": GCP_ZONE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infra/start")
async def start_inference(user: User = Depends(require_auth)):
    """Start the inference VM"""
    if not GCP_PROJECT:
        raise HTTPException(status_code=400, detail="GCP not configured")

    try:
        client = get_compute_client()
        operation = client.start(project=GCP_PROJECT, zone=GCP_ZONE, instance=GCP_INFERENCE_INSTANCE)
        return {
            "status": "starting",
            "instance": GCP_INFERENCE_INSTANCE,
            "operation": operation.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infra/stop")
async def stop_inference(user: User = Depends(require_auth)):
    """Stop the inference VM"""
    if not GCP_PROJECT:
        raise HTTPException(status_code=400, detail="GCP not configured")

    try:
        client = get_compute_client()
        operation = client.stop(project=GCP_PROJECT, zone=GCP_ZONE, instance=GCP_INFERENCE_INSTANCE)
        return {
            "status": "stopping",
            "instance": GCP_INFERENCE_INSTANCE,
            "operation": operation.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Adapter/Training Routes ==============

@app.get("/adapters/training-data/{customer_id}")
async def get_customer_training_data(
    customer_id: str,
    user: User = Depends(require_auth)
):
    """Get training data (feedback) for a specific customer. Admins only."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    db = get_db()

    # Get unused feedback for this customer
    feedback_list = await db.feedback.find({
        "customer_id": customer_id,
        "used_in_training": False
    }).to_list(1000)

    # Format as training examples
    examples = []
    for f in feedback_list:
        examples.append({
            "id": str(f["_id"]),
            "input": f.get("prompt", ""),
            "output": f.get("edited", ""),
            "original": f.get("original", ""),
            "rating": f.get("rating"),
            "created_at": f.get("created_at").isoformat() if f.get("created_at") else None
        })

    return {
        "customer_id": customer_id,
        "count": len(examples),
        "examples": examples
    }


@app.post("/adapters/train", response_model=AdapterTrainResponse)
async def train_adapter(request: AdapterTrainRequest, user: User = Depends(require_auth)):
    """Trigger LoRA training for a customer. Admins only."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    db = get_db()

    # Get unused feedback for this customer
    feedback_list = await db.feedback.find({
        "customer_id": request.customer_id,
        "used_in_training": False
    }).to_list(1000)

    if len(feedback_list) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 3 feedback samples for training. Have {len(feedback_list)}."
        )

    # Format as training examples
    examples = []
    for f in feedback_list:
        examples.append({
            "input": f.get("prompt", ""),
            "output": f.get("edited", "")
        })

    # Send to trainer service
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{TRAINER_URL}/train",
                json={
                    "customer_id": request.customer_id,
                    "examples": examples,
                    "epochs": request.epochs,
                    "learning_rate": request.learning_rate,
                    "lora_r": request.lora_r,
                    "lora_alpha": request.lora_alpha
                }
            )
            response.raise_for_status()
            result = response.json()

            # NOTE: Don't mark feedback as used here - it will be marked in /adapters/record
            # only after training completes successfully

            return AdapterTrainResponse(
                job_id=result.get("job_id", ""),
                status=result.get("status", "queued"),
                customer_id=request.customer_id,
                feedback_count=len(examples),
                message=f"Training started with {len(examples)} samples"
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Trainer service unavailable: {str(e)}")


@app.post("/adapters/record")
async def record_adapter(request: AdapterRecordRequest):
    """Record a completed adapter in the database. Called by trainer service."""
    db = get_db()

    # Get next version number for this customer
    latest = await db.adapters.find_one(
        {"customer_id": request.customer_id},
        sort=[("version", -1)]
    )
    next_version = (latest["version"] + 1) if latest else 1

    # Deactivate all existing adapters for this customer
    await db.adapters.update_many(
        {"customer_id": request.customer_id},
        {"$set": {"is_active": False}}
    )

    # Create new adapter record (active by default)
    adapter = {
        "customer_id": request.customer_id,
        "version": next_version,
        "gcs_url": request.gcs_url,
        "local_path": request.local_path,
        "is_active": True,
        "job_id": request.job_id,
        "epochs": request.epochs,
        "learning_rate": request.learning_rate,
        "lora_r": request.lora_r,
        "lora_alpha": request.lora_alpha,
        "training_samples": request.training_samples,
        "created_at": datetime.utcnow()
    }
    result = await db.adapters.insert_one(adapter)

    # Mark feedback as used in training NOW (after successful completion)
    # We mark all unused feedback for this customer since that's what was sent
    await db.feedback.update_many(
        {"customer_id": request.customer_id, "used_in_training": False},
        {"$set": {"used_in_training": True}}
    )

    return {
        "adapter_id": str(result.inserted_id),
        "customer_id": request.customer_id,
        "version": next_version,
        "gcs_url": request.gcs_url,
        "message": f"Adapter v{next_version} recorded and activated"
    }


@app.get("/adapters")
async def list_adapters(
    customer_id: str = None,
    user: User = Depends(require_auth)
):
    """List adapters. Admins can see all/filter by customer. Regular users see their own."""
    db = get_db()

    if user.is_admin and customer_id:
        query = {"customer_id": customer_id}
    elif user.is_admin:
        query = {}
    else:
        query = {"customer_id": user.customer_id}

    adapters = await db.adapters.find(query).sort("created_at", -1).to_list(100)

    result = []
    for a in adapters:
        result.append({
            "id": str(a["_id"]),
            "customer_id": a.get("customer_id"),
            "version": a.get("version"),
            "gcs_url": a.get("gcs_url"),
            "is_active": a.get("is_active", False),
            "epochs": a.get("epochs"),
            "training_samples": a.get("training_samples"),
            "created_at": a.get("created_at").isoformat() if a.get("created_at") else None
        })

    return {"adapters": result}


@app.get("/adapters/active/{customer_id}")
async def get_active_adapter(customer_id: str, user: User = Depends(require_auth)):
    """Get the active adapter for a customer."""
    db = get_db()

    # Authorization check
    if not user.is_admin and user.customer_id != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    adapter = await db.adapters.find_one({
        "customer_id": customer_id,
        "is_active": True
    })

    if not adapter:
        return {"adapter": None, "message": "No active adapter for this customer"}

    return {
        "adapter": {
            "id": str(adapter["_id"]),
            "customer_id": adapter.get("customer_id"),
            "version": adapter.get("version"),
            "gcs_url": adapter.get("gcs_url"),
            "is_active": True,
            "epochs": adapter.get("epochs"),
            "training_samples": adapter.get("training_samples"),
            "created_at": adapter.get("created_at").isoformat() if adapter.get("created_at") else None
        }
    }


@app.put("/adapters/{adapter_id}/activate")
async def activate_adapter(adapter_id: str, user: User = Depends(require_auth)):
    """Activate a specific adapter version. Deactivates others for the same customer."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    db = get_db()

    # Get the adapter
    adapter = await db.adapters.find_one({"_id": ObjectId(adapter_id)})
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")

    customer_id = adapter["customer_id"]

    # Deactivate all adapters for this customer
    await db.adapters.update_many(
        {"customer_id": customer_id},
        {"$set": {"is_active": False}}
    )

    # Activate the selected adapter
    await db.adapters.update_one(
        {"_id": ObjectId(adapter_id)},
        {"$set": {"is_active": True}}
    )

    # Notify inference service to reload adapter
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{INFERENCE_URL}/reload-adapter",
                json={
                    "customer_id": customer_id,
                    "adapter_path": adapter.get("gcs_url") or adapter.get("local_path")
                }
            )
    except Exception as e:
        print(f"[WARN] Failed to notify inference service: {e}")

    return {
        "status": "activated",
        "adapter_id": adapter_id,
        "customer_id": customer_id,
        "version": adapter.get("version"),
        "message": f"Adapter v{adapter.get('version')} is now active"
    }


@app.put("/adapters/{adapter_id}/deactivate")
async def deactivate_adapter(adapter_id: str, user: User = Depends(require_auth)):
    """Deactivate an adapter. Inference will use the base model only."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    db = get_db()

    # Get the adapter
    adapter = await db.adapters.find_one({"_id": ObjectId(adapter_id)})
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")

    customer_id = adapter["customer_id"]

    # Deactivate the adapter
    await db.adapters.update_one(
        {"_id": ObjectId(adapter_id)},
        {"$set": {"is_active": False}}
    )

    return {
        "status": "deactivated",
        "adapter_id": adapter_id,
        "customer_id": customer_id,
        "message": "Adapter deactivated. Inference will use the base model."
    }


@app.get("/training-jobs/{job_id}")
async def get_training_job_status(job_id: str, user: User = Depends(require_auth)):
    """Get status of a training job from the trainer service.
    Note: Trainer writes adapter to MongoDB directly after GCS upload.
    """
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{TRAINER_URL}/jobs/{job_id}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=e.response.status_code, detail="Trainer error")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Trainer service unavailable: {str(e)}")
