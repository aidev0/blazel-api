import os
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import workos
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.database import get_db

WORKOS_API_KEY = os.getenv("WORKOS_API_KEY")
WORKOS_CLIENT_ID = os.getenv("WORKOS_CLIENT_ID")
WORKOS_REDIRECT_URI = os.getenv("WORKOS_REDIRECT_URI")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

workos_client = workos.WorkOSClient(api_key=WORKOS_API_KEY)

security = HTTPBearer(auto_error=False)

class User(BaseModel):
    id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    workos_user_id: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User

def create_token(user_data: dict) -> str:
    payload = {
        "sub": user_data["id"],
        "email": user_data["email"],
        "first_name": user_data.get("first_name"),
        "last_name": user_data.get("last_name"),
        "workos_user_id": user_data.get("workos_user_id"),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Optional[User]:
    if not credentials:
        return None

    payload = decode_token(credentials.credentials)
    return User(
        id=payload["sub"],
        email=payload["email"],
        first_name=payload.get("first_name"),
        last_name=payload.get("last_name"),
        workos_user_id=payload.get("workos_user_id")
    )

async def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> User:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_token(credentials.credentials)
    return User(
        id=payload["sub"],
        email=payload["email"],
        first_name=payload.get("first_name"),
        last_name=payload.get("last_name"),
        workos_user_id=payload.get("workos_user_id")
    )

def get_authorization_url() -> str:
    return workos_client.user_management.get_authorization_url(
        provider="authkit",
        redirect_uri=WORKOS_REDIRECT_URI,
    )

async def authenticate_with_code(code: str) -> TokenResponse:
    """Authenticate with WorkOS code and store/update user in database"""
    auth_response = workos_client.user_management.authenticate_with_code(
        code=code,
    )

    workos_user = auth_response.user
    db = get_db()

    # Check if user exists
    existing_user = await db.users.find_one({"workos_user_id": workos_user.id})

    now = datetime.utcnow()

    if existing_user:
        # Update existing user
        await db.users.update_one(
            {"_id": existing_user["_id"]},
            {
                "$set": {
                    "email": workos_user.email,
                    "first_name": workos_user.first_name,
                    "last_name": workos_user.last_name,
                    "last_login": now,
                    "updated_at": now
                }
            }
        )
        user_id = str(existing_user["_id"])
    else:
        # Create new user
        new_user = {
            "workos_user_id": workos_user.id,
            "email": workos_user.email,
            "first_name": workos_user.first_name,
            "last_name": workos_user.last_name,
            "created_at": now,
            "updated_at": now,
            "last_login": now
        }
        result = await db.users.insert_one(new_user)
        user_id = str(result.inserted_id)

    user_data = {
        "id": user_id,
        "email": workos_user.email,
        "first_name": workos_user.first_name,
        "last_name": workos_user.last_name,
        "workos_user_id": workos_user.id,
    }

    token = create_token(user_data)

    return TokenResponse(
        access_token=token,
        user=User(**user_data)
    )
