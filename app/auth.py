import os
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Security, Depends
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
WORKOS_ORGANIZATION_ID = os.getenv("WORKOS_ORGANIZATION_ID")  # Optional: default org for login
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
    customer_id: Optional[str] = None  # WorkOS organization ID
    is_admin: bool = False  # From WorkOS organization membership role

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
        "customer_id": user_data.get("customer_id"),
        "is_admin": user_data.get("is_admin", False),
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
        workos_user_id=payload.get("workos_user_id"),
        customer_id=payload.get("customer_id"),
        is_admin=payload.get("is_admin", False)
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
        workos_user_id=payload.get("workos_user_id"),
        customer_id=payload.get("customer_id"),
        is_admin=payload.get("is_admin", False)
    )

def require_admin():
    """Dependency to require admin role (from WorkOS)"""
    async def admin_checker(user: User = Depends(require_auth)) -> User:
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        return user
    return admin_checker

def get_authorization_url() -> str:
    # Don't force organization - let users log in freely
    # We'll check their org membership after authentication
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

    # Check if user is admin via WorkOS organization membership
    # We check ALL their memberships, not just from auth response
    ADMIN_ROLE_SLUGS = {"admin", "owner", "marketing_manager"}

    is_admin = False
    organization_id = None

    try:
        # List ALL organization memberships for this user
        memberships = workos_client.user_management.list_organization_memberships(
            user_id=workos_user.id
        )
        print(f"[AUTH DEBUG] User: {workos_user.email}")
        print(f"[AUTH DEBUG] Memberships found: {len(memberships.data)}")

        for membership in memberships.data:
            # Handle role as dict or object
            role = membership.role
            role_slug = role.get('slug') if isinstance(role, dict) else (role.slug if role else None)
            print(f"[AUTH DEBUG] Membership: org={membership.organization_id}, role_slug={role_slug}")

            # Use the first org as customer_id
            if not organization_id:
                organization_id = membership.organization_id
            # Check if user has an admin-level role
            if role_slug and role_slug in ADMIN_ROLE_SLUGS:
                is_admin = True
                organization_id = membership.organization_id  # Use admin org
                print(f"[AUTH DEBUG] Admin role found: {role_slug}")
                break
    except Exception as e:
        print(f"[AUTH DEBUG] Error fetching memberships: {e}")
        pass

    print(f"[AUTH DEBUG] Final: is_admin={is_admin}, organization_id={organization_id}")

    # Check if user exists
    existing_user = await db.users.find_one({"workos_user_id": workos_user.id})

    now = datetime.utcnow()

    # IMPORTANT: Always use workos_user_id as customer_id for consistency
    # Organization membership is only used for determining admin status
    customer_id = workos_user.id

    if existing_user:
        # Update existing user
        update_fields = {
            "email": workos_user.email,
            "first_name": workos_user.first_name,
            "last_name": workos_user.last_name,
            "customer_id": customer_id,  # Always set to workos_user_id
            "last_login": now,
            "updated_at": now
        }

        await db.users.update_one(
            {"_id": existing_user["_id"]},
            {"$set": update_fields}
        )
        user_id = str(existing_user["_id"])
    else:
        # Create new user
        new_user = {
            "workos_user_id": workos_user.id,
            "email": workos_user.email,
            "first_name": workos_user.first_name,
            "last_name": workos_user.last_name,
            "customer_id": customer_id,  # Always set to workos_user_id
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
        "customer_id": customer_id,
        "is_admin": is_admin,
    }

    token = create_token(user_data)

    return TokenResponse(
        access_token=token,
        user=User(**user_data)
    )
