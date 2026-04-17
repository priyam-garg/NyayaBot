from datetime import datetime, timezone
from bson import ObjectId
from fastapi import APIRouter, HTTPException, status
from pymongo.errors import DuplicateKeyError

from app.models.schemas import SignupRequest, LoginRequest, TokenResponse, UserPublic
from app.services.mongo import users_col
from app.services.security import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


def _to_public(doc) -> UserPublic:
    return UserPublic(id=str(doc["_id"]), name=doc["name"], email=doc["email"])


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest) -> TokenResponse:
    doc = {
        "name": body.name.strip(),
        "email": body.email.lower(),
        "password_hash": hash_password(body.password),
        "created_at": datetime.now(timezone.utc),
    }
    try:
        result = await users_col().insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    doc["_id"] = result.inserted_id
    token = create_access_token(str(result.inserted_id))
    return TokenResponse(access_token=token, user=_to_public(doc))


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    doc = await users_col().find_one({"email": body.email.lower()})
    if not doc or not verify_password(body.password, doc["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(str(doc["_id"]))
    return TokenResponse(access_token=token, user=_to_public(doc))
