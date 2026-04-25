from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserPublic"


class UserPublic(BaseModel):
    id: str
    name: str
    email: EmailStr


class SessionCreate(BaseModel):
    title: str | None = None


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    doc_id: str | None = None


class Source(BaseModel):
    source: str
    score: float
    chunk_index: int
    origin: str = "legal"


class MessageOut(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime
    sources: list[Source] = []


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    top_score: float | None = None
    sources: list[Source] = []
    follow_ups: list[str] = []


class DocumentOut(BaseModel):
    doc_id: str
    session_id: str
    filename: str
    display_name: str
    chunk_count: int
    uploaded_at: datetime


class UploadResponse(BaseModel):
    doc_id: str
    display_name: str
    chunk_count: int


TokenResponse.model_rebuild()
