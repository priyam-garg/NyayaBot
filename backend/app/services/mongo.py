from functools import lru_cache
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import ASCENDING, DESCENDING
from app.config import get_settings


@lru_cache
def get_client() -> AsyncIOMotorClient:
    return AsyncIOMotorClient(get_settings().mongodb_uri)


def get_db() -> AsyncIOMotorDatabase:
    return get_client()[get_settings().mongodb_db]


def users_col() -> AsyncIOMotorCollection:
    return get_db()["users"]


def sessions_col() -> AsyncIOMotorCollection:
    return get_db()["sessions"]


def messages_col() -> AsyncIOMotorCollection:
    return get_db()["messages"]


def documents_col() -> AsyncIOMotorCollection:
    return get_db()["documents"]


async def ensure_indexes() -> None:
    await users_col().create_index("email", unique=True)
    await sessions_col().create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
    await messages_col().create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])
    await documents_col().create_index("session_id", unique=True)
    await documents_col().create_index("doc_id", unique=True)
    await documents_col().create_index([("user_id", ASCENDING), ("uploaded_at", DESCENDING)])
