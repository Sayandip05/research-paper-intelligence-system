"""
MongoDB Connection Singleton
"""
from pymongo import MongoClient
from app.config import get_settings

_mongo_client = None
_mongo_db = None


def get_mongo_db():
    """Get MongoDB database instance (singleton)"""
    global _mongo_client, _mongo_db
    if _mongo_db is None:
        settings = get_settings()
        _mongo_client = MongoClient(settings.mongodb_uri)
        _mongo_db = _mongo_client[settings.mongodb_db_name]
        print(f"✅ MongoDB connected: {settings.mongodb_db_name}")
    return _mongo_db
