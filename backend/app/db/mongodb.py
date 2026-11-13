from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings
from app.core.logging import logger

class MongoDBManager:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None

db_manager = MongoDBManager()

async def connect_to_mongo():
    logger.info("Connecting to MongoDB Atlas...")
    try:
        db_manager.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT_MS,
            uuidRepresentation="standard"
        )
        db_manager.db = db_manager.client[settings.MONGODB_DB_NAME]
        # Quick ping to verify connectivity
        await db_manager.client.admin.command('ping')
        logger.info(f"Connected to MongoDB Atlas: database '{settings.MONGODB_DB_NAME}'")
    except Exception as e:
        logger.warning(f"Could not connect to live MongoDB: {e}. Running with database in fallback mode if applicable.")

async def close_mongo_connection():
    if db_manager.client:
        logger.info("Closing MongoDB connection...")
        db_manager.client.close()
        db_manager.client = None
        db_manager.db = None

def get_database() -> Optional[AsyncIOMotorDatabase]:
    return db_manager.db
