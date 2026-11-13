from pymongo import ASCENDING, DESCENDING
from app.db.mongodb import get_database
from app.core.logging import logger

async def create_indexes():
    """Ensure all required indexes exist on MongoDB collections."""
    db = get_database()
    if db is None:
        logger.warning("Database unavailable, skipping index creation.")
        return

    try:
        # Users Collection Indexes
        await db.users.create_index([("email", ASCENDING)], unique=True, name="idx_users_email_unique")
        logger.info("Created unique index on users.email")

        # Analyses Collection Indexes
        await db.analyses.create_index([("analysis_id", ASCENDING)], unique=True, name="idx_analyses_analysis_id_unique")
        await db.analyses.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)], name="idx_analyses_user_created")
        await db.analyses.create_index([("user_id", ASCENDING), ("status", ASCENDING)], name="idx_analyses_user_status")
        await db.analyses.create_index([("user_id", ASCENDING), ("classification.predicted_class", ASCENDING)], name="idx_analyses_user_predicted_class")
        logger.info("Created indexes on analyses collection")

        # Models Collection Index
        await db.models.create_index([("id", ASCENDING)], unique=True, name="idx_models_id_unique")
        logger.info("Created index on models collection")

    except Exception as e:
        logger.error(f"Error creating MongoDB indexes: {e}")
