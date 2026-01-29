from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Use a data directory for the DB
DB_DIR = Path(__file__).parent.parent / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)

# Support external DATABASE_URL (for PostgreSQL on Render) or fallback to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_DIR}/energy_regen.db")

# Fix for Render's postgres:// vs postgresql:// issue
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SQLite-specific connection args
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables. Safe to call multiple times."""
    from . import models  # Import models to register them
    try:
        # checkfirst=True prevents "table already exists" errors
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        # Log but don't crash - tables likely already exist from another worker
        logger.warning(f"Database init warning (likely harmless): {e}")
