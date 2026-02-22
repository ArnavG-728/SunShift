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
pool_config = {}

if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
    # SQLite uses StaticPool for thread safety
    from sqlalchemy.pool import StaticPool
    pool_config = {"poolclass": StaticPool}
else:
    # PostgreSQL connection pooling for concurrent requests
    pool_config = {
        "pool_size": 10,          # Base concurrent connections
        "max_overflow": 20,       # Burst capacity (total 30)
        "pool_pre_ping": True,    # Health check before use
        "pool_recycle": 300,      # Recycle connections every 5 min
    }

engine = create_engine(DATABASE_URL, connect_args=connect_args, **pool_config)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables and handle migrations. Safe to call multiple times."""
    from . import models  # Import models to register them
    from sqlalchemy import inspect, text
    
    try:
        # 1. Create tables if they don't exist
        Base.metadata.create_all(bind=engine, checkfirst=True)
        
        # 2. Check for missing columns (Self-healing migration)
        # We uniquely handle simulation_state.location_key which was added in v2
        inspector = inspect(engine)
        columns = [c['name'] for c in inspector.get_columns("simulation_state")]
        
        if "location_key" not in columns:
            logger.info("Migration: Adding missing 'location_key' column to simulation_state table...")
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE simulation_state ADD COLUMN location_key VARCHAR"))
                conn.execute(text("CREATE INDEX ix_simulation_state_location_key ON simulation_state (location_key)"))
            logger.info("Migration: Successfully added location_key column")
            
        logger.info("Database tables initialized successfully")
    except Exception as e:
        # Log but don't crash - tables likely already exist or have been modified
        logger.warning(f"Database init warning: {e}")
