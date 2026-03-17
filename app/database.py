from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base
from dotenv import load_dotenv
import os
import time
import logging

# Load environment variables
load_dotenv()

# Get database URL from environment variables - using MySQL
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:123abc@localhost:3306/wardrobe")

# Initialize engine and session
engine = None
SessionLocal = None

def init_db():
    """Initialize database connection with retry logic"""
    global engine, SessionLocal
    
    if engine is None:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
    
    if SessionLocal is None:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Try to create tables with retry logic
    max_retries = 5
    for attempt in range(max_retries):
        try:
            Base.metadata.create_all(bind=engine)
            logging.info("Database tables created successfully")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to create database tables after {max_retries} attempts: {e}")
                raise
            logging.warning(f"Database connection attempt {attempt + 1} failed: {e}. Retrying in 2 seconds...")
            time.sleep(2)

def get_database_url():
    """Get the database URL for migrations and other scripts"""
    return DATABASE_URL

def get_db():
    """Get database session"""
    if engine is None or SessionLocal is None:
        init_db()
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()