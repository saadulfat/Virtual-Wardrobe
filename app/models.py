from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from enum import Enum as PyEnum
from datetime import datetime

Base = declarative_base()

class GenderEnum(PyEnum):
    male = "male"
    female = "female"
    other = "other"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    gender = Column(Enum(GenderEnum), nullable=False)
    age = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)  # in cm
    outfits = relationship("Outfit", back_populates="user")
    chat_messages = relationship("ChatMessage", back_populates="user")
    model_images = relationship("ModelImage", back_populates="user")


class Outfit(Base):
    __tablename__ = "outfits"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # MySQL requires explicit VARCHAR lengths. Adjust if needed.
    category = Column(String(100), nullable=False)  # <-- set desired max length
    subcategory = Column(String(100), nullable=True)  # <-- subcategory like "t-shirt", "jeans", etc.
    image_path = Column(String(500), nullable=False)  # <-- set desired max length
    primary_color = Column(String(50), nullable=True)  # <-- detected primary color name
    secondary_color = Column(String(50), nullable=True)  # <-- detected secondary color name
    color_confidence = Column(Integer, nullable=True)  # <-- confidence percentage
    color_data = Column(Text, nullable=True)  # <-- JSON string with detailed color info
    user = relationship("User", back_populates="outfits")


class ModelImage(Base):
    __tablename__ = "model_images"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="model_images")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message = Column(Text, nullable=False)  # User's message
    response = Column(Text, nullable=True)  # AI's response
    is_from_user = Column(Boolean, default=True)  # True for user messages, False for AI responses
    created_at = Column(DateTime, default=datetime.utcnow)
    message_type = Column(String(50), default="general")  # general, outfit_suggestion, trend_query, etc.
    context_data = Column(Text, nullable=True)  # JSON string for additional context
    user = relationship("User", back_populates="chat_messages")