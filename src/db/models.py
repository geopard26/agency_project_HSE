import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, nullable=False, index=True)
    features = Column(JSON, nullable=False)
    label = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
