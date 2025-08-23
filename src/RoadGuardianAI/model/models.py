from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, JSON, func
from RoadGuardianAI.utils.db import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    segment_id = Column(String(64), nullable=False)
    ts = Column(TIMESTAMP, nullable=False)
    risk_score = Column(Float, nullable=False)
    features = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
