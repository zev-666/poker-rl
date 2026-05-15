from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://poker:poker@localhost:5432/poker_logs")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class DecisionLog(Base):
    __tablename__ = "decision_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    round_idx = Column(Integer)
    hole_cards = Column(JSON)
    community_cards = Column(JSON)
    bet_history = Column(JSON)
    pot_size = Column(Integer)
    stack_size = Column(Integer)
    action_taken = Column(String)
    strategy_used = Column(String, nullable=True)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def save_decision(decision_data: dict):
    async with async_session() as session:
        async with session.begin():
            log = DecisionLog(**decision_data)
            session.add(log)
