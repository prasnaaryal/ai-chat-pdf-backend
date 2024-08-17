from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from db import Base

class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255))
    context = Column(Text)

    chats = relationship("Chat", back_populates="chat_history")

class Chat(Base):
    __tablename__ = 'chat'

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_history_id = Column(Integer, ForeignKey('chat_history.id'))
    question = Column(Text)
    answer = Column(Text)

    chat_history = relationship("ChatHistory", back_populates="chats")
