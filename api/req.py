from pydantic import BaseModel

class ConversationRequest(BaseModel):
    chat_id: int
    question: str
