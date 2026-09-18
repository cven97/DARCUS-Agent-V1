from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str

class AutoApproveRequest(BaseModel):
    command_base: str

class ModelConfigUpdate(BaseModel):
    server_url: str
    default_model: str
    temperature: float
    max_tokens: int
    system_prompt: Optional[str] = None
    memory_prompt: Optional[str] = None
    planner_prompt: Optional[str] = None

