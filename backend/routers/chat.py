from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from graph.workflow import workflow_instance

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

class ChatRequest(BaseModel):
    """Chat request model"""
    query: str

@router.post("/chat")
async def chat(request: ChatRequest):
    """Chat with the AI assistant"""
    try:
        logger.info(f"API: Processing chat query: {request.query}")
        result = workflow_instance.chat(request.query)
        
        return {
            "status": "success",
            "query": request.query,
            "response": result["response"]
        }
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
