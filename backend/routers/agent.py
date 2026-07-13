from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import agent coordinator
from core.agent.agent import run_agent_query

router = APIRouter(prefix="/api/agent", tags=["AI Analyst"])

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"

@router.post("/chat")
async def chat_with_analyst(request: ChatRequest):
    """
    POST endpoint to send queries to the AI Retail Analyst.
    Runs classification, structured tool execution, and returns natural language responses.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
    try:
        response_text = await run_agent_query(request.message, request.session_id)
        return {
            "response": response_text
        }
    except Exception as e:
        import logging
        logger = logging.getLogger("insightforge")
        logger.error(f"Error in analyst conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An error occurred in the analyst conversation. Please try again."
        )
