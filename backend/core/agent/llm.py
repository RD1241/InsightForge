import os
import json
import httpx
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# --- Config and Defaults ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# In-memory session memory for v1 (retains last 5 message pairs)
session_memories = {}

def get_session_history(session_id: str) -> list:
    """
    Retrieves the short term memory history for a session ID.
    Limits to last 10 messages (5 turns).
    """
    if session_id not in session_memories:
        session_memories[session_id] = []
    return session_memories[session_id]

def add_message_to_history(session_id: str, role: str, content: str):
    """
    Adds a message to the session memory, maintaining the limit.
    """
    history = get_session_history(session_id)
    history.append({"role": role, "content": content})
    # Keep last 10 messages (5 user, 5 assistant)
    if len(history) > 10:
        session_memories[session_id] = history[-10:]

def call_llm(messages: list, require_json: bool = False) -> str:
    """
    Unified client that routes Chat Completions to either Groq or Ollama.
    """
    # 1. Setup API parameters based on LLM_PROVIDER
    if LLM_PROVIDER == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        model = GROQ_MODEL
        if not GROQ_API_KEY:
            raise ValueError("LLM_PROVIDER is set to 'groq' but GROQ_API_KEY is missing from environment (.env).")
    else:
        # Ollama endpoint
        url = f"{OLLAMA_HOST.rstrip('/')}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        model = OLLAMA_MODEL
        
    # 2. Build payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1
    }
    
    # Enable JSON output mode if requested and supported
    if require_json:
        payload["response_format"] = {"type": "json_object"}
        
    try:
        # Send synchronous POST request using httpx
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                raise ValueError(
                    f"LLM API request failed with status {response.status_code}: {response.text}"
                )
                
            res_json = response.json()
            completion_text = res_json["choices"][0]["message"]["content"]
            return completion_text
            
    except Exception as e:
        print(f"Error calling LLM provider '{LLM_PROVIDER}': {str(e)}")
        # Raise standard error so caller handles fallback/alert
        raise e
