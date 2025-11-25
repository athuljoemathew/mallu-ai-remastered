import os
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    # Replace with your actual Gemini API key from https://aistudio.google.com/app/apikey
    API_KEY = "AIzaSyD2EIDJAn7ZlGIpu_D8COQcCXMtWZ9C0tA"  
    MODEL_NAME = "gemini-1.5-flash"

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(1000, ge=1, le=4000)

class ChatResponse(BaseModel):
    reply: str
    token_count: Optional[int]
    request_id: str
    timestamp: str

class EnhancedGenAIClient:
    def __init__(self, api_key: str):
        try:
            self.client = genai.Client(api_key=api_key)
            logger.info("✅ Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {str(e)}")
            raise

    async def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            logger.info(f"🤖 Sending request to Gemini: {prompt[:100]}...")
            
            response = self.client.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                generation_config=types.GenerationConfig(
                    temperature=kwargs.get('temperature', 0.7),
                    max_output_tokens=kwargs.get('max_tokens', 1000),
                    top_p=0.8,
                )
            )
            
            logger.info("✅ Received response from Gemini")
            return {
                "text": response.text,
                "token_count": getattr(response, 'usage_metadata', {}).get('total_token_count', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Gemini API error: {str(e)}")
            # Provide a helpful error message
            error_msg = f"I apologize, but I'm having trouble connecting to the AI service. "
            error_msg += f"Error: {str(e)}. "
            error_msg += "Please check if the API key is valid and has proper permissions."
            
            return {
                "text": error_msg,
                "token_count": 0
            }

def get_genai_client() -> EnhancedGenAIClient:
    return EnhancedGenAIClient(Config.API_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Mallu AI Server...")
    
    # Test the API key on startup
    try:
        client = genai.Client(api_key=Config.API_KEY)
        test_response = client.models.generate_content(
            model=Config.MODEL_NAME,
            contents="Hello"
        )
        logger.info("✅ Gemini API connection test successful!")
    except Exception as e:
        logger.error(f"❌ Gemini API connection failed: {str(e)}")
        logger.error("💡 Get a free API key from: https://aistudio.google.com/app/apikey")
        logger.error("💡 Replace the API_KEY in the code with your valid key")
    
    yield
    logger.info("🛑 Shutting down Mallu AI Server")

app = FastAPI(
    title="Mallu AI API",
    description="AI Assistant with Gemini integration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    genai_client: EnhancedGenAIClient = Depends(get_genai_client)
):
    request_id = str(uuid.uuid4())
    logger.info(f"📨 Received chat request: {request.prompt[:50]}...")
    
    result = await genai_client.generate_text(
        prompt=request.prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    response = ChatResponse(
        reply=result["text"],
        token_count=result.get("token_count"),
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat()
    )
    
    logger.info(f"📤 Sent response: {response.reply[:50]}...")
    return response

@app.get("/health")
async def health_check():
    # Test the API connection for health check
    try:
        client = genai.Client(api_key=Config.API_KEY)
        test_response = client.models.generate_content(
            model=Config.MODEL_NAME,
            contents="Health check"
        )
        ai_status = "healthy"
    except Exception as e:
        ai_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "ai_service": ai_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "model": Config.MODEL_NAME
    }

@app.get("/")
async def root():
    return {
        "message": "Mallu AI Server is running!", 
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Mallu AI Server Starting...")
    print("=" * 50)
    print(f"📡 Server URL: http://localhost:8000")
    print(f"🔗 Health check: http://localhost:8000/health")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print(f"🤖 AI Model: {Config.MODEL_NAME}")
    print("=" * 50)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")