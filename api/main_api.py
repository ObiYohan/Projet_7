from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks, Security, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager
import sys
import os
import subprocess
import faiss
from requests import request

# Add parent directory to path to import chatbot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chatbot import EventChatbot

# Global chatbot instance
chatbot_instance = None

# Security: API Key for sensitive endpoints
API_KEY = os.getenv("API_KEY", "your-super-secret-key-change-me-in-production")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for protected endpoints"""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key"
        )
    return api_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown)"""
    # Startup
    global chatbot_instance
    try:
        chatbot_instance = EventChatbot()
        print("✅ Chatbot initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        chatbot_instance = None
    
    yield  # Application is running
    
    # Shutdown (optional cleanup)
    print("🔄 Shutting down chatbot...")
    chatbot_instance = None

app = FastAPI(
    title="Event Chatbot API",
    description="API pour recommander des événements culturels avec RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS pour une environnement de développement
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Question de l'utilisateur", min_length=1)
    k: int = Field(default=5, description="Nombre d'événements à rechercher", ge=1, le=20)
    show_sources: bool = Field(default=True, description="Afficher les sources utilisées")

class Source(BaseModel):
    uid: str
    chunk_id: int
    text: str
    distance: float
    firstdate_begin: Optional[str] = None
    lastdate_end: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    sources: Optional[List[Source]] = None
    query: str
    k: int

class HealthResponse(BaseModel):
    status: str
    chatbot_loaded: bool
    index_size: Optional[int] = None

class RebuildResponse(BaseModel):
    status: str
    message: str
    index_size: int
    total_events: int
    total_chunks: int


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Vérifie l'état de l'API et du chatbot"""
    if chatbot_instance is None:
        return HealthResponse(
            status="unhealthy",
            chatbot_loaded=False,
            index_size=None
        )
    
    return HealthResponse(
        status="healthy",
        chatbot_loaded=True,
        index_size=chatbot_instance.index.ntotal
    )

# Main chat endpoint
@app.post("/ask", response_model=QueryResponse, tags=["Chatbot"])
async def ask(request: QueryRequest):
    """
    Pose une question au chatbot et obtient une réponse augmentée
    
    - **query**: La question de l'utilisateur
    - **k**: Nombre d'événements similaires à rechercher (1-20)
    - **show_sources**: Inclure les sources dans la réponse
    """
    if chatbot_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized. Please check server logs."
        )
    
    try:
        # Get chatbot response
        result = chatbot_instance.chat(
            request.query,
            k=request.k,
            show_sources=False
        )
        
        # Format sources if requested
        sources = None
        if request.show_sources:
            sources = []
            for idx, (_, row) in enumerate(result["sources"].iterrows()):
                sources.append(Source(
                    uid=str(row["uid"]),
                    chunk_id=int(row["chunk_id"]),
                    text=row["text"][:500] + "..." if len(row["text"]) > 500 else row["text"],
                    distance=float(result["distances"][idx]),
                    firstdate_begin=str(row.get("firstdate_begin")) if "firstdate_begin" in row else None,
                    lastdate_end=str(row.get("lastdate_end")) if "lastdate_end" in row else None
                ))
        
        return QueryResponse(
            response=result["response"],
            sources=sources,
            query=request.query,
            k=request.k
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# Rebuild vector database
@app.post("/rebuild", response_model=RebuildResponse, tags=["System"])
async def rebuild_index(api_key: str = Depends(verify_api_key)):
    """
    Rebuild the FAISS index with new data (requires API key)
    
    **Important**: Place your new data file at `data/new-data.json` before calling this endpoint.
    """
    global chatbot_instance
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    new_data_file = os.path.join(base_dir, "data", "new-data.json")
    
    # Check if new data file exists
    if not os.path.exists(new_data_file):
        raise HTTPException(
            status_code=400,
            detail=f"New data file not found at: {new_data_file}. Please upload your data file first."
        )
    
    try:
        # Étape 1: Exécuter le script de chargement des nouvelles données
        print("📥 Chargement des nouvelles données...")
        result = subprocess.run(
            [sys.executable, "src/load_new_data.py"],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        print(result.stdout)
        
        # Étape 2: Réinitialiser le chatbot avec les nouvelles données
        print("🔄 Réinitialisation du chatbot...")
        chatbot_instance = EventChatbot()
        
        return RebuildResponse(
            status="success",
            message="Index rebuilt successfully",
            index_size=chatbot_instance.index.ntotal,
            total_events=len(chatbot_instance.chunks_df["uid"].unique()),
            total_chunks=len(chatbot_instance.chunks_df)
        )
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du chargement des données: {e.stderr}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load new data: {e.stderr}"
        )
    except Exception as e:
        print(f"❌ Erreur lors de la reconstruction: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to rebuild index: {str(e)}"
        )
    
# Get chatbot statistics
@app.get("/stats", tags=["System"])
async def get_stats():
    """Obtient les statistiques du chatbot"""
    if chatbot_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not initialized"
        )
    
    try:
        return {
            "total_events": len(chatbot_instance.chunks_df["uid"].unique()),
            "total_chunks": len(chatbot_instance.chunks_df),
            "index_size": chatbot_instance.index.ntotal,
            "embedding_dimension": chatbot_instance.index.d
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """Get current index status"""
    if chatbot_instance is None:
        return {
            "index_loaded": False,
            "total_chunks": 0,
            "index_size": 0,
            "data_file_exists": os.path.exists("../data/chunks_with_embeddings.json"),
            "index_file_exists": os.path.exists("../data/faiss_index.idx")
        }
    
    return {
        "index_loaded": True,
        "total_chunks": len(chatbot_instance.chunks_df),
        "index_size": chatbot_instance.index.ntotal,
        "data_file_exists": os.path.exists("../data/chunks_with_embeddings.json"),
        "index_file_exists": os.path.exists("../data/faiss_index.idx")
    }

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Event Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)