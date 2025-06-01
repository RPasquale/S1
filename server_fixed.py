from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import os, shutil, importlib, asyncio, subprocess, time, requests, threading, json, mimetypes
from typing import List, Dict, Any, Optional, Union
import model
from model_training import get_trainer, EmbeddingModelTrainer, process_documents
# Import CFA embedding controller
from cfa_embedding_controller import CFAEmbeddingController
from datetime import datetime

# Ollama Management Functions
def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama_service():
    """Start Ollama service in background"""
    try:
        # First check if ollama is installed
        if not check_ollama_installed():
            print("✗ Ollama is not installed or not in PATH")
            print("📥 Please install Ollama from: https://ollama.ai/download")
            return False
            
        # Check if already running
        if check_ollama_running():
            print("✓ Ollama service is already running")
            return True
            
        print("🚀 Starting Ollama service...")
        
        if os.name == 'nt':  # Windows
            # Start ollama serve in background
            subprocess.Popen(['ollama', 'serve'], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Linux/Mac
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        print("⏳ Waiting for Ollama service to start...")
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if check_ollama_running():
                print("✓ Ollama service started successfully")
                return True
        
        print("✗ Failed to start Ollama service within 30 seconds")
        return False
        
    except Exception as e:
        print(f"✗ Error starting Ollama service: {e}")
        return False

def check_model_available(model_name="deepseek-r1:1.5b"):
    """Check if the required model is available in Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            for model_info in models:
                if model_info.get('name') == model_name:
                    return True
        return False
    except:
        return False

def pull_model(model_name="deepseek-r1:1.5b"):
    """Pull the required model if not available"""
    try:
        # First check if ollama is installed
        if not check_ollama_installed():
            print("✗ Ollama is not installed. Cannot pull model.")
            print("📥 Please install Ollama from: https://ollama.ai/download")
            return False
            
        print(f"🔄 Pulling model {model_name}... This may take several minutes.")
        result = subprocess.run(['ollama', 'pull', model_name], 
                              capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print(f"✓ Successfully pulled model {model_name}")
            return True
        else:
            print(f"✗ Failed to pull model {model_name}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout while pulling model {model_name}")
        return False
    except (FileNotFoundError, OSError) as e:
        print(f"✗ Ollama command not found: {e}")
        print("📥 Please install Ollama from: https://ollama.ai/download")
        return False
    except Exception as e:
        print(f"✗ Error pulling model {model_name}: {e}")
        return False

def serve_model_directly(model_name="deepseek-r1:1.5b"):
    """
    Serve a model directly using Ollama's API without requiring ollama run.
    This makes the model available for chat requests automatically.
    """
    try:
        # First ensure the model is available
        if not check_model_available(model_name):
            print(f"⚠️ Model {model_name} not available, attempting to pull it...")
            if not pull_model(model_name):
                return False
        
        # Create a chat request to "warm up" the model and make it available
        print(f"🤖 Warming up model {model_name}...")
        warm_up_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        response = requests.post(
            'http://localhost:11434/api/chat', 
            json=warm_up_payload,
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"✓ Model {model_name} is now ready for serving")
            return True
        else:
            print(f"✗ Failed to warm up model {model_name}: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error serving model {model_name}: {e}")
        return False

def ensure_ollama_model_ready(model_name="deepseek-r1:1.5b"):
    """Ensure Ollama service is running and model is available"""
    print("🔍 Checking Ollama service and model availability...")
    
    # Step 0: Check if Ollama is installed
    if not check_ollama_installed():
        print("❌ Ollama is not installed!")
        print("📥 Please install Ollama from: https://ollama.ai/download")
        print("   After installation, restart your terminal and run:")
        print(f"   ollama pull {model_name}")
        print("   ollama serve")
        return False
    
    # Step 1: Check if Ollama is running
    if not check_ollama_running():
        print("🚀 Starting Ollama service...")
        if not start_ollama_service():
            print("❌ Failed to start Ollama service")
            print("💡 Try running manually: ollama serve")
            return False
    else:
        print("✓ Ollama service is already running")
    
    # Step 2: Check if model is available
    if not check_model_available(model_name):
        print(f"📦 Model {model_name} not found, pulling...")
        if not pull_model(model_name):
            print("❌ Failed to pull model")
            print(f"💡 Try running manually: ollama pull {model_name}")
            return False
    else:
        print(f"✓ Model {model_name} is available")
    
    # Step 3: Serve the model directly (make it available for chat)
    print(f"🤖 Serving model {model_name}...")
    if not serve_model_directly(model_name):
        print("⚠️ Model serving might have issues, but will try to continue...")
    
    print("🎉 Ollama and required model are ready!")
    return True

# Initialize app
app = FastAPI(title="PDF QA Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize Ollama and required models on server startup"""
    print("=" * 60)
    print("🚀 Starting S1 Agent Server")
    print("=" * 60)
    
    # Initialize Ollama and model in background to not block startup
    def init_ollama():
        try:
            ensure_ollama_model_ready("deepseek-r1:1.5b")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize Ollama: {e}")
            print("You may need to start Ollama manually with: ollama serve")
    
    # Run in background thread to not block server startup
    ollama_thread = threading.Thread(target=init_ollama)
    ollama_thread.daemon = True
    ollama_thread.start()

# WebSocket connections for live training updates
training_connections: List[WebSocket] = []

# Application state
training_status = {
    "is_training": False,
    "start_time": None,
    "progress": 0,
    "status_message": "",
    "completed_steps": [],
    "errors": []
}

# Track conversations
conversations = {}

@app.post("/api/chat")
async def chat(payload: dict):
    """Process chat messages and return answers."""
    question = payload.get("question", "")
    conversation_id = payload.get("conversation_id", "default")
    
    if not question:
        return {"answer": "No question provided."}
    
    # Store question in conversation history
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    conversations[conversation_id].append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        # Check if Ollama is running before attempting to get answer
        if not check_ollama_running():
            error_msg = "🔧 Ollama service is not running. Starting it now..."
            print(error_msg)
            
            # Try to start Ollama
            if not ensure_ollama_model_ready():
                error_response = ("❌ **Ollama Service Error**\n\n"
                               "I couldn't connect to the Ollama service. Please:\n"
                               "1. Make sure Ollama is installed\n"
                               "2. Run `ollama serve` in a terminal\n"
                               "3. Run `ollama pull deepseek-r1:1.5b` to get the required model\n\n"
                               "Then try your question again.")
                
                conversations[conversation_id].append({
                    "role": "assistant", 
                    "content": error_response,
                    "timestamp": datetime.now().isoformat()
                })
                
                return {"answer": error_response}
        
        # Check if model is available
        if not check_model_available("deepseek-r1:1.5b"):
            error_response = ("🤖 **Model Not Available**\n\n"
                           "The required model `deepseek-r1:1.5b` is not available. "
                           "I'm trying to download it now, which may take a few minutes...\n\n"
                           "Please wait and try your question again in a moment.")
            
            # Try to pull the model in background
            def pull_model_bg():
                pull_model("deepseek-r1:1.5b")
            
            thread = threading.Thread(target=pull_model_bg)
            thread.daemon = True
            thread.start()
            
            conversations[conversation_id].append({
                "role": "assistant",
                "content": error_response,
                "timestamp": datetime.now().isoformat()
            })
            
            return {"answer": error_response}
        
        # Get answer from model
        answer = model.answer_question_with_docs(question)
        
    except Exception as e:
        # Handle any other errors gracefully
        error_msg = f"❌ **Error Processing Your Question**\n\n{str(e)}\n\nPlease make sure:\n- Ollama is running (`ollama serve`)\n- The model is available (`ollama pull deepseek-r1:1.5b`)"
        print(f"Chat error: {e}")
        
        conversations[conversation_id].append({
            "role": "assistant",
            "content": error_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        return {"answer": error_msg}
    
    # Store answer in conversation history
    conversations[conversation_id].append({
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now().isoformat()
    })
    
    return {"answer": answer}

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    ollama_running = check_ollama_running()
    model_available = check_model_available("deepseek-r1:1.5b") if ollama_running else False
    
    return {
        "status": "healthy" if ollama_running and model_available else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ollama": {
                "running": ollama_running,
                "url": "http://localhost:11434"
            },
            "model": {
                "available": model_available,
                "name": "deepseek-r1:1.5b"
            }
        },
        "capabilities": {
            "sentence_transformers": True,
            "pylate": True,
            "embedding_training": True,
            "chat": ollama_running and model_available
        }
    }

@app.post("/api/ollama/start")
async def start_ollama():
    """Manually start Ollama service and pull required model"""
    try:
        success = ensure_ollama_model_ready("deepseek-r1:1.5b")
        if success:
            return {
                "status": "success",
                "message": "Ollama service started and model is ready",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error", 
                "message": "Failed to start Ollama or pull model",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error starting Ollama: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/ollama/status")
async def ollama_status():
    """Get Ollama service status"""
    try:
        ollama_running = check_ollama_running()
        model_available = check_model_available("deepseek-r1:1.5b") if ollama_running else False
        
        return {
            "ollama_running": ollama_running,
            "model_available": model_available,
            "model_name": "deepseek-r1:1.5b",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "ollama_running": False,
            "model_available": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting S1 Agent Server...")
    print("📋 Server will be available at: http://localhost:8000")
    print("📊 API documentation at: http://localhost:8000/docs")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(
        "server_fixed:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False for production
        log_level="info"
    )
