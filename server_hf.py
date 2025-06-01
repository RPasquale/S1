from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import os, shutil, importlib, asyncio, time, threading, json, mimetypes
from typing import List, Dict, Any, Optional, Union
import model_hf  # We'll create this
from model_training import get_trainer, EmbeddingModelTrainer, process_documents
from cfa_embedding_controller import CFAEmbeddingController
from datetime import datetime

# Initialize app
app = FastAPI(title="S1 Agent - PDF QA Chatbot API (Hugging Face)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"],
)

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

@app.on_event("startup")
async def startup_event():
    """Initialize Hugging Face models on server startup"""
    print("=" * 60)
    print("🚀 Starting S1 Agent Server (Hugging Face)")
    print("=" * 60)
    
    # Initialize models in background to not block startup
    def init_models():
        try:
            print("🤖 Initializing Hugging Face models...")
            model_hf.initialize_models()
            print("✅ Models initialized successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize models: {e}")
            print("Models will be initialized on first use.")
    
    # Run in background thread to not block server startup
    models_thread = threading.Thread(target=init_models)
    models_thread.daemon = True
    models_thread.start()

@app.post("/api/chat")
async def chat(payload: dict):
    """Process chat messages and return answers using Hugging Face models."""
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
        # Get answer using Hugging Face models
        answer = model_hf.answer_question_with_docs(question)
        
    except Exception as e:
        # Handle any errors gracefully
        error_msg = f"❌ **Error Processing Your Question**\n\n{str(e)}\n\nThe system is using Hugging Face models directly."
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

@app.post("/api/upload")
async def upload(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload PDF files and build search index."""
    upload_count = 0
    for f in files:
        try:
            # Sanitize filename
            filename = os.path.basename(f.filename)
            filename = filename.replace('/', '_').replace('\\', '_')
            invalid_chars = '<>:"|?*'
            for char in invalid_chars:
                filename = filename.replace(char, '_')
            dest = os.path.join(model_hf.DOC_FOLDER, filename)
            with open(dest, "wb") as out:
                out.write(await f.read())
            upload_count += 1
        except Exception as e:
            return {"status": "error", "message": f"Error saving {f.filename}: {str(e)}"}
    
    # Remove existing index to force rebuild
    if os.path.exists(model_hf.INDEX_FOLDER):
        shutil.rmtree(model_hf.INDEX_FOLDER)
    
    # Reload model to trigger indexing
    importlib.reload(model_hf)
    
    return {
        "status": "indexed", 
        "files_uploaded": upload_count,
        "message": f"Successfully uploaded {upload_count} files"
    }

@app.get("/api/conversations")
async def get_conversations():
    """Get list of all conversation IDs."""
    return {"conversation_ids": list(conversations.keys())}

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get messages from a specific conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"messages": conversations[conversation_id]}

@app.post("/api/conversations")
async def create_conversation(payload: dict):
    """Create a new conversation."""
    conversation_id = payload.get("conversation_id", f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if conversation_id in conversations:
        raise HTTPException(status_code=400, detail="Conversation ID already exists")
    
    conversations[conversation_id] = []
    return {"conversation_id": conversation_id}

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[conversation_id]
    return {"status": "deleted", "conversation_id": conversation_id}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test if models are available
        models_ready = model_hf.test_models()
        
        return {
            "status": "healthy" if models_ready else "degraded",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "huggingface_models": {
                    "ready": models_ready,
                    "embedding_model": "sentence-transformers or custom CFA model",
                    "language_model": "microsoft/DialoGPT-medium or similar"
                }
            },
            "capabilities": {
                "chat": models_ready,
                "document_search": True,
                "embedding_training": True,
                "file_upload": True
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/api/models/status")
async def models_status():
    """Get model status and information"""
    try:
        status = model_hf.get_models_status()
        return status
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# File serving endpoints
@app.get("/api/files/list")
async def list_uploaded_files():
    """List all uploaded files with their structure."""
    try:
        files = []
        
        def scan_directory(directory, relative_path=""):
            items = []
            try:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    relative_item_path = os.path.join(relative_path, item) if relative_path else item
                    
                    if os.path.isfile(item_path):
                        stat = os.stat(item_path)
                        items.append({
                            "name": item,
                            "path": relative_item_path.replace("\\", "/"),
                            "type": "file",
                            "size": stat.st_size,
                            "lastModified": int(stat.st_mtime * 1000)
                        })
                    elif os.path.isdir(item_path):
                        children = scan_directory(item_path, relative_item_path)
                        items.append({
                            "name": item,
                            "path": relative_item_path.replace("\\", "/"),
                            "type": "folder",
                            "children": children
                        })
            except PermissionError:
                pass
            
            return items
        
        if os.path.exists(model_hf.DOC_FOLDER):
            files = scan_directory(model_hf.DOC_FOLDER)
        
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

# Training endpoints
@app.post("/api/train/embedding")
async def start_embedding_training(
    background_tasks: BackgroundTasks,
    files: Optional[List[UploadFile]] = File(None),
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    use_uploaded_docs: bool = True
):
    """Start embedding model training with uploaded documents"""
    
    # Check if already training
    trainer = get_trainer()
    if trainer.is_model_training():
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    try:
        documents = []
        temp_dir = None
        
        # Use documents from DOC_FOLDER
        if use_uploaded_docs:
            from model_training import get_uploaded_documents
            documents = get_uploaded_documents()
            if not documents:
                raise HTTPException(status_code=400, detail="No documents found in upload folder. Please upload documents first.")
        
        # Process newly uploaded files
        elif files:
            temp_dir = "temp_training_docs"
            os.makedirs(temp_dir, exist_ok=True)
            
            file_paths = []
            for file in files:
                filename = os.path.basename(file.filename)
                filename = filename.replace('/', '_').replace('\\', '_')
                invalid_chars = '<>:"|?*'
                for char in invalid_chars:
                    filename = filename.replace(char, '_')
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                file_paths.append(file_path)
            
            documents = process_documents(file_paths)
            
        if not documents:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="No valid documents found")
        
        # Start training in background
        background_tasks.add_task(
            run_embedding_training,
            trainer,
            documents,
            temp_dir,
            epochs,
            batch_size,
            learning_rate
        )
        
        source = "uploaded folder" if use_uploaded_docs else "newly uploaded files"
        return {
            "message": f"Training started successfully using documents from {source}",
            "num_documents": len(documents),
            "document_source": source,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
        
    except Exception as e:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

async def run_embedding_training(
    trainer: EmbeddingModelTrainer,
    documents: List[str],
    temp_dir: Optional[str],
    epochs: int,
    batch_size: int,
    learning_rate: float
):
    """Run embedding training in background"""
    try:
        result = await trainer.train_embedding_model(
            documents=documents,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        print(f"Training completed: {result}")
        
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/api/train/status")
async def get_training_status():
    """Get current training status"""
    trainer = get_trainer()
    
    return {
        "is_training": trainer.is_model_training(),
        "progress": trainer.get_training_progress()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting S1 Agent Server (Hugging Face)...")
    print("📋 Server will be available at: http://localhost:8000")
    print("📊 API documentation at: http://localhost:8000/docs")
    print("🤗 Using Hugging Face models directly - no Ollama needed!")
    print("=" * 60)
    
    uvicorn.run(
        "server_hf:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
