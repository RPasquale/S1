from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import os, shutil, importlib, asyncio
from typing import List, Dict, Any, Optional, Union
import model
from model_training import get_trainer, EmbeddingModelTrainer, process_documents
# Import CFA embedding controller
from cfa_embedding_controller import CFAEmbeddingController
import json
from datetime import datetime
import mimetypes

# Initialize app
app = FastAPI(title="PDF QA Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
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

def extract_documents_from_pdfs():
    """Extract text from PDF files in the document folder."""
    from PyPDF2 import PdfReader
    documents = []
    
    for root, dirs, files in os.walk(model.DOC_FOLDER):
        for fname in files:
            if fname.lower().endswith('.pdf'):
                try:
                    file_path = os.path.join(root, fname)
                    reader = PdfReader(file_path)
                    text_pages = [page.extract_text() or "" for page in reader.pages]
                    documents.append("\n".join(text_pages))
                except Exception as e:
                    print(f"Error processing {fname}: {e}")
    
    return documents

async def train_models_background(background_tasks: BackgroundTasks):
    """Run model training in background to avoid blocking the API."""
    global training_status
    
    training_status["is_training"] = True
    training_status["start_time"] = datetime.now().isoformat()
    training_status["status_message"] = "Starting model training..."
    training_status["progress"] = 0
    training_status["completed_steps"] = []
    training_status["errors"] = []
    
    try:
        # Extract documents from PDFs
        training_status["status_message"] = "Extracting documents from PDFs..."
        documents = extract_documents_from_pdfs()
        if not documents:
            raise ValueError("No documents found or extracted.")
        
        training_status["progress"] = 20
        training_status["completed_steps"].append("document_extraction")
        
        # Initialize model trainer
        trainer = get_trainer()
        
        # Step 1: Prepare training data
        training_status["status_message"] = "Preparing training data..."
        training_data = trainer.prepare_training_data(documents)
        training_status["progress"] = 40
        training_status["completed_steps"].append("data_preparation")
        
        # Step 2: Train embedding model
        training_status["status_message"] = "Training embedding model..."
        result = await trainer.train_embedding_model(
            documents=documents,
            num_epochs=3,
            batch_size=16,
            learning_rate=2e-5
        )
        training_status["progress"] = 80
        training_status["completed_steps"].append("embedding_training")
        
        # Save training result summary
        output_dir = os.path.join("trained_models", "training-summary")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "num_documents": len(documents),
                "training_result": result,
                "completed_steps": training_status["completed_steps"],
            }, f, indent=2)
        
        # Complete
        training_status["status_message"] = "Training completed successfully."
        training_status["progress"] = 100
        
    except Exception as e:
        training_status["status_message"] = f"Error during training: {str(e)}"
        training_status["errors"].append(str(e))
        print(f"Training error: {e}")
    
    finally:
        training_status["is_training"] = False

@app.post("/api/upload")
async def upload(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload PDF files and build search index."""
    # Save uploaded PDFs into the doc_folder, preserving subpaths
    upload_count = 0
    for f in files:
        try:
            # Sanitize filename: remove directory separators and invalid characters
            filename = os.path.basename(f.filename)
            # Replace any remaining path separators with underscores
            filename = filename.replace('/', '_').replace('\\', '_')
            # Remove any invalid Windows filename characters
            invalid_chars = '<>:"|?*'
            for char in invalid_chars:
                filename = filename.replace(char, '_')
            dest = os.path.join(model.DOC_FOLDER, filename)
            with open(dest, "wb") as out:
                out.write(await f.read())
            upload_count += 1
        except Exception as e:
            return {"status": "error", "message": f"Error saving {f.filename}: {str(e)}"}
    
    # Remove existing index to force rebuild
    if os.path.exists(model.INDEX_FOLDER):
        shutil.rmtree(model.INDEX_FOLDER)
    
    # Reload model to trigger indexing on import
    importlib.reload(model)
    
    # Start background training task
    if upload_count > 0 and not training_status["is_training"]:
        background_tasks.add_task(train_models_background, background_tasks)
    
    return {
        "status": "indexed", 
        "files_uploaded": upload_count,
        "training_started": not training_status["is_training"]
    }

@app.post("/api/upload-training-documents")
async def upload_training_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload training documents for advanced training."""
    try:
        upload_count = 0
        for f in files:
            try:
                # Sanitize filename: remove directory separators and invalid characters
                filename = os.path.basename(f.filename)
                # Replace any remaining path separators with underscores
                filename = filename.replace('/', '_').replace('\\', '_')
                # Remove any invalid Windows filename characters
                invalid_chars = '<>:"|?*'
                for char in invalid_chars:
                    filename = filename.replace(char, '_')
                dest = os.path.join(model.DOC_FOLDER, filename)
                with open(dest, "wb") as out:
                    out.write(await f.read())
                upload_count += 1
            except Exception as e:
                return {"status": "error", "message": f"Error saving {f.filename}: {str(e)}"}
        
        return {
            "status": "success", 
            "files_uploaded": upload_count,
            "message": f"Successfully uploaded {upload_count} training documents"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading training documents: {str(e)}")

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
    
    # Get answer from model
    answer = model.answer_question_with_docs(question)
    
    # Store answer in conversation history
    conversations[conversation_id].append({
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now().isoformat()
    })
    
    return {"answer": answer}

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

@app.get("/api/training/status")
async def get_training_status():
    """Get current model training status."""
    return training_status

@app.post("/api/training/start")
async def start_training(background_tasks: BackgroundTasks):
    """Manually start model training."""
    global training_status
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    background_tasks.add_task(train_models_background, background_tasks)
    return {"status": "started"}

@app.post("/api/training/cancel")
async def cancel_training():
    """Cancel ongoing model training."""
    global training_status
    if not training_status["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    # In a real implementation, we would need a proper way to cancel the background task
    # This is just a simple state change
    training_status["is_training"] = False
    training_status["status_message"] = "Training cancelled by user"
    return {"status": "cancelled"}

# DSPy Function API endpoints
@app.post("/api/dspy/functions")
async def save_dspy_function(function_data: dict):
    """Save a new DSPy function."""
    try:
        # Create DSPy functions directory if it doesn't exist
        dspy_dir = os.path.join(".", "dspy_functions")
        os.makedirs(dspy_dir, exist_ok=True)
        
        # Save function definition to file
        function_file = os.path.join(dspy_dir, f"{function_data['name']}.json")
        with open(function_file, 'w') as f:
            json.dump(function_data, f, indent=2)
        
        return {"status": "success", "message": "DSPy function saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving DSPy function: {str(e)}")

@app.get("/api/dspy/functions")
async def get_dspy_functions():
    """Get all saved DSPy functions."""
    try:
        dspy_dir = os.path.join(".", "dspy_functions")
        if not os.path.exists(dspy_dir):
            return {"functions": []}
        
        functions = []
        for filename in os.listdir(dspy_dir):
            if filename.endswith('.json'):
                with open(os.path.join(dspy_dir, filename), 'r') as f:
                    function_data = json.load(f)
                    functions.append(function_data)
        
        return {"functions": functions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading DSPy functions: {str(e)}")

@app.delete("/api/dspy/functions/{function_name}")
async def delete_dspy_function(function_name: str):
    """Delete a DSPy function."""
    try:
        dspy_dir = os.path.join(".", "dspy_functions")
        function_file = os.path.join(dspy_dir, f"{function_name}.json")
        
        if os.path.exists(function_file):
            os.remove(function_file)
            return {"status": "success", "message": "DSPy function deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="DSPy function not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting DSPy function: {str(e)}")

# Data Extraction API endpoints  
@app.get("/api/data-extraction/status")
async def get_data_extraction_status():
    """Get data extraction system status."""
    try:
        # Check if data extraction server is running
        import requests
        response = requests.get("http://127.0.0.1:8001/status", timeout=5)
        server_status = response.json() if response.status_code == 200 else {"status": "offline"}
    except:
        server_status = {"status": "offline"}
    
    return {
        "server_status": server_status,
        "recent_extractions": [],  # TODO: Implement extraction history
        "statistics": {
            "total_extractions": 0,
            "success_rate": 0,
            "last_extraction": None
        }
    }

@app.post("/api/data-extraction/start")
async def start_data_extraction(extraction_config: dict):
    """Start a data extraction job."""
    try:
        # TODO: Integrate with data extraction server
        # For now, return a placeholder response
        return {
            "status": "started",
            "job_id": f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "message": "Data extraction job started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting data extraction: {str(e)}")

# Advanced Training API endpoints
@app.post("/api/training/advanced")
async def start_advanced_training(training_config: dict):
    """Start advanced training with custom configuration."""
    try:
        from model_training import start_training_session
        
        # Extract training parameters
        training_type = training_config.get("type", "next_token_prediction")
        documents = training_config.get("documents", [])
        file_paths = training_config.get("file_paths", [])
        model_config = training_config.get("model_config", {})
        
        # Start training session
        result = start_training_session(
            training_type=training_type,
            documents=documents,
            file_paths=file_paths,
            model_config=model_config
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting advanced training: {str(e)}")

@app.get("/api/training/capabilities")
async def get_training_capabilities():
    """Get available training capabilities."""
    try:
        from model_training import get_training_capabilities
        return get_training_capabilities()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training capabilities: {str(e)}")

# File serving endpoints for document viewer
@app.get("/api/files/view")
async def view_file(path: str = Query(...)):
    """Serve a file for viewing in the document viewer."""
    try:
        # Ensure the path is relative to the DOC_FOLDER
        file_path = os.path.join(model.DOC_FOLDER, path)
        
        # Security check: make sure the path is within DOC_FOLDER
        if not os.path.abspath(file_path).startswith(os.path.abspath(model.DOC_FOLDER)):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine content type
        content_type, _ = mimetypes.guess_type(file_path)
        content_type = content_type or "application/octet-stream"
        
        return FileResponse(file_path, media_type=content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

@app.get("/api/files/download")
async def download_file(path: str = Query(...)):
    """Serve a file for download."""
    try:
        # Ensure the path is relative to the DOC_FOLDER
        file_path = os.path.join(model.DOC_FOLDER, path)
        
        # Security check: make sure the path is within DOC_FOLDER
        if not os.path.abspath(file_path).startswith(os.path.abspath(model.DOC_FOLDER)):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine content type
        content_type, _ = mimetypes.guess_type(file_path)
        content_type = content_type or "application/octet-stream"
        
        return FileResponse(
            file_path, 
            media_type=content_type, 
            filename=os.path.basename(file_path),
            headers={"Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

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
                        # Get file stats
                        stat = os.stat(item_path)
                        items.append({
                            "name": item,
                            "path": relative_item_path.replace("\\", "/"),  # Use forward slashes for web
                            "type": "file",
                            "size": stat.st_size,
                            "lastModified": int(stat.st_mtime * 1000)  # Convert to milliseconds for JavaScript
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
                pass  # Skip directories we can't read
            
            return items
        
        if os.path.exists(model.DOC_FOLDER):
            files = scan_directory(model.DOC_FOLDER)
        
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

# WebSocket endpoint for live training updates
@app.websocket("/ws/training")
async def websocket_training_updates(websocket: WebSocket):
    """WebSocket endpoint for live training updates."""
    await websocket.accept()
    training_connections.append(websocket)
    
    try:
        while True:
            # Wait for a message from the client (optional, we can also push updates)
            data = await websocket.receive_text()
            print(f"Received from client: {data}")
    except WebSocketDisconnect:
        training_connections.remove(websocket)
        print("Training WebSocket disconnected")

async def broadcast_training_progress(progress_data: Dict[str, Any]):
    """Broadcast training progress to all connected WebSocket clients"""
    if training_connections:
        disconnected = []
        for connection in training_connections:
            try:
                await connection.send_text(json.dumps({
                    'type': 'training_progress',
                    'data': progress_data
                }))
            except Exception as e:
                print(f"Error sending progress update: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            if conn in training_connections:
                training_connections.remove(conn)

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
        
        # Option 1: Use documents from model.DOC_FOLDER (CFA docs)
        if use_uploaded_docs:
            from model_training import get_uploaded_documents
            documents = get_uploaded_documents()
            if not documents:
                raise HTTPException(status_code=400, detail="No documents found in upload folder. Please upload documents first.")
        
        # Option 2: Process newly uploaded files
        elif files:
            # Save uploaded files temporarily
            temp_dir = "temp_training_docs"
            os.makedirs(temp_dir, exist_ok=True)
            
            file_paths = []
            for file in files:
                # Sanitize filename: remove directory separators and invalid characters
                filename = os.path.basename(file.filename)
                # Replace any remaining path separators with underscores
                filename = filename.replace('/', '_').replace('\\', '_')
                # Remove any invalid Windows filename characters
                invalid_chars = '<>:"|?*'
                for char in invalid_chars:
                    filename = filename.replace(char, '_')
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                file_paths.append(file_path)
            
            # Process documents
            documents = process_documents(file_paths)
            
        if not documents:
            # Clean up temp files if they were created
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="No valid documents found")
        
        # Set up progress callback
        trainer.set_progress_callback(broadcast_training_progress)
        
        # Start training in background
        background_tasks.add_task(
            run_embedding_training,
            trainer,
            documents,
            temp_dir,  # Might be None if using uploaded docs
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
        # Clean up temp files on error
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
        # Start training
        result = await trainer.train_embedding_model(
            documents=documents,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Broadcast completion
        await broadcast_training_progress({
            'type': 'training_complete',
            'result': result
        })
        
    except Exception as e:
        # Broadcast error
        await broadcast_training_progress({
            'type': 'training_error',
            'error': str(e)
        })
    finally:
        # Clean up temp files if they exist
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

@app.post("/api/train/stop")
async def stop_training():
    """Stop current training"""
    trainer = get_trainer()
    
    if not trainer.is_model_training():
        raise HTTPException(status_code=400, detail="No training in progress")
    
    trainer.stop_training()
    
    # Broadcast stop message
    await broadcast_training_progress({
        'type': 'training_stopped',
        'message': 'Training stopped by user'
    })
    
    return {"message": "Training stopped successfully"}

@app.get("/api/train/logs")
async def get_training_logs():
    """Get training logs"""
    trainer = get_trainer()
    
    return {
        "logs": trainer.get_training_progress()
    }

# Add document upload endpoint for training
@app.post("/api/documents/upload")
async def upload_training_documents(files: List[UploadFile] = File(...)):
    """Upload documents for training"""
    
    uploaded_files = []
    
    try:
        # Create upload directory if it doesn't exist
        upload_dir = "uploaded_docs"
        os.makedirs(upload_dir, exist_ok=True)
        
        for file in files:
            # Sanitize filename: remove directory separators and invalid characters
            filename = os.path.basename(file.filename)
            # Replace any remaining path separators with underscores
            filename = filename.replace('/', '_').replace('\\', '_')
            # Remove any invalid Windows filename characters
            invalid_chars = '<>:"|?*'
            for char in invalid_chars:
                filename = filename.replace(char, '_')
            file_path = os.path.join(upload_dir, filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Get file info
            stat = os.stat(file_path)
            uploaded_files.append({
                "name": file.filename,
                "path": file_path,
                "size": stat.st_size,
                "type": file.content_type or "application/octet-stream"
            })
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {str(e)}")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "capabilities": {
            "sentence_transformers": True,
            "pylate": True,
            "embedding_training": True
        }
    }

@app.post("/api/train/cfa")
async def train_cfa_embeddings(
    background_tasks: BackgroundTasks,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    """
    Train embedding models directly on CFA documents from the configured document folder.
    This endpoint uses the documents in model.DOC_FOLDER without requiring re-upload.
    """
    
    # Check if already training
    trainer = get_trainer()
    if trainer.is_model_training():
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    try:
        # Initialize the CFA controller
        cfa_controller = CFAEmbeddingController()
        
        # Set progress callback to use the same broadcast function
        cfa_controller.set_progress_callback(broadcast_training_progress)
        
        # Check if documents are available
        documents = cfa_controller.get_cfa_documents()
        if not documents:
            raise HTTPException(
                status_code=400, 
                detail=f"No CFA documents found in {model.DOC_FOLDER}. Please ensure documents are present."
            )
            
        # Start training in background
        background_tasks.add_task(
            run_cfa_training,
            cfa_controller,
            epochs,
            batch_size,
            learning_rate
        )
        
        return {
            "status": "started",
            "message": f"Started CFA embedding training with {len(documents)} documents",
            "document_count": len(documents),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting CFA training: {str(e)}"
        )

async def run_cfa_training(
    cfa_controller: CFAEmbeddingController,
    epochs: int,
    batch_size: int,
    learning_rate: float
):
    """Run CFA embedding training in the background"""
    global training_status
    
    try:
        # Start training with CFA documents
        result = await cfa_controller.train_cfa_embeddings(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        if result["status"] == "completed":
            # Training completed successfully
            training_status["is_training"] = False
            training_status["status_message"] = "CFA embedding training completed successfully"
            training_status["progress"] = 100
            training_status["completed_steps"].append("cfa_embedding_training")
            
            # Broadcast completion message to all connections
            for connection in training_connections:
                try:
                    await connection.send_json({
                        "type": "training_complete",
                        "message": "CFA embedding training completed successfully",
                        "output_dir": result.get("model_path", ""),
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"Error sending completion message: {e}")
        else:
            # Training failed
            training_status["is_training"] = False
            training_status["status_message"] = f"CFA embedding training failed: {result.get('message', '')}"
            training_status["errors"].append(result.get("message", "Unknown error"))
            
            # Broadcast error message
            for connection in training_connections:
                try:
                    await connection.send_json({
                        "type": "training_error",
                        "error": result.get("message", "Training failed"),
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"Error sending error message: {e}")
    
    except Exception as e:
        # Handle unexpected errors
        training_status["is_training"] = False
        training_status["status_message"] = f"CFA embedding training error: {str(e)}"
        training_status["errors"].append(str(e))
        
        # Broadcast error message
        for connection in training_connections:
            try:
                await connection.send_json({
                    "type": "training_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Error sending error message: {e}")
                
    finally:
        print(f"CFA training task completed with status: {training_status['status_message']}")

@app.get("/api/cfa-documents")
async def get_cfa_documents():
    """Get list of available CFA documents"""
    try:
        from model_training import get_uploaded_documents
        documents = get_uploaded_documents()
        
        # Also get document file names
        doc_files = []
        if os.path.exists(model.DOC_FOLDER):
            for root, _, files in os.walk(model.DOC_FOLDER):
                for filename in files:
                    if filename.lower().endswith(('.pdf', '.txt', '.md', '.docx')):
                        rel_path = os.path.relpath(os.path.join(root, filename), model.DOC_FOLDER)
                        doc_files.append(rel_path)
        
        return {
            "status": "success",
            "document_count": len(documents),
            "document_files": doc_files,
            "doc_folder": model.DOC_FOLDER,
            "message": f"Found {len(documents)} CFA documents"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting CFA documents: {str(e)}")

@app.post("/api/train/use-cfa-docs")
async def train_with_cfa_documents(
    background_tasks: BackgroundTasks,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """Train embedding model using the uploaded CFA documents"""
    
    # Check if already training
    trainer = get_trainer()
    if trainer.is_model_training():
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    try:
        # Get CFA documents
        from model_training import get_uploaded_documents
        documents = get_uploaded_documents()
        if not documents:
            raise HTTPException(status_code=400, detail="No CFA documents found. Please upload documents first.")
        
        # Set up progress callback
        trainer.set_progress_callback(broadcast_training_progress)
        
        # Start training in background
        background_tasks.add_task(
            run_embedding_training,
            trainer,
            documents,
            None,  # No temp dir since using existing docs
            epochs,
            batch_size,
            learning_rate
        )
        
        return {
            "status": "started", 
            "message": f"Training started with {len(documents)} CFA documents",
            "document_count": len(documents),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")
