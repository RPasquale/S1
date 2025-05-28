from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, importlib, asyncio
from typing import List, Dict, Any, Optional
import model
from model_training import ModelTrainer
import json
from datetime import datetime

# Initialize app
app = FastAPI(title="PDF QA Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

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
    
    for root, dirs, files in os.walk(model.doc_folder):
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
        
        training_status["progress"] = 10
        training_status["completed_steps"].append("document_extraction")
        
        # Initialize model trainer
        trainer = ModelTrainer()
        
        # Step 1: Extract high-quality queries from documents
        training_status["status_message"] = "Generating training queries..."
        queries = trainer._extract_sample_queries(documents)
        training_status["progress"] = 25
        training_status["completed_steps"].append("query_generation")
        
        # Step 2: Train embedding model
        training_status["status_message"] = "Training embedding model..."
        embedding_model_dir = trainer.train_embedding_model(documents, queries)
        training_status["progress"] = 60
        training_status["completed_steps"].append("embedding_model")
        
        # Step 3: Prepare dataset for language model fine-tuning
        training_status["status_message"] = "Preparing training dataset..."
        dataset = trainer.prepare_dataset_from_documents(documents)
        training_status["progress"] = 70
        training_status["completed_steps"].append("dataset_preparation")
        
        # Step 4: Augment DSPy pipeline
        training_status["status_message"] = "Optimizing DSPy pipeline..."
        dspy_modules_dir = trainer.augment_dspy_pipeline(documents=documents)
        training_status["progress"] = 90
        training_status["completed_steps"].append("dspy_optimization")
        
        # Save training result summary
        output_dir = os.path.join(trainer.output_dir, "training-summary")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "num_documents": len(documents),
                "num_queries": len(queries),
                "modules_trained": training_status["completed_steps"],
                "embedding_model_dir": embedding_model_dir,
                "dspy_modules_dir": dspy_modules_dir,
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

@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload PDF files and build search index."""
    # Save uploaded PDFs into the doc_folder, preserving subpaths
    upload_count = 0
    for f in files:
        try:
            dest = os.path.join(model.doc_folder, f.filename)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as out:
                out.write(await f.read())
            upload_count += 1
        except Exception as e:
            return {"status": "error", "message": f"Error saving {f.filename}: {str(e)}"}
    
    # Remove existing index to force rebuild
    if os.path.exists(model.index_folder):
        shutil.rmtree(model.index_folder)
    
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

@app.post("/chat")
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

@app.get("/conversations")
async def get_conversations():
    """Get list of all conversation IDs."""
    return {"conversation_ids": list(conversations.keys())}

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get messages from a specific conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"messages": conversations[conversation_id]}

@app.post("/conversations")
async def create_conversation(payload: dict):
    """Create a new conversation."""
    conversation_id = payload.get("conversation_id", f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if conversation_id in conversations:
        raise HTTPException(status_code=400, detail="Conversation ID already exists")
    
    conversations[conversation_id] = []
    return {"conversation_id": conversation_id}

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[conversation_id]
    return {"status": "deleted", "conversation_id": conversation_id}

@app.get("/training/status")
async def get_training_status():
    """Get current model training status."""
    return training_status

@app.post("/training/start")
async def start_training(background_tasks: BackgroundTasks):
    """Manually start model training."""
    global training_status
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    background_tasks.add_task(train_models_background, background_tasks)
    return {"status": "started"}

@app.post("/training/cancel")
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
@app.post("/dspy/functions")
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

@app.get("/dspy/functions")
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

@app.delete("/dspy/functions/{function_name}")
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
@app.get("/data-extraction/status")
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

@app.post("/data-extraction/start")
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
@app.post("/training/advanced")
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

@app.get("/training/capabilities")
async def get_training_capabilities():
    """Get available training capabilities."""
    try:
        from model_training import get_training_capabilities
        return get_training_capabilities()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training capabilities: {str(e)}")
