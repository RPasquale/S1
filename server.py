from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, importlib, asyncio, signal, sys
from typing import List, Dict, Any, Optional
import model
from model_training import ModelTrainer
import json
from datetime import datetime
from contextlib import asynccontextmanager
from dspy_generator_system import (
    dspy_registry, register_signature, create_generator, 
    create_chain, get_trace, save_trace
)
import dspy
from typing import Type

# Global state for cleanup
training_task = None
background_tasks_tracker = []

# Async context manager for app lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    print("Application starting up...")
    # Startup code can go here if needed
    
    yield  # This is where the application runs
    
    # Shutdown cleanup
    print("Application shutting down...")
    global training_task, background_tasks_tracker
    
    # Cancel any running training task
    if training_task and not training_task.done():
        print("Cancelling training task...")
        training_task.cancel()
        try:
            await training_task
        except asyncio.CancelledError:
            print("Training task cancelled successfully")
    
    # Cancel any other background tasks
    for task in background_tasks_tracker:
        if not task.done():
            task.cancel()
    
    print("Cleanup complete")

# Initialize app with lifespan handler
app = FastAPI(title="PDF QA Chatbot API", lifespan=lifespan)
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

async def train_models_background():
    """Run model training in background to avoid blocking the API."""
    global training_status, training_task
    
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
        
        # Check for cancellation
        if training_status.get("cancelled", False):
            return
        
        # Initialize model trainer
        trainer = ModelTrainer()
        
        # Step 1: Extract high-quality queries from documents
        training_status["status_message"] = "Generating training queries..."
        queries = trainer._extract_sample_queries(documents)
        training_status["progress"] = 25
        training_status["completed_steps"].append("query_generation")
        
        # Check for cancellation
        if training_status.get("cancelled", False):
            return
        
        # Step 2: Train embedding model
        training_status["status_message"] = "Training embedding model..."
        embedding_model_dir = trainer.train_embedding_model(documents, queries)
        training_status["progress"] = 60
        training_status["completed_steps"].append("embedding_model")
        
        # Check for cancellation
        if training_status.get("cancelled", False):
            return
        
        # Step 3: Prepare dataset for language model fine-tuning
        training_status["status_message"] = "Preparing training dataset..."
        dataset = trainer.prepare_dataset_from_documents(documents)
        training_status["progress"] = 70
        training_status["completed_steps"].append("dataset_preparation")
        
        # Check for cancellation
        if training_status.get("cancelled", False):
            return
        
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
        
    except asyncio.CancelledError:
        training_status["status_message"] = "Training cancelled"
        training_status["cancelled"] = True
        print("Training cancelled gracefully")
    except Exception as e:
        training_status["status_message"] = f"Error during training: {str(e)}"
        training_status["errors"].append(str(e))
        print(f"Training error: {e}")
    
    finally:
        training_status["is_training"] = False
        training_task = None

@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload PDF files and build search index."""
    global training_task
    
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
        training_task = asyncio.create_task(train_models_background())
        background_tasks_tracker.append(training_task)
    
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
    global training_status
    return training_status

@app.post("/training/start")
async def start_training():
    """Manually start model training."""
    global training_status, training_task
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    training_task = asyncio.create_task(train_models_background())
    background_tasks_tracker.append(training_task)
    return {"status": "started"}

@app.post("/training/cancel")
async def cancel_training():
    """Cancel ongoing model training."""
    global training_status, training_task
    
    if not training_status["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    # Cancel the training task properly
    if training_task and not training_task.done():
        training_task.cancel()
        training_status["cancelled"] = True
        training_status["status_message"] = "Training cancellation requested..."
        
        try:
            await training_task
        except asyncio.CancelledError:
            pass
    
    training_status["is_training"] = False
    training_status["status_message"] = "Training cancelled by user"
    return {"status": "cancelled"}

@app.post("/generate")
async def generate_dspy_chain(payload: dict):
    """Create a custom DSPy generator."""
    try:
        # Extract parameters from payload
        generator_name = payload.get("name")
        description = payload.get("description", "")
        input_vars = payload.get("input_vars", [])
        output_vars = payload.get("output_vars", [])
        steps = payload.get("steps", [])
        tags = payload.get("tags", [])
        
        # Validate input
        if not generator_name:
            raise ValueError("Generator name is required")
        
        # Register the DSPy generator signature
        register_signature(
            name=generator_name,
            input_vars=input_vars,
            output_vars=output_vars,
            description=description,
            tags=tags
        )
        
        # Create the generator function
        exec_locals = {}
        exec(f"def {generator_name}({', '.join(input_vars)}): pass", {}, exec_locals)
        generator_func = exec_locals[generator_name]
        
        # Create the DSPy generator
        dspy_generator = create_generator(
            func=generator_func,
            input_vars=input_vars,
            output_vars=output_vars,
            steps=steps,
            name=generator_name
        )
        
        return {"status": "generator_created", "generator_name": generator_name}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/trace")
async def trace_dspy_chain(payload: dict):
    """Trace a DSPy chain execution."""
    chain_id = payload.get("chain_id")
    inputs = payload.get("inputs", {})
    
    try:
        # Get the chain by ID
        chain = dspy_registry.get(chain_id)
        if not chain:
            raise ValueError(f"Chain ID {chain_id} not found")
        
        # Execute the chain with the given inputs
        output = chain.predict(**inputs)
        
        return {"status": "traced", "output": output}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/save")
async def save_dspy_trace(payload: dict):
    """Save a DSPy trace to file."""
    trace_id = payload.get("trace_id")
    file_path = payload.get("file_path")
    
    try:
        # Get the trace by ID
        trace = get_trace(trace_id)
        if not trace:
            raise ValueError(f"Trace ID {trace_id} not found")
        
        # Save the trace to the specified file
        save_trace(trace_id, file_path)
        
        return {"status": "trace_saved", "file_path": file_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dspy/signature")
async def register_dspy_signature(payload: dict):
    """Register a custom DSPy signature."""
    try:
        name = payload.get("name")
        description = payload.get("description", "")
        input_fields = payload.get("input_fields", {})  # {field_name: field_description}
        output_fields = payload.get("output_fields", {})  # {field_name: field_description}
        
        if not name or not input_fields or not output_fields:
            raise ValueError("Name, input_fields, and output_fields are required")
        
        # Dynamically create the signature class
        class_attrs = {"__doc__": description}
        
        # Add input fields
        for field_name, field_desc in input_fields.items():
            class_attrs[field_name] = dspy.InputField(desc=field_desc)
        
        # Add output fields
        for field_name, field_desc in output_fields.items():
            class_attrs[field_name] = dspy.OutputField(desc=field_desc)
        
        # Create the signature class
        signature_class = type(name, (dspy.Signature,), class_attrs)
        
        # Register it
        register_signature(name, signature_class)
        
        return {
            "status": "signature_registered", 
            "name": name,
            "input_fields": list(input_fields.keys()),
            "output_fields": list(output_fields.keys())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dspy/generator")
async def create_dspy_generator(payload: dict):
    """Create a custom DSPy generator."""
    try:
        name = payload.get("name")
        signature_name = payload.get("signature_name")
        generator_type = payload.get("generator_type", "ChainOfThought")
        
        if not name or not signature_name:
            raise ValueError("Name and signature_name are required")
        
        # Create the generator
        generator = create_generator(name, signature_name, generator_type)
        
        return {
            "status": "generator_created",
            "name": name,
            "signature_name": signature_name,
            "generator_type": generator_type,
            "shared_model": dspy_registry.shared_lm.model if dspy_registry.shared_lm else "not_configured"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dspy/execute")
async def execute_dspy_generator(payload: dict):
    """Execute a DSPy generator with given inputs."""
    try:
        generator_name = payload.get("generator_name")
        inputs = payload.get("inputs", {})
        
        if not generator_name:
            raise ValueError("Generator name is required")
        
        if generator_name not in dspy_registry.generators:
            raise ValueError(f"Generator '{generator_name}' not found")
        
        # Get the generator
        generator_info = dspy_registry.generators[generator_name]
        generator = generator_info["generator"]
        
        # Execute it
        result = generator(**inputs)
        
        # Extract result data
        output_data = {}
        if hasattr(result, '__dict__'):
            for key, value in result.__dict__.items():
                if not key.startswith('_'):
                    output_data[key] = value
        else:
            output_data = {"result": str(result)}
        
        return {
            "status": "execution_completed",
            "generator_name": generator_name,
            "inputs": inputs,
            "outputs": output_data,
            "shared_model": dspy_registry.shared_lm.model if dspy_registry.shared_lm else "not_configured"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dspy/chain")
async def create_dspy_chain(payload: dict):
    """Create a chain of DSPy generators."""
    try:
        chain_name = payload.get("chain_name")
        generator_sequence = payload.get("generator_sequence", [])
        
        if not chain_name or not generator_sequence:
            raise ValueError("Chain name and generator sequence are required")
        
        # Create the chain
        chain = create_chain(chain_name, generator_sequence)
        
        return {
            "status": "chain_created",
            "chain_name": chain_name,
            "steps": len(generator_sequence),
            "generators": [step["generator"] for step in generator_sequence]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dspy/chain/execute")
async def execute_dspy_chain(payload: dict):
    """Execute a DSPy generator chain."""
    try:
        chain_name = payload.get("chain_name")
        initial_input = payload.get("initial_input", {})
        
        if not chain_name:
            raise ValueError("Chain name is required")
        
        # Find the chain
        chain = None
        for c in dspy_registry.generator_chains:
            if c.name == chain_name:
                chain = c
                break
        
        if not chain:
            raise ValueError(f"Chain '{chain_name}' not found")
        
        # Execute the chain
        result = chain.execute(initial_input)
        
        return {
            "status": "chain_executed",
            "chain_name": chain_name,
            "initial_input": initial_input,
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dspy/trace")
async def get_dspy_reasoning_trace(format_type: str = "detailed"):
    """Get the reasoning trace of all DSPy generator executions."""
    try:
        trace = get_trace(format_type)
        return trace
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dspy/trace/save")
async def save_dspy_trace(payload: dict):
    """Save the reasoning trace to a file."""
    try:
        filepath = payload.get("filepath")
        if not filepath:
            # Generate default filepath
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"reasoning_trace_{timestamp}.json"
        
        save_trace(filepath)
        
        return {
            "status": "trace_saved",
            "filepath": filepath
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dspy/generators")
async def list_dspy_generators():
    """List all registered DSPy generators and signatures."""
    try:
        return {
            "generators": {
                name: {
                    "signature": info["signature"],
                    "type": info["type"],
                    "created_at": info["created_at"],
                    "call_count": info["call_count"]
                }
                for name, info in dspy_registry.generators.items()
            },
            "signatures": list(dspy_registry.signatures.keys()),
            "chains": [
                {
                    "name": chain.name,
                    "steps": len(chain.sequence),
                    "generators": [step["generator"] for step in chain.sequence]
                }
                for chain in dspy_registry.generator_chains
            ],
            "shared_model": dspy_registry.shared_lm.model if dspy_registry.shared_lm else "not_configured"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
