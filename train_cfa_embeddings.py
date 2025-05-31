"""
Standalone script to train embeddings on CFA documents.
This avoids issues with the FastAPI reloading mechanism.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any
import model
from model_training import get_trainer, get_uploaded_documents

async def train_cfa_embeddings(
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    output_dir: str = "./trained_models/cfa_models"
):
    """Train embeddings on CFA documents"""
    print(f"Starting training with documents from: {model.DOC_FOLDER}")
    
    # Get documents
    documents = get_uploaded_documents()
    if not documents:
        print(f"No documents found in {model.DOC_FOLDER}")
        print("Please ensure CFA documents are uploaded and accessible")
        return

    print(f"Loaded {len(documents)} documents. Starting training...")
    
    # Initialize trainer
    trainer = get_trainer()
    
    # Define progress callback
    def progress_callback(progress_data: Dict[str, Any]):
        """Print progress updates"""
        if isinstance(progress_data, dict):
            epoch = progress_data.get('epoch', 0)
            step = progress_data.get('step', 0)
            loss = progress_data.get('train_loss', 0.0)
            print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
    
    # Set callback
    trainer.set_progress_callback(progress_callback)
    
    try:
        # Start training
        result = await trainer.train_embedding_model(
            documents=documents,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir
        )
        
        print("\n====== Training Complete ======")
        print(f"Status: {result.get('status')}")
        print(f"Final loss: {result.get('final_loss', 0.0):.4f}")
        print(f"Output directory: {result.get('output_dir')}")
        print(f"Total epochs: {result.get('total_epochs')}")
        print(f"Total steps: {result.get('total_steps')}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        
if __name__ == "__main__":
    print("CFA Document Embedding Training")
    print("="*40)
    
    # Get parameters from command line if provided
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
    
    print(f"Using parameters: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
    
    # Create output directory if it doesn't exist
    output_dir = "./trained_models/cfa_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run training
    asyncio.run(train_cfa_embeddings(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir
    ))
