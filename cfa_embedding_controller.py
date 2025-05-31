"""
CFA Embedding Controller

This module connects the embedding model trainer to the CFA documents uploaded to the system.
It allows training of embedding models on the CFA documents without requiring re-upload.

Key features:
1. Direct access to CFA documents in the specified folder
2. Integration with existing WebSocket progress reporting
3. Training configuration specific to financial document embeddings
4. Automatic model selection and fallback
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import model training modules
from model_training import get_trainer, EmbeddingModelTrainer, process_documents, get_uploaded_documents
import model

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
CFA_DOC_FOLDER = model.DOC_FOLDER
CFA_MODEL_OUTPUT = "./trained_models/cfa_models"

class CFAEmbeddingController:
    """Controller for CFA document embedding training"""
    
    def __init__(self, 
                doc_folder: str = CFA_DOC_FOLDER, 
                output_dir: str = CFA_MODEL_OUTPUT,
                progress_callback=None):
        self.doc_folder = doc_folder
        self.output_dir = output_dir
        self.trainer = get_trainer()
        self.progress_callback = progress_callback
        
    def set_progress_callback(self, callback):
        """Set callback for training progress updates"""
        self.progress_callback = callback
        self.trainer.set_progress_callback(callback)
    
    def get_cfa_documents(self) -> List[str]:
        """Get all CFA documents from the document folder"""
        logger.info(f"Loading CFA documents from {self.doc_folder}")
        return get_uploaded_documents()
        
    async def train_cfa_embeddings(self, 
                               epochs: int = 3, 
                               batch_size: int = 16,
                               learning_rate: float = 2e-5) -> Dict[str, Any]:
        """Train embedding model on CFA documents"""
        
        # Get CFA documents
        documents = self.get_cfa_documents()
        if not documents:
            logger.error(f"No CFA documents found in {self.doc_folder}")
            return {
                "status": "error",
                "message": f"No CFA documents found in {self.doc_folder}",
                "timestamp": datetime.now().isoformat()
            }
            
        logger.info(f"Found {len(documents)} CFA documents for training")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Train embedding model
        logger.info("Starting CFA embedding model training")
        try:
            result = await self.trainer.train_embedding_model(
                documents=documents,
                output_dir=self.output_dir,
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Save training metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "num_documents": len(documents),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "status": "completed"
            }
            
            with open(os.path.join(self.output_dir, "training_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"CFA embedding training completed and saved to {self.output_dir}")
            return {
                "status": "completed",
                "message": "CFA embedding training completed successfully",
                "model_path": self.output_dir,
                "num_documents": len(documents),
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"CFA embedding training failed: {e}")
            return {
                "status": "error",
                "message": f"CFA embedding training failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

# Command-line interface for direct execution
async def main():
    controller = CFAEmbeddingController()
    print(f"Starting CFA embedding training with {len(controller.get_cfa_documents())} documents")
    
    def print_progress(progress):
        """Simple console progress callback"""
        if isinstance(progress, dict) and 'epoch' in progress and 'train_loss' in progress:
            print(f"Epoch {progress['epoch']}, Step {progress['step']}, Loss: {progress['train_loss']:.4f}")
    
    controller.set_progress_callback(print_progress)
    result = await controller.train_cfa_embeddings()
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())