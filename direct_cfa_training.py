"""
Direct CFA Training Script
This script directly trains embeddings on your CFA documents without relying on the 
server's background tasks or WebSocket communications.
"""

import os
import sys
import asyncio
from typing import List
from PyPDF2 import PdfReader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Document folder
DOC_FOLDER = r"C:\Users\robbi\OneDrive\CFA"

def extract_docs_from_folder() -> List[str]:
    """Extract text from PDF files in the document folder."""
    documents = []
    
    if not os.path.exists(DOC_FOLDER):
        logger.error(f"Document folder {DOC_FOLDER} does not exist")
        return []
    
    for root, dirs, files in os.walk(DOC_FOLDER):
        for fname in files:
            if fname.lower().endswith('.pdf'):
                try:
                    file_path = os.path.join(root, fname)
                    reader = PdfReader(file_path)
                    text_pages = [page.extract_text() or "" for page in reader.pages]
                    documents.append("\n".join(text_pages))
                    logger.info(f"Processed {fname}")
                except Exception as e:
                    logger.error(f"Error processing {fname}: {e}")
    
    return documents

async def train_with_docs():
    """Train model with extracted documents."""
    # Extract documents
    documents = extract_docs_from_folder()
    
    if not documents:
        logger.error("No documents extracted. Please check the document folder.")
        return
    
    logger.info(f"Extracted {len(documents)} documents")
    
    # Configure output directory
    output_dir = "./trained_models/cfa_models"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import here to avoid errors during import time
        from sentence_transformers import SentenceTransformer, InputExample, losses
        import torch
        import numpy as np
        
        # Create model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Model loaded: {model}")
        
        # Create training examples
        training_examples = []
        for i, doc in enumerate(documents):
            # Simple query generation
            sentences = doc.split('. ')
            query = sentences[0][:100] if sentences else doc[:50]
            
            # Create positive pair
            training_examples.append(InputExample(texts=[query, doc], label=1.0))
            
            # Create negative pairs
            for j in range(min(2, len(documents))):
                if j != i:
                    training_examples.append(InputExample(texts=[query, documents[j]], label=0.0))
        
        # Configure training
        train_dataloader = torch.utils.data.DataLoader(
            training_examples, shuffle=True, batch_size=16
        )
        
        train_loss = losses.CosineSimilarityLoss(model=model)
        
        # Set up training loop
        epochs = 3
        warmup_steps = int(len(train_dataloader) * 0.1)
        
        # Train
        logger.info(f"Starting training with {len(training_examples)} examples")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_dir
        )
        
        logger.info(f"Training complete! Model saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting CFA document training")
    asyncio.run(train_with_docs())
