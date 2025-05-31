"""
Test script for CFA Embedding Controller

This script tests the integration between the CFA documents and embedding training.
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Import the embedding controller
from cfa_embedding_controller import CFAEmbeddingController

async def test_cfa_controller():
    """Test the CFA embedding controller functionality"""
    
    print("=== CFA Embedding Controller Test ===")
    
    # Initialize controller
    controller = CFAEmbeddingController()
    
    # 1. Test document loading
    print("\nStep 1: Testing document loading...")
    documents = controller.get_cfa_documents()
    print(f"Found {len(documents)} documents")
    if documents:
        print(f"First document preview: {documents[0][:150]}...")
    else:
        print("No documents found - check your CFA document folder path")
        return
    
    # 2. Test with minimal training (1 epoch)
    print("\nStep 2: Testing embedding training with 1 epoch...")
    
    def progress_callback(progress):
        """Progress tracking callback"""
        if isinstance(progress, dict) and 'epoch' in progress and 'train_loss' in progress:
            print(f"  Epoch {progress['epoch']}, Step {progress['step']}, Loss: {progress['train_loss']:.4f}")
    
    # Set progress callback
    controller.set_progress_callback(progress_callback)
    
    # Start training with only 1 epoch for testing
    result = await controller.train_cfa_embeddings(epochs=1, batch_size=8)
    
    # Print result
    print("\nTraining result:")
    print(json.dumps(result, indent=2))
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_cfa_controller())
