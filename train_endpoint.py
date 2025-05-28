"""
Training endpoint module for the PDF QA Chatbot system.

This script provides a simple command-line interface to trigger model training
as well as a set of utilities for integration with the FastAPI server.
"""

import os
import sys
import argparse
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from model_training import ModelTrainer, load_text_from_files, get_document_metadata

def train_from_folder(folder_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Run the complete training pipeline on all documents in a folder."""
    start_time = time.time()
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        return {
            "status": "error",
            "message": f"Folder not found: {folder_path}",
            "timestamp": datetime.now().isoformat()
        }
    
    # Find PDF files in the folder
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        return {
            "status": "error",
            "message": f"No PDF files found in {folder_path}",
            "timestamp": datetime.now().isoformat()
        }
    
    # Load documents from files
    print(f"Loading {len(pdf_files)} PDF files...")
    documents = load_text_from_files(pdf_files)
    
    if not documents:
        return {
            "status": "error",
            "message": "Failed to extract text from PDF files",
            "timestamp": datetime.now().isoformat()
        }
    
    # Get document metadata
    doc_metadata = get_document_metadata(documents)
    
    # Initialize trainer
    trainer = ModelTrainer(output_dir=output_path or os.path.join(os.getcwd(), "trained-models"))
    
    # Extract queries
    print("Generating training queries...")
    queries = trainer._extract_sample_queries(documents)
    
    # Train embedding model
    print("Training embedding model...")
    embedding_model_dir = trainer.train_embedding_model(documents, queries)
    
    # Augment DSPy pipeline
    print("Optimizing DSPy pipeline...")
    dspy_dir = trainer.augment_dspy_pipeline(documents=documents)
    
    # Create training summary
    training_summary = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": time.time() - start_time,
        "documents": {
            "count": len(documents),
            "total_words": sum(meta["word_count"] for meta in doc_metadata),
            "metadata": doc_metadata[:5]  # Include metadata for first 5 docs
        },
        "queries": {
            "count": len(queries),
            "samples": queries[:5]  # Include first 5 queries
        },
        "models": {
            "embedding_model": embedding_model_dir,
            "dspy_modules": dspy_dir
        }
    }
    
    # Save summary
    summary_path = os.path.join(trainer.output_dir, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"Training complete. Summary saved to {summary_path}")
    return training_summary

def main():
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(description="Train models on PDF documents")
    parser.add_argument("folder", help="Folder containing PDF documents")
    parser.add_argument("--output", help="Output directory for trained models")
    args = parser.parse_args()
    
    result = train_from_folder(args.folder, args.output)
    
    if result["status"] == "success":
        print("Training completed successfully!")
        print(f"Processed {result['documents']['count']} documents")
        print(f"Generated {result['queries']['count']} training queries")
    else:
        print(f"Training failed: {result['message']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
