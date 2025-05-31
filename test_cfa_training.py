#!/usr/bin/env python3
"""
CFA Document Training Test Script

This script will:
1. Load all CFA documents from the specified folder
2. Process and extract text content
3. Generate training queries using the enhanced pipeline
4. Train embedding models on CFA content
5. Save trained models for production use

Usage:
    python test_cfa_training.py
"""

import os
import sys
import time
import glob
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_training import ModelTrainer

def main():
    print("🚀 Starting CFA Document Training Test")
    print("=" * 60)
    
    # Configuration
    CFA_FOLDER = r"C:\Users\robbi\OneDrive\CFA"
    MODELS_OUTPUT_DIR = "./trained_models/cfa_models"
    
    # Create output directory
    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    
    print(f"📁 CFA Documents Folder: {CFA_FOLDER}")
    print(f"💾 Models Output Directory: {MODELS_OUTPUT_DIR}")
    print()
    
    # Step 1: Check CFA folder and count documents
    print("Step 1: Checking CFA Documents...")
    if not os.path.exists(CFA_FOLDER):
        print(f"❌ ERROR: CFA folder not found at {CFA_FOLDER}")
        return False
    
    # Find all document files
    document_extensions = ['*.pdf', '*.txt', '*.docx', '*.md']
    all_documents = []
    
    for ext in document_extensions:
        pattern = os.path.join(CFA_FOLDER, '**', ext)
        files = glob.glob(pattern, recursive=True)
        all_documents.extend(files)
    
    print(f"✅ Found {len(all_documents)} documents in CFA folder")
    if len(all_documents) == 0:
        print("❌ No documents found. Please check the folder path.")
        return False
    
    # Display first few files
    print("📋 Sample documents found:")
    for i, doc in enumerate(all_documents[:5]):
        filename = os.path.basename(doc)
        size_mb = os.path.getsize(doc) / (1024 * 1024)
        print(f"   {i+1}. {filename} ({size_mb:.2f} MB)")
    
    if len(all_documents) > 5:
        print(f"   ... and {len(all_documents) - 5} more documents")
    print()
    
    # Step 2: Initialize ModelTrainer
    print("Step 2: Initializing ModelTrainer...")
    try:
        trainer = ModelTrainer()
        print("✅ ModelTrainer initialized successfully")
        print(f"   Device: {trainer.device}")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize ModelTrainer: {e}")
        return False
    
    # Step 3: Load and process documents
    print("Step 3: Loading CFA Documents...")
    start_time = time.time()
    
    try:
        # Use the load_documents_from_path method
        documents = trainer.load_documents_from_path(CFA_FOLDER)
        load_time = time.time() - start_time
        
        print(f"✅ Loaded {len(documents)} documents successfully")
        print(f"   Loading time: {load_time:.2f} seconds")
          # Display document statistics
        # Documents are returned as strings, not dictionaries
        total_chars = sum(len(doc) for doc in documents if isinstance(doc, str))
        avg_chars = total_chars / len(documents) if documents else 0
        
        print(f"   Total content: {total_chars:,} characters")
        print(f"   Average per document: {avg_chars:.0f} characters")
        print()
        
    except Exception as e:
        print(f"❌ Failed to load documents: {e}")
        return False
      # Step 4: Generate training queries
    print("Step 4: Generating Training Queries...")
    start_time = time.time()
    
    try:
        # Use enhanced query extraction
        queries = trainer._extract_sample_queries(
            documents, 
            num_samples=30  # Generate 30 queries total
        )
        query_time = time.time() - start_time
        
        print(f"✅ Generated {len(queries)} training queries")
        print(f"   Query generation time: {query_time:.2f} seconds")
        
        # Display sample queries
        print("📝 Sample queries generated:")
        for i, query in enumerate(queries[:3]):
            print(f"   {i+1}. {query[:100]}...")
        print()
        
    except Exception as e:
        print(f"❌ Failed to generate queries: {e}")
        print("⚠️  Continuing with document-based training...")
        queries = []
    
    # Step 5: Train embedding model
    print("Step 5: Training Embedding Model on CFA Content...")
    start_time = time.time()
    
    try:
        # Train with CFA documents
        training_result = trainer.train_embedding_model(
            documents=documents,
            queries=queries,
            output_dir=os.path.join(MODELS_OUTPUT_DIR, "cfa_embeddings"),
            num_epochs=3,  # Moderate training for testing
            batch_size=8,  # Reasonable batch size
            learning_rate=2e-5
        )
        
        training_time = time.time() - start_time
        
        print(f"✅ Embedding model training completed!")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Model saved to: {os.path.join(MODELS_OUTPUT_DIR, 'cfa_embeddings')}")
        
        # Display training metrics if available
        if isinstance(training_result, dict) and 'metrics' in training_result:
            metrics = training_result['metrics']
            print("📊 Training metrics:")
            for key, value in metrics.items():
                print(f"   {key}: {value}")
        print()
        
    except Exception as e:
        print(f"❌ Embedding model training failed: {e}")
        print("⚠️  This might be due to missing dependencies or hardware limitations")
        print()
    
    # Step 6: Train/Optimize DSPy pipeline
    print("Step 6: Setting up DSPy Pipeline for CFA Content...")
    start_time = time.time()
    
    try:
        # Augment DSPy pipeline with CFA-specific content
        dspy_result = trainer.augment_dspy_pipeline(
            documents=documents,
            queries=queries[:50] if queries else [],  # Use first 50 queries for pipeline training
            output_dir=os.path.join(MODELS_OUTPUT_DIR, "cfa_dspy_pipeline")
        )
        
        dspy_time = time.time() - start_time
        
        print(f"✅ DSPy pipeline setup completed!")
        print(f"   Setup time: {dspy_time:.2f} seconds")
        print(f"   Pipeline saved to: {os.path.join(MODELS_OUTPUT_DIR, 'cfa_dspy_pipeline')}")
        print()
        
    except Exception as e:
        print(f"❌ DSPy pipeline setup failed: {e}")
        print("⚠️  This might be due to missing DSPy dependencies or LLM connectivity")
        print()
    
    # Step 7: Summary and next steps
    total_time = time.time() - start_time
    print("🎉 CFA Training Test Summary")
    print("=" * 40)
    print(f"📁 Documents processed: {len(documents)}")
    print(f"📝 Queries generated: {len(queries)}")
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    print(f"💾 Models saved to: {MODELS_OUTPUT_DIR}")
    print()
    
    print("🚀 Next Steps:")
    print("1. Check the trained models in the output directory")
    print("2. Test the models with sample CFA questions")
    print("3. Integrate the trained models into your chatbot")
    print("4. Fine-tune parameters based on performance")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ CFA training test completed successfully!")
    else:
        print("❌ CFA training test failed. Check the errors above.")
    
    input("\nPress Enter to exit...")
