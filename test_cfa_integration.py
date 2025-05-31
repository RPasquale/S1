"""
Test CFA Model Integration

This script tests the trained CFA model integration with the main application.
"""

import os
import sys

def test_model_loading():
    """Test if the CFA trained model can be loaded"""
    print("=== Testing CFA Model Loading ===")
    
    try:
        import model
        
        # Initialize the model (should load CFA model if available)
        embedding_model = model.initialize_colbert_model()
        print(f"✅ Model loaded successfully: {type(embedding_model)}")
        
        # Check if it's our trained model
        cfa_model_path = "./trained_models/cfa_models"
        if os.path.exists(cfa_model_path):
            print(f"✅ CFA model directory exists: {cfa_model_path}")
            
            # List model files
            model_files = os.listdir(cfa_model_path)
            print(f"📁 Model files: {model_files[:5]}..." if len(model_files) > 5 else f"📁 Model files: {model_files}")
        else:
            print(f"❌ CFA model directory not found: {cfa_model_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_cfa_query():
    """Test querying with CFA-related content"""
    print("\n=== Testing CFA Query ===")
    
    try:
        import model
        
        # Test query about CFA topics
        test_query = "What are the key topics for the CFA Level 2 exam?"
        print(f"🔍 Query: {test_query}")
        
        # Try to answer the question
        answer = model.answer_question_with_docs(test_query, top_k=3)
        print(f"💡 Answer: {answer[:200]}..." if len(answer) > 200 else f"💡 Answer: {answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during query: {e}")
        return False

def test_document_retrieval():
    """Test document retrieval functionality"""
    print("\n=== Testing Document Retrieval ===")
    
    try:
        import model
        
        # Initialize components
        model.initialize_dspy_model()
        model.initialize_colbert_model()
        model.initialize_index()
        model.initialize_retriever()
        
        print(f"✅ All components initialized")
        print(f"📚 Document count: {len(model.doc_texts)}")
        
        if model.doc_texts:
            print(f"📄 Sample documents: {list(model.doc_texts.keys())[:3]}")
        else:
            print("ℹ️  No documents loaded in index")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing retrieval: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🧪 Testing CFA Model Integration\n")
    
    # Test 1: Model loading
    model_ok = test_model_loading()
    
    # Test 2: Document retrieval
    retrieval_ok = test_document_retrieval() 
    
    # Test 3: CFA query
    if model_ok and retrieval_ok:
        query_ok = test_cfa_query()
    else:
        print("\n⏭️  Skipping query test due to previous failures")
        query_ok = False
    
    # Summary
    print(f"\n🏁 Integration Test Summary:")
    print(f"   Model Loading: {'✅' if model_ok else '❌'}")
    print(f"   Document Retrieval: {'✅' if retrieval_ok else '❌'}")
    print(f"   CFA Query: {'✅' if query_ok else '❌'}")
    
    if model_ok and retrieval_ok and query_ok:
        print("\n🎉 All tests passed! CFA integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
