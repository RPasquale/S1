import os
import sys
from PyPDF2 import PdfReader
from pylate import indexes, models, retrieve
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Global constants
DOC_FOLDER = r"C:\Users\Admin\OneDrive\CFAL2" 
INDEX_FOLDER = "pylate-index"

# Global variables to store initialized components
embedding_model = None
language_model = None
tokenizer = None
index = None
retriever = None
doc_texts = {}
documents_ids = []
documents = []

def initialize_language_model():
    """Initialize a Hugging Face language model for text generation"""
    global language_model, tokenizer
    
    if language_model is None:
        try:
            # Use the specific DeepSeek model as requested
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            
            print(f"Loading language model: {model_name}")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Load tokenizer first with error handling
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False
            )
            print("✅ Tokenizer loaded successfully")
            
            # Load model
            print("Loading language model...")
            language_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            print("✅ Language model loaded successfully")
            
            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print("✅ Pad token set")
            
            print(f"✅ Language model loaded on {device}")
            
        except Exception as e:
            print(f"❌ Error loading language model: {e}")
            raise e
    
    return language_model, tokenizer

def initialize_embedding_model():
    """Initialize the embedding model"""
    global embedding_model
    
    if embedding_model is None:
        pylate_model_id = "lightonai/Reason-ModernColBERT"
        
        # Check if CUDA is available for embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on device: {device}")
        
        embedding_model = models.ColBERT(
            model_name_or_path=pylate_model_id,
            device=device
        )
        print(f"✅ Loaded PyLate embedding model: {pylate_model_id} on {device}")
    
    return embedding_model

def initialize_index():
    """Initialize or load the Voyager index"""
    global index, documents_ids, documents, doc_texts
    
    if index is not None:
        return index
        
    voyager_index_path = os.path.join(INDEX_FOLDER, "index.voyager")
    index_exists = os.path.exists(voyager_index_path)

    # Initialize or load the Voyager index
    if not index_exists:
        index = indexes.Voyager(
            index_folder=INDEX_FOLDER,
            index_name="index",
            override=True,  # build a new index only once
        )
    else:
        index = indexes.Voyager(
            index_folder=INDEX_FOLDER,
            index_name="index",
            override=False,  # reuse existing index
        )

    # If index did not exist, encode and add documents
    if not index_exists:
        initialize_embedding_model()  # Ensure model is loaded
        
        documents_ids = []
        documents = []
        
        if os.path.exists(DOC_FOLDER):
            print(f"📁 Scanning documents in {DOC_FOLDER}")
            for root, dirs, files in os.walk(DOC_FOLDER):
                for fname in files:
                    if fname.lower().endswith('.pdf'):
                        file_path = os.path.join(root, fname)
                        try:
                            reader = PdfReader(file_path)
                            text_pages = [page.extract_text() or "" for page in reader.pages]
                            documents.append("\n".join(text_pages))
                            rel_id = os.path.relpath(file_path, DOC_FOLDER)
                            documents_ids.append(rel_id)
                            print(f"📄 Processed: {rel_id}")
                        except Exception as e:
                            print(f"❌ Error reading {file_path}: {e}")

            if documents:
                print(f"🔍 Encoding {len(documents)} documents...")
                documents_embeddings = embedding_model.encode(
                    documents,
                    batch_size=32,
                    is_query=False,
                    show_progress_bar=True,
                )

                index.add_documents(
                    documents_ids=documents_ids,
                    documents_embeddings=documents_embeddings,
                )
                print("✅ Index created successfully")
            else:
                print("⚠️ No documents found to index")
        else:
            print(f"⚠️ Warning: Document folder {DOC_FOLDER} not found")

    # Load existing index if it exists
    if index_exists:
        index = indexes.Voyager(
            index_folder=INDEX_FOLDER,
            index_name="index",
        )
        print("✅ Existing index loaded")

    # Build document mapping if we have documents
    if documents_ids and documents:
        if len(documents_ids) != len(documents):
            print("⚠️ Document ID/text mismatch")
        else:
            doc_texts = dict(zip(documents_ids, documents))

    return index

def initialize_retriever():
    """Initialize the ColBERT retriever"""
    global retriever
    if retriever is None:
        initialize_index()  # Ensure index is loaded
        retriever = retrieve.ColBERT(index=index)
        print("✅ Retriever initialized")
    return retriever

def generate_response(prompt: str, max_new_tokens: int = 300) -> str:
    """Generate a response using the Hugging Face language model"""
    try:
        lm, tok = initialize_language_model()
        
        print(f"🤖 Generating response for prompt length: {len(prompt)} chars")
        
        # Using model directly (not pipeline)
        device = next(lm.parameters()).device
        
        # Properly tokenize with attention mask
        tokenized = tok(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1500  # Leave room for generation
        )
        
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        
        print(f"📝 Input tokens: {input_ids.shape[1]}, generating {max_new_tokens} new tokens")
        
        with torch.no_grad():
            outputs = lm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tok.eos_token_id,
                do_sample=True,
                repetition_penalty=1.1  # Prevent repetition
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][input_ids.shape[1]:]
        response = tok.decode(new_tokens, skip_special_tokens=True)
        
        print(f"✅ Generated response length: {len(response)} chars")
        return response.strip()
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return f"I apologize, but I encountered an error while generating a response: {str(e)}"

def answer_question_with_docs(query: str, top_k: int = 3) -> str:
    """Answer a question using retrieved documents and Hugging Face language model"""
    
    try:
        initialize_embedding_model()
        initialize_retriever()
        
        if not doc_texts:
            # Fallback to simple language model response if no documents available
            prompt = f"Question: {query}\nAnswer:"
            return generate_response(prompt)
          # Encode and retrieve
        query_emb = embedding_model.encode([query], batch_size=32, is_query=True)
        raw_results = retriever.retrieve(queries_embeddings=query_emb, k=top_k)[0]
          # Collect top-k document texts
        selected = [doc_texts[res['id']] for res in raw_results if res['id'] in doc_texts]
        
        print(f"🔍 Retrieved {len(selected)} documents for query: {query[:50]}...")
        for i, res in enumerate(raw_results[:3]):
            print(f"  Doc {i+1}: {res['id']} (score: {res.get('score', 'N/A')})")
        
        if selected:
            # Limit context length to avoid token limits - use smaller chunks
            contexts = []
            for i, text in enumerate(selected):
                # Take first 500 chars and look for a good break point
                chunk = text[:800]
                if len(text) > 800:
                    # Try to break at a sentence
                    last_period = chunk.rfind('.')
                    if last_period > 400:
                        chunk = chunk[:last_period + 1]
                contexts.append(f"Document {i+1} ({raw_results[i]['id']}):\n{chunk}")
            
            context = "\n\n".join(contexts)
            prompt = f"""Based on the following CFA documents, please provide a comprehensive answer:

{context}

Question: {query}
Answer:"""
        else:
            prompt = f"Question: {query}\nAnswer:"
            
        return generate_response(prompt, max_new_tokens=300)
        
    except Exception as e:
        print(f"❌ Error in answer_question_with_docs: {e}")
        return f"I apologize, but I encountered an error while processing your question: {str(e)}"

def initialize_models():
    """Initialize all models"""
    try:
        print("🤗 Initializing Hugging Face models...")
        initialize_embedding_model()
        initialize_language_model()
        initialize_index()
        initialize_retriever()
        print("✅ All models initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return False

def test_models() -> bool:
    """Test if models are working correctly"""
    try:
        # Test embedding model
        if embedding_model is None:
            initialize_embedding_model()
          # Test language model
        if language_model is None:
            initialize_language_model()
        
        # Quick test
        test_response = generate_response("Hello", max_new_tokens=50)
        return len(test_response) > 0
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def get_models_status() -> dict:
    """Get status of all models"""
    voyager_index_path = os.path.join(INDEX_FOLDER, "index.voyager")
    
    return {
        "embedding_model": {
            "loaded": embedding_model is not None,
            "type": "Custom CFA" if os.path.exists("./trained_models/cfa_models") else "PyLate ColBERT",
        },
        "language_model": {
            "loaded": language_model is not None,
            "type": "Hugging Face Transformers",
        },
        "index": {
            "exists": os.path.exists(voyager_index_path),
            "documents_count": len(documents_ids) if documents_ids else 0,
            "doc_texts_loaded": len(doc_texts) if doc_texts else 0,
        },
        "retriever": {
            "loaded": retriever is not None,
        },
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp": str(os.times())
    }

# Main interactive loop
if __name__ == "__main__":
    print("🤗 Interactive QA using Hugging Face models. Blank input to exit.")
    try:
        initialize_models()
        
        while True:
            q = input("\nAsk a question: ")
            if not q:
                break
            print("🤔 Thinking...")
            answer = answer_question_with_docs(q, top_k=3)
            print(f"\n✅ Answer:\n{answer}")
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Error: {e}")
