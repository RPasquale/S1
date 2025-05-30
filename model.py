import dspy
import os
import sys
from PyPDF2 import PdfReader
from pylate import indexes, models, retrieve
from dspy import ChainOfThought

# Global constants
DOC_FOLDER = r"C:\Users\Admin\OneDrive\CFAL2"
INDEX_FOLDER = "pylate-index"

# Global variables to store initialized components
lm = None
model = None
index = None
retriever = None
doc_texts = {}
documents_ids = []
documents = []

def initialize_dspy_model():
    """Initialize the DSPy language model"""
    global lm
    if lm is None:
        lm = dspy.LM('ollama_chat/deepseek-r1:1.5b', api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=lm)
    return lm

def initialize_colbert_model():
    """Initialize the ColBERT embedding model"""
    global model
    if model is None:
        pylate_model_id = "lightonai/Reason-ModernColBERT"
        model = models.ColBERT(model_name_or_path=pylate_model_id)
    return model

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
        initialize_colbert_model()  # Ensure model is loaded
        
        documents_ids = []
        documents = []
        
        if os.path.exists(DOC_FOLDER):
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
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")

            if documents:
                documents_embeddings = model.encode(
                    documents,
                    batch_size=32,
                    is_query=False,
                    show_progress_bar=True,
                )

                index.add_documents(
                    documents_ids=documents_ids,
                    documents_embeddings=documents_embeddings,
                )
        else:
            print(f"Warning: Document folder {DOC_FOLDER} not found")

    # Load existing index if it exists
    if index_exists:
        index = indexes.Voyager(
            index_folder=INDEX_FOLDER,
            index_name="index",
        )

    # Build document mapping if we have documents
    if documents_ids and documents:
        if len(documents_ids) != len(documents):
            print("Document ID/text mismatch")
        else:
            doc_texts = dict(zip(documents_ids, documents))

    return index

def initialize_retriever():
    """Initialize the ColBERT retriever"""
    global retriever
    if retriever is None:
        initialize_index()  # Ensure index is loaded
        retriever = retrieve.ColBERT(index=index)
    return retriever

def test_dspy_model():
    """Test the DSPy model with a sample question"""
    initialize_dspy_model()
    cot = dspy.ChainOfThought('question -> response')
    response = cot(question="what are the key topics for the cfa level 2 exam?")
    return response.response

def answer_question_with_docs(query: str, top_k: int = 3) -> str:
    """Answer a question using retrieved documents and LLM"""
    initialize_dspy_model()
    initialize_colbert_model()
    initialize_retriever()
    
    if not doc_texts:
        # Fallback to simple DSPy response if no documents available
        cot = ChainOfThought('question -> response')
        res = cot(question=query)
        return res.response
    
    # Encode and retrieve
    query_emb = model.encode([query], batch_size=32, is_query=True)
    raw_results = retriever.retrieve(queries_embeddings=query_emb, k=top_k)[0]
    
    # Collect top-k document texts
    selected = [doc_texts[res['id']] for res in raw_results if res['id'] in doc_texts]

    if selected:
        context = "\n\n".join(f"Document {i+1}:\n{text}" for i, text in enumerate(selected))
        prompt = f"Use the following documents to answer the question:\n{context}\n\nQuestion: {query}\nAnswer:"
    else:
        prompt = query
        
    cot = ChainOfThought('question -> response')
    res = cot(question=prompt)
    return res.response

# Main interactive loop
if __name__ == "__main__":
    print("Interactive QA over indexed PDFs. Blank input to exit.")
    try:
        while True:
            q = input("Ask a question: ")
            if not q:
                break
            answer = answer_question_with_docs(q, top_k=3)
            print("\nAnswer:\n", answer)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
