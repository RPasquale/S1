import dspy
lm = dspy.LM('ollama_chat/deepseek-r1:8b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

cot = dspy.ChainOfThought('question -> response')
response_1 = cot(question="what are the key topics for the cfa level 2 exam?")

print(response_1.response)

# Now load the embedding model
from pylate import indexes, models, retrieve
pylate_model_id = "lightonai/Reason-ModernColBERT"  # identifier for the ColBERT embedding model

# Step 1: Load the ColBERT model
model = models.ColBERT(
    model_name_or_path=pylate_model_id,
)

import os

# Determine if index already exists
index_folder = "pylate-index"
voyager_index_path = os.path.join(index_folder, "index.voyager")
index_exists = os.path.exists(voyager_index_path)

# Step 1: Initialize or load the Voyager index
from pylate import indexes
if not index_exists:
    index = indexes.Voyager(
        index_folder=index_folder,
        index_name="index",
        override=True,  # build a new index only once
    )
else:
    index = indexes.Voyager(
        index_folder=index_folder,
        index_name="index",
        override=False,  # reuse existing index
    )

# Step 2: If index did not exist, encode and add documents
if not index_exists:
    # Step 3: Encode the documents
    import os
    from PyPDF2 import PdfReader

    doc_folder = r"C:\Users\Admin\OneDrive\CFAL2"  # CFAL2 root containing subfolders of PDFs
    documents_ids = []
    documents = []
    for root, dirs, files in os.walk(doc_folder):
        for fname in files:
            if fname.lower().endswith('.pdf'):
                file_path = os.path.join(root, fname)
                reader = PdfReader(file_path)
                text_pages = [page.extract_text() or "" for page in reader.pages]
                documents.append("\n".join(text_pages))
                rel_id = os.path.relpath(file_path, doc_folder)
                documents_ids.append(rel_id)

    documents_embeddings = model.encode(
        documents,
        batch_size=32,
        is_query=False,  # Ensure that it is set to False to indicate that these are documents, not queries
        show_progress_bar=True,
    )

    # Step 4: Add document embeddings to the index by providing embeddings and corresponding ids
    index.add_documents(
        documents_ids=documents_ids,
        documents_embeddings=documents_embeddings,
    )

# To load an index, simply instantiate it with the correct folder/name and without overriding it
index = indexes.Voyager(
    index_folder="pylate-index",
    index_name="index",
)

# Build a mapping from document IDs to their text content
import sys
# documents_ids, documents already defined above
if len(documents_ids) != len(documents):
    print("Document ID/text mismatch, exiting.")
    sys.exit(1)
doc_texts = dict(zip(documents_ids, documents))

# Initialize the Voyager retriever
retriever = retrieve.ColBERT(index=index)

# Define an answering function that retrieves relevant docs and uses the LLM
from dspy import ChainOfThought
def answer_question_with_docs(query: str, top_k: int = 3) -> str:
    # Encode and retrieve
    query_emb = model.encode([query], batch_size=32, is_query=True)
    raw_results = retriever.retrieve(queries_embeddings=query_emb, k=top_k)[0]
    # Collect top-k document texts
    selected = [doc_texts[res['id']] for res in raw_results]

    context = "\n\n".join(f"Document {i+1}:\n{text}" for i, text in enumerate(selected))
    # Build prompt for LLM
    prompt = f"Use the following documents to answer the question:\n{context}\n\nQuestion: {query}\nAnswer:"
    cot = ChainOfThought('question -> response')
    res = cot(question=prompt)
    return res.response

# Main interactive loop
if __name__ == "__main__":
    print("Interactive QA over indexed PDFs. Blank input to exit.")
    while True:
        q = input("Ask a question: ")
        if not q:
            break
        answer = answer_question_with_docs(q, top_k=3)
        print("\nAnswer:\n", answer)
