# PDF QA Chatbot with Advanced Training

This project lets you interactively query a collection of PDFs with advanced model training and optimization features:

- **Modern React/TypeScript Frontend**: Sleek, responsive UI with conversation management
- **pylate/Reasoning ModernColBERT**: Advanced document embedding and retrieval
- **dspy/deepseek-r1:1.5b**: LLM-powered Chain-of-Thought prompting
- **Adaptive Model Training**: Automatic fine-tuning of models on your documents
- **Multi-stage Query Generation**: Sophisticated query extraction for better understanding


## Key Features

- **Conversational Interface**: Manage multiple conversations with your documents
- **Document Processing**: Upload PDF files with real-time progress tracking
- **Advanced Retrieval**: Semantic search powered by ColBERT embeddings
- **Model Training**: Automatic fine-tuning with sophisticated techniques:
  - Multi-stage query generation with factual, analytical, and comparative questions
  - Hierarchical document clustering for comprehensive coverage
  - Entity and concept extraction for domain-specific questions
  - Embedding model training with hard negative mining
  - DSPy pipeline optimization with teleprompter

## Prerequisites

- Python 3.8+
- Windows OS
- Node.js 16+ and npm
- An Ollama-compatible LLM server running (e.g., via `ollama serve`)
- PDFs for document QA

## Backend Setup

1. Clone or download this repository:
   ```powershell
   cd C:\Users\Admin\S1
   ```
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```
3. Install backend dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   *(Requirements include `fastapi`, `pylate`, `PyPDF2`, `dspy`, `scikit-learn` and their dependencies.)*
4. Start the FastAPI server:
   ```powershell
   uvicorn server:app --host 127.0.0.1 --port 5000 --reload
   ```

## Frontend Setup

1. Navigate to the frontend directory:
   ```powershell
   cd frontend
   ```
2. Install frontend dependencies:
   ```powershell
   npm install
   ```
3. Start the development server:
   ```powershell
   npm run dev
   ```
4. Open your browser at http://127.0.0.1:3000

## Document Processing

The system automatically builds and maintains an embedding index for your documents:

1. Upload PDF files through the web interface
2. Watch real-time progress of document processing
3. After upload completes, model training starts automatically in the background
4. Track training progress through the training status modal
5. Once training completes, the system will use optimized models for retrieval and question answering

## Advanced Usage

After indexing, re-running the script skips re-indexing and drops you into an interactive prompt:

```powershell
python model.py
# Interactive QA over indexed PDFs. Blank input to exit.
Ask a question: <ask questions about the documents>
```

The script will retrieve the top K relevant PDFs, assemble their text, and call your LLM via dspy's `ChainOfThought` to generate a detailed answer.


## Configuration

- **LLM Model**: Change the `pylate_model_id` in `model.py` to any valid Hugging Face or Ollama model (default: `lightonai/Reason-ModernColBERT`).
- **Top-K Results**: Adjust `top_k` in `answer_question_with_docs()` for more or fewer retrieved documents.
- **Document Path**: Modify `doc_folder` in `model.py` to point to your PDF collection.


## Troubleshooting

- **File Not Found**: Ensure your PDFs are in the correct folder and accessible.
- **Indexing Again**: Delete the `pylate-index` folder if you need to rebuild from scratch.
- **LLM Errors**: Verify your Ollama server is running and `api_base`/`api_key` in `model.py` are correct.


## Running the Application

### 1. Start the Python Backend
This serves `/upload` and `/chat` endpoints and powers the Retriever + LLM logic.

```powershell
# Activate your virtual environment
.\.\venv\Scripts\Activate

# Install any new dependencies
pip install fastapi uvicorn python-multipart

# Run the server with hot-reload on localhost
uvicorn server:app --host 127.0.0.1 --reload --port 5000
```

### 2. Start the React + TypeScript Frontend
This provides the chat UI, markdown rendering, and folder uploader.

```powershell
cd frontend
npm install
npm run dev
```

By default, Vite is configured to proxy `/chat` and `/upload` calls to `http://localhost:5000`. 

Open http://localhost:5173 in your browser to start using the chat interface.


## Next Steps
- Customize styling or component layout in `frontend/src`.
- Adjust `model.py` or `server.py` for advanced QA flows.
- Add error handling and validation in the upload/chat endpoints.


## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
