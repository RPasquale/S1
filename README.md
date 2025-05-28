# PDF QA Chatbot with Advanced Training

This project lets you interactively query a collection of PDFs with advanced model training and optimization features:

- **Modern React/TypeScript Frontend**: Sleek, responsive UI with conversation management
- **pylate/Reasoning ModernColBERT**: Advanced document embedding and retrieval
- **dspy/deepseek-r1:8b**: LLM-powered Chain-of-Thought prompting
- **Adaptive Model Training**: Automatic fine-tuning of models on your documents
- **Multi-stage Query Generation**: Sophisticated query extraction for better understanding


## Prerequisites

- Python 3.8+
- Windows OS
- An Ollama-compatible LLM server running (e.g., via `ollama serve`)
- PDFs


## Setup

1. Clone or download this repository:
   ```powershell
   cd C:\Users\Admin\S1
   ```
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   *(Requirements include `pylate`, `PyPDF2`, `dspy`, and their dependencies.)*


## One-time Indexing

On the first run, the script will build a **Voyager** index under `pylate-index/`. This may take a minute:

```powershell
python model.py
```

You should see progress bars for:
- Encoding documents (batch size 32)
- Adding document embeddings to the index

Once complete, the index folder will contain `index.voyager` and supporting SQLite files.


## Interactive QA

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
