# PDF QA Chatbot

This project lets you interactively query a collection of pdfs using: 

- **pylate/Reasoning ModernColBERT** for document embedding and retrieval
- **dspy/deepseek-r1:8b** for LLM-powered Chain-of-Thought prompting


## Prerequisites

- Python 3.8+
- Windows OS
- An Ollama-compatible LLM server running (e.g., via `ollama serve`)
- CFA Level 2 PDFs organized under `C:\Users\Admin\OneDrive\CFAL2`


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


## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
