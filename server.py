from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, importlib
from typing import List
import model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    # Save uploaded PDFs into the doc_folder, preserving subpaths
    for f in files:
        dest = os.path.join(model.doc_folder, f.filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as out:
            out.write(await f.read())
    # Remove existing index to force rebuild
    if os.path.exists(model.index_folder):
        shutil.rmtree(model.index_folder)
    # Reload model to trigger indexing on import
    importlib.reload(model)
    return {"status": "indexed"}

@app.post("/chat")
async def chat(payload: dict):
    question = payload.get("question", "")
    answer = model.answer_question_with_docs(question)
    return {"answer": answer}
