import os
import time
import faiss
import logging
import numpy as np
import ollama

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = r"D:\Projects\AI_chatbot\NEW\knowledge"
FAISS_DB_PATH = r"D:\Projects\AI_chatbot\NEW\vector_store"

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(FAISS_DB_PATH, exist_ok=True)

# FastAPI App
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedding Model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

# Load FAISS Index
if os.path.exists(FAISS_DB_PATH):
    logger.info("Loading existing FAISS index...")
    vector_store = FaissVectorStore.from_persist_dir(FAISS_DB_PATH)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=FAISS_DB_PATH
    )
    query_engine = load_index_from_storage(storage_context=storage_context).as_query_engine()
else:
    query_engine = None


# Function to build and persist index
def build_and_store_index(documents):
    global query_engine
    faiss_index = faiss.IndexFlatL2(384)
    faiss_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=faiss_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=FAISS_DB_PATH)
    query_engine = index.as_query_engine()
    logger.info("FAISS index built and stored successfully.")


# Upload endpoint
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(DOCUMENTS_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    logger.info(f"File {file.filename} uploaded. Rebuilding index...")
    docs = SimpleDirectoryReader(DOCUMENTS_DIR).load_data()
    build_and_store_index(docs)

    return {"message": "File uploaded and indexed successfully!"}


# Query and stream response
@app.post("/query/")
async def query_model(question: str = Form(...)):
    return StreamingResponse(run_llm_query(question), media_type="text/plain")


# LLM Query Execution
async def run_llm_query(question: str):
    if query_engine is None:
        raise HTTPException(status_code=400, detail="No FAISS index found. Upload documents first.")

    logger.info(f"User Question: {question}")
    
    try:
        response = query_engine.query(question)
        context = response.response.strip() if response else "No relevant information found."
    except Exception as e:
        logger.error(f"Error while querying vector store: {e}")
        context = "Error retrieving context from documents."

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Avoid repetition. Give concise answers based on the context.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer clearly:",
        },
    ]

    try:
        ollama_response = ollama.chat(model="llama3.2", messages=messages)
        raw_answer = ollama_response["message"]["content"].strip()

        # Optional: deduplicate repeated lines
        unique_lines = []
        for line in raw_answer.split("\n"):
            cleaned_line = line.strip()
            if cleaned_line and cleaned_line not in unique_lines:
                unique_lines.append(cleaned_line)

        final_answer = "\n".join(unique_lines)
        logger.info(f"LLM Response: {final_answer}")
    except Exception as e:
        logger.error(f"LLM response error: {e}")
        final_answer = "Error generating the response."

    yield final_answer
