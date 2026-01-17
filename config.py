"""
Configuration settings for Excel RAG System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Excel storage
EXCEL_STORE_PATH = DATA_DIR / "data_store.xlsx"
CANONICAL_SHEET_NAME = "All"

# ChromaDB settings
CHROMA_COLLECTION_NAME = "excel_rag_all"

# Nebius LLM settings (Primary - using OpenAI-compatible API)
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
LLM_CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "openai/gpt-oss-120b")
LLM_EMBEDDING_MODEL = os.getenv("LLM_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

# Gemini settings (Backup - currently paused)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

# RAG settings
TOP_K_RESULTS = 5
EMBEDDING_DIMENSION = 4096  # Qwen3-Embedding-8B dimension

# Column name typo fixes mapping
COLUMN_TYPO_FIXES = {
    "title_of_publicaion": "title",
    "title_of_publication": "title",
    "dept.": "dept",
    "department": "dept",
    "date_of_publication ": "date_of_publication",
    "date_of_publication": "publication_date",
    "pub_date": "publication_date",
    "author_name": "author",
    "authors": "author",
    "publication_type": "pub_type",
    "type": "pub_type",
}

# System prompt for RAG
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context from an Excel dataset.

IMPORTANT RULES:
1. Answer ONLY based on the information provided in the context below.
2. If the answer cannot be found in the context, say: "I don't know based on the uploaded data."
3. Be precise and cite specific data from the context when possible.
4. Do not make up or infer information not present in the context.
"""

RAG_USER_PROMPT_TEMPLATE = """Context from the Excel dataset:
{context}

Question: {question}

Please answer the question based only on the above context."""
