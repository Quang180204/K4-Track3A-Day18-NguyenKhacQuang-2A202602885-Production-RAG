"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys & LLM Provider ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-..."):
    if GEMINI_API_KEY:
        OPENAI_API_KEY = GEMINI_API_KEY
        OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
        LLM_MODEL = "gemini-3.1-flash-lite"
    else:
        OPENAI_BASE_URL = None
        LLM_MODEL = "gpt-4o-mini"
else:
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Set in os.environ so standard OpenAI() calls pick up base_url and api_key automatically
if OPENAI_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
