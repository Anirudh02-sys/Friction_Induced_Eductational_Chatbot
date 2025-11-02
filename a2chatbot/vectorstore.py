import chromadb
from sentence_transformers import SentenceTransformer
import pdfplumber

chroma_client = chromadb.PersistentClient(path="./chromadb_storage")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_collection(name):
    return chroma_client.get_or_create_collection(name)

def embed_text(text_list):
    return model.encode(text_list).tolist()

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=600):
    words = text.split()
    chunks = []
    current = []
    for w in words:
        current.append(w)
        if len(current) >= chunk_size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks
