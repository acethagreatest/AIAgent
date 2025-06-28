import os
import numpy as np
from numpy.typing import NDArray
import faiss  # type: ignore
from typing import List
from sentence_transformers import SentenceTransformer

def compute_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> NDArray[np.float32]:
    """
    Computes embeddings for a list of texts using a Sentence Transformer model.
    """
    model = SentenceTransformer(model_name)
    # Ensure the output is a 2D NumPy array of type np.float32
    embeddings: NDArray[np.float32] = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float32)

def build_index(embeddings: NDArray[np.float32]) -> faiss.IndexFlatL2:
    """
    Builds a FAISS index from embeddings.
    """
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be a 2D array.")
    index.add(np.array(embeddings, dtype=np.float32))  # Ensure embeddings are of type np.float32
    return index

def retrieve_related_text(query: str, model: SentenceTransformer, index: faiss.IndexFlatL2, texts: List[str], k: int = 3) -> List[str]:
    """
    Given a query, retrieves the top k most similar texts from the FAISS index.
    """
    # Compute the query embedding and ensure it is a 2D NumPy array of type np.float32
    query_embedding: NDArray[np.float32] = model.encode([query], show_progress_bar=False)
    if len(query_embedding.shape) != 2:
        query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    results = [texts[i] for i in indices[0]]
    return results

# -- Functions for working with a repository (e.g., Rainlang) --

def load_repo_files(repo_path: str, file_ext: str = ".rain") -> List[str]:
    """
    Walks through the repository directory and returns a list of file contents
    for files ending with the given extension.
    """
    file_contents = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(file_ext):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_contents.append(f.read())
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return file_contents

def compute_repo_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> NDArray[np.float32]:
    """
    Computes embeddings for repository texts.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def build_repo_index(embeddings: NDArray[np.float32]) -> faiss.IndexFlatL2:
    """
    Builds a FAISS index for repository embeddings.
    """
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be a 2D array.")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_repo_context(query: str, model: SentenceTransformer, index: faiss.IndexFlatL2, texts: List[str], k: int = 3) -> List[str]:
    """
    Retrieves the top k most similar repository texts for the given query.
    """
    query_embedding: NDArray[np.float32] = model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    if len(query_embedding.shape) != 2:
        query_embedding = query_embedding.reshape(1, -1)
    _, indices = index.search(query_embedding, k)
    results = [texts[i] for i in indices[0]]
    return results