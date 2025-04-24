import faiss
import numpy as np
import os
import pdb

def build_faiss_hnsw_index(embedding_path, output_index_path, M=32, ef_construction=200):
    """
    Build a FAISS HNSW index for dense retrieval.

    :param embedding_path: Path to the .npy file containing embeddings
    :param doc_ids_path: Path to the .npy file containing document IDs
    :param output_index_path: Path to save the FAISS HNSW index
    :param M: Number of connections per node (higher = more accurate, slower indexing)
    :param ef_construction: Number of neighbors considered during graph construction
    """

    # Load embeddings and document IDs
    print(f"Loading embeddings from {embedding_path}...")
    plm_size = 1024 if 'large' in embedding_path else 768
    embeddings = np.fromfile(embedding_path, dtype=np.float32).reshape(-1, plm_size) # FAISS requires float32
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Initialize FAISS HNSW index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(embedding_dim, M)  # HNSW with Inner Product (Cosine similarity)

    # Set efConstruction (controls graph quality)
    index.hnsw.efConstruction = ef_construction

    # Add embeddings to the index
    print("Adding embeddings to FAISS HNSW index...")
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, output_index_path)
    print(f"FAISS HNSW index saved to {output_index_path}")



if __name__ == '__main__':
    # File paths
    output_dir = "data/esci/raw/esci"

    ##### Modify
    model_name = 'simcse-large'
    embedding_path = f"{output_dir}/esci.simcse-largeCLS"

    save_dir = f'data/esci/raw/dense_index/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    index_output_path = f"{save_dir}/faiss_hnsw_index.bin"

    # Create FAISS HNSW index
    build_faiss_hnsw_index(embedding_path, index_output_path, M=32, ef_construction=200)
