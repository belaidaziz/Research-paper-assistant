import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sentence_transformers import CrossEncoder

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
reranker = CrossEncoder("BAAI/bge-reranker-base")

def embedding_baseline(sections):
    texts = [s.page_content for s in sections]
    metadatas = [s.metadata for s in sections]
    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    return texts, metadatas, index

def embedding_improved(chunks):
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(np.array(embeddings).astype("float32"))

    return index

def retrieve_chunks(index, chunks, metadatas, question, k=15):

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    distances, indices = index.search(
        np.array(question_embedding).astype("float32"),
        k
    )

    candidate_chunks = [chunks[i] for i in indices[0]]
    candidate_metadata = [metadatas[i] for i in indices[0]]

    pairs = [[question, chunk] for chunk in candidate_chunks]
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidate_chunks, candidate_metadata, scores),
        key=lambda x: x[2],
        reverse=True
    )

    return reranked[:5]

def save_index(index, chunks, metadatas, path_prefix="index_cache"):
    """Persist the FAISS index plus its chunks/metadatas so a multi-paper
    corpus doesn't need to be re-embedded on every question."""
    faiss.write_index(index, f"{path_prefix}.faiss")
    np.save(f"{path_prefix}_chunks.npy", np.array(chunks, dtype=object))
    np.save(f"{path_prefix}_metadatas.npy", np.array(metadatas, dtype=object))

def load_index(path_prefix="index_cache"):
    index = faiss.read_index(f"{path_prefix}.faiss")
    chunks = np.load(f"{path_prefix}_chunks.npy", allow_pickle=True).tolist()
    metadatas = np.load(f"{path_prefix}_metadatas.npy", allow_pickle=True).tolist()
    return index, chunks, metadatas

def build_context(retrieved):

    context = ""

    for chunk, metadata, score in retrieved:

        context += f"""
[Paper: {metadata['paper']}]
[Section: {metadata['section']}]
[Subsection: {metadata['subsection']}]

{chunk}

---
"""

    return context