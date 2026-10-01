import os
from retriever import chunking_multiple
from embedding import (
    embedding_improved,
    retrieve_chunks,
    build_context,
    save_index,
    load_index,
)
import ollama
import numpy as np


def build_knowledge_base(pdf_files, cache_prefix="index_cache", use_cache=True):

    if use_cache and os.path.exists(f"{cache_prefix}.faiss"):
        index, chunks, metadatas = load_index(cache_prefix)
        return index, chunks, metadatas

    chunks, metadatas = chunking_multiple(pdf_files)
    index = embedding_improved(chunks)

    if use_cache:
        save_index(index, chunks, metadatas, cache_prefix)

    return index, chunks, metadatas


def answer_question(user_question, index, chunks, metadatas, k=25, return_context=False):
    retrieved = retrieve_chunks(index, chunks, metadatas, user_question, k=k)

    context = build_context(retrieved)

    prompt = f"""
You are an AI research assistant.

Answer the question using ONLY the provided context, which may come
from multiple papers.
If the answer is not present in the context, say:
"The provided papers do not contain this information."

Always cite sources in the format:
(Paper: <paper>, Section: <section>)

If different papers disagree or say different things, mention each
paper's position separately rather than blending them into one answer.

Context:
{context}

Question:
{user_question}

Answer:
"""

    response = ollama.generate(
        model="llama3",
        prompt=prompt
    )

    if return_context:
        return response["response"], context

    return response["response"]


def query_pipeline_multi(user_question, pdf_files, cache_prefix="index_cache", use_cache=True):

    index, chunks, metadatas = build_knowledge_base(pdf_files, cache_prefix, use_cache)
    return answer_question(user_question, index, chunks, metadatas)
