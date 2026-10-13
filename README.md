# Research Paper Assistant (RAG)

A local, multi-paper Retrieval-Augmented Generation (RAG) assistant that
answers questions about one or more research papers, grounded strictly in
their content — with citations, no cloud API dependency, and a built-in
evaluation harness.

## Features

- **Multi-paper support** — upload and query across several PDFs at once;
  answers are grounded in whichever paper(s) actually contain the relevant
  information.
- **Structure-aware chunking** — splits papers by heading (`#`, `##`, `###`)
  before sub-chunking, so retrieved context stays tied to its section/
  subsection instead of being cut arbitrarily.
- **Two-stage retrieval** — dense vector search (FAISS + `nomic-embed-text`)
  followed by cross-encoder reranking (`bge-reranker-base`) for more
  relevant context.
- **Source-grounded answers** — the LLM is instructed to answer only from
  retrieved context and to cite `(Paper: ..., Section: ...)` for every
  claim; it explicitly says when the papers don't contain an answer.
- **Local LLM** — generation runs through [Ollama](https://ollama.com)
  (`llama3`), so no API keys or external calls are required.
- **Index caching** — the FAISS index is built once per paper set and
  cached to disk, avoiding re-embedding on every question.
- **Gradio UI** — a simple web interface for uploading PDFs and asking
  questions interactively.
- **Evaluation harness** — generates a synthetic Q&A test set from the
  papers themselves and scores both retrieval (hit rate, MRR) and
  generation quality (faithfulness, relevancy, correctness via LLM-judge).

## Architecture

```
PDF(s)
  │
  ▼
pymupdf4llm (PDF → Markdown)
  │
  ▼
MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter  (retriever.py)
  │
  ▼
SentenceTransformer embeddings → FAISS index                 (embedding.py)
  │
  ▼
Vector search (top-k) → CrossEncoder reranking → top-5        (embedding.py)
  │
  ▼
Context assembly + prompt → Ollama (llama3)                   (main.py)
  │
  ▼
Grounded answer with citations
```

## Project Structure

```
.
├── retriever.py     # PDF-to-markdown conversion and chunking
├── embedding.py      # Embedding, FAISS indexing, retrieval, reranking
├── main.py            # Pipeline orchestration (build index, answer questions)
├── app.py              # Gradio interface
├── evaluation.py        # RAG evaluation: synthetic test sets + metrics
└── README.md
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the `llama3` model pulled:
  ```bash
  ollama pull llama3
  ```

### Python dependencies

```bash
pip install sentence-transformers faiss-cpu langchain-text-splitters pymupdf4llm gradio ollama numpy
```

## Web interface

```bash
python app.py
```

Then open the local Gradio URL, upload one or more PDFs, and ask a question.

### Evaluation

```python
from evaluation import run_evaluation

pdf_files = ["papers/paper1.pdf", "papers/paper2.pdf", "papers/paper3.pdf"]
run_evaluation(pdf_files, n_per_paper=3)
```

This generates a synthetic question set from the papers, evaluates
retrieval (embedding recall, MRR, post-rerank hit rate) and generation
quality (faithfulness, relevancy, correctness scored by an LLM judge),
prints a summary, and writes full per-question results to
`eval_results.json`.

## Notes & Limitations

- The index cache (`index_cache.*`) does not detect changes to the
  underlying PDF set — delete the cache files or pass `use_cache=False`
  after adding/removing papers.
- The evaluation judge currently reuses the same model (`llama3`) used for
  generation, which can be lenient toward its own answer style; treat
  scores as a relative/directional signal rather than an absolute
  benchmark.
- Synthetic evaluation questions are generated from single chunks, so they
  mainly test whether clearly-present information is retrieved and used
  correctly — not multi-hop reasoning across sections or papers.

## Tech Stack

Python · Sentence-Transformers · FAISS · LangChain (text splitters) ·
PyMuPDF4LLM · Ollama (Llama 3) · Gradio
