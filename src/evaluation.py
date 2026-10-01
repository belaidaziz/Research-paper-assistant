import json
import random
import re

import numpy as np
import ollama

from retriever import chunking_multiple
from embedding import model as embed_model, retrieve_chunks, build_context
from main import build_knowledge_base, answer_question


def _parse_json_response(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _raw_search(index, question, k):
    q_emb = embed_model.encode([question], normalize_embeddings=True)
    _, indices = index.search(np.array(q_emb).astype("float32"), k)
    return list(indices[0])

QUESTION_GEN_PROMPT = """Below is an excerpt from a research paper.

Excerpt:
{chunk}

Write ONE specific question that can be answered using ONLY this excerpt, \
along with a concise (1-3 sentence) correct answer based only on this excerpt. \
The question must be answerable without needing information from elsewhere \
in the paper, and should not refer to "the excerpt" or "this section" directly.

Respond with ONLY a JSON object, no other text, in this exact format:
{{"question": "<question>", "answer": "<answer>"}}
"""


def generate_eval_set(pdf_files, n_per_paper=3, min_chunk_len=200, seed=42):
    chunks, metadatas = chunking_multiple(pdf_files)

    by_paper = {}
    for i, m in enumerate(metadatas):
        by_paper.setdefault(m["paper"], []).append(i)

    random.seed(seed)
    eval_set = []

    for paper, indices in by_paper.items():
        candidates = [i for i in indices if len(chunks[i]) > min_chunk_len]
        sample = random.sample(candidates, min(n_per_paper, len(candidates)))

        for idx in sample:
            prompt = QUESTION_GEN_PROMPT.format(chunk=chunks[idx])
            response = ollama.generate(model="llama3", prompt=prompt)
            parsed = _parse_json_response(response["response"])

            if not parsed or "question" not in parsed or "answer" not in parsed:
                continue

            eval_set.append({
                "question": parsed["question"],
                "reference_answer": parsed["answer"],
                "source_paper": metadatas[idx]["paper"],
                "source_section": metadatas[idx]["section"],
                "source_chunk_id": metadatas[idx]["chunk_id"],
            })

    return eval_set


def evaluate_retrieval(eval_set, index, chunks, metadatas, k=15):

    raw_hits = 0
    reciprocal_ranks = []
    post_rerank_hits = 0

    for item in eval_set:
        target_id = item["source_chunk_id"]

        raw_indices = _raw_search(index, item["question"], k)
        if target_id in raw_indices:
            raw_hits += 1
            reciprocal_ranks.append(1.0 / (raw_indices.index(target_id) + 1))
        else:
            reciprocal_ranks.append(0.0)

        reranked = retrieve_chunks(index, chunks, metadatas, item["question"], k=k)
        reranked_ids = [m["chunk_id"] for _, m, _ in reranked]
        if target_id in reranked_ids:
            post_rerank_hits += 1

    n = len(eval_set)
    return {
        "n_questions": n,
        f"embedding_recall@{k}": round(raw_hits / n, 3),
        f"mrr@{k}": round(sum(reciprocal_ranks) / n, 3),
        "post_rerank_hit_rate": round(post_rerank_hits / n, 3),
    }

JUDGE_PROMPT = """You are evaluating an AI research assistant's answer.

Question:
{question}

Retrieved Context (what the assistant had access to):
{context}

Reference Answer (ground truth, may be incomplete but factually correct):
{reference_answer}

Generated Answer (to be evaluated):
{answer}

Rate the Generated Answer on each of the following, as an integer 1-5:
- faithfulness: is every claim in it directly supported by the Retrieved \
Context? (5 = fully grounded, no invented facts; 1 = mostly unsupported)
- relevancy: does it directly address the Question? (5 = fully relevant; \
1 = off-topic)
- correctness: does it match the Reference Answer in substance? \
(5 = equivalent; 1 = contradicts or misses it)

Respond with ONLY a JSON object, no other text, in this exact format:
{{"faithfulness": <int>, "relevancy": <int>, "correctness": <int>, "reasoning": "<one sentence>"}}
"""


def judge_answer(question, context, reference_answer, answer):
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context,
        reference_answer=reference_answer,
        answer=answer,
    )
    response = ollama.generate(model="llama3", prompt=prompt)
    return _parse_json_response(response["response"])


def evaluate_generation(eval_set, index, chunks, metadatas):
    results = []

    for item in eval_set:
        answer, context = answer_question(
            item["question"], index, chunks, metadatas, return_context=True
        )
        scores = judge_answer(item["question"], context, item["reference_answer"], answer)

        results.append({
            **item,
            "generated_answer": answer,
            "faithfulness": scores.get("faithfulness") if scores else None,
            "relevancy": scores.get("relevancy") if scores else None,
            "correctness": scores.get("correctness") if scores else None,
            "judge_reasoning": scores.get("reasoning") if scores else "judge parse failed",
        })

    return results


def _average(results, field):
    values = [r[field] for r in results if r[field] is not None]
    return round(sum(values) / len(values), 2) if values else None

def run_evaluation(pdf_files, n_per_paper=3, k=15, cache_prefix="eval_index_cache"):
    print(f"Generating synthetic eval set from {len(pdf_files)} paper(s)...")
    eval_set = generate_eval_set(pdf_files, n_per_paper=n_per_paper)
    print(f"  -> {len(eval_set)} question/answer pairs generated\n")

    index, chunks, metadatas = build_knowledge_base(
        pdf_files, cache_prefix=cache_prefix, use_cache=True
    )

    print("Evaluating retrieval...")
    retrieval_scores = evaluate_retrieval(eval_set, index, chunks, metadatas, k=k)
    for key, value in retrieval_scores.items():
        print(f"  {key}: {value}")

    print("\nEvaluating generation (this calls the LLM judge per question)...")
    generation_results = evaluate_generation(eval_set, index, chunks, metadatas)

    print("\nGeneration scores (1-5, averaged):")
    print(f"  faithfulness: {_average(generation_results, 'faithfulness')}")
    print(f"  relevancy:    {_average(generation_results, 'relevancy')}")
    print(f"  correctness:  {_average(generation_results, 'correctness')}")

    with open("eval_results.json", "w") as f:
        json.dump({
            "retrieval": retrieval_scores,
            "generation_per_question": generation_results,
        }, f, indent=2)
    print("\nFull per-question results written to eval_results.json")

    return retrieval_scores, generation_results

if __name__ == "__main__":
    pdf_files = [
        "../data/Deep Residual Learning for Image Recognition.pdf",
    ]
    run_evaluation(pdf_files, n_per_paper=3)