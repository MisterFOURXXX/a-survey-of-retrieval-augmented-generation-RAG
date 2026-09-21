# %% [markdown]
# # 08 — Evaluation
# Perplexity, BERTScore, semantic similarity, NLI, and RAGAS.

# %%
from rag_pipeline.embeddings import build_embeddings
from rag_pipeline.vectorstores import load_vectorstore
from rag_pipeline.retrieval import build_retriever
from rag_pipeline.generation import build_llms, RAG_PROMPT_TEMPLATE
from rag_pipeline.evaluation import (
    perplexity_score, bertscore, SemanticEvaluator, NLIEvaluator,
)

# %%
emb = build_embeddings({"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"})
store = load_vectorstore(emb, {"type": "faiss", "persist_dir": "indexes/faiss_arxiv"})
retriever = build_retriever(store, {"search_type": "similarity", "k": 3})
llm = build_llms({
    "models": [{
        "name": "Llama-3.1-8B-Instruct",
        "hf_path": "meta-llama/Llama-3.1-8B-Instruct",
        "max_new_tokens": 120,
        "load_in_4bit": True,
    }]
})["Llama-3.1-8B-Instruct"]

# %%
# Build a small test set from the index
docs = store.similarity_search("neural networks", k=5)
question_templates = [
    "What is the main contribution of this paper?",
    "Summarize the key findings of this research.",
    "What methods are used in this study?",
    "What problem does this paper address?",
    "What are the implications of this research?",
]
test_data = [
    {
        "question": question_templates[i % len(question_templates)],
        "context": d.page_content,
        "reference": d.page_content[:200],
    }
    for i, d in enumerate(docs)
]
questions = [x["question"] for x in test_data]
contexts = [x["context"] for x in test_data]
references = [x["reference"] for x in test_data]

# %%
# Generate answers
answers = []
for q, c in zip(questions, contexts):
    prompt = RAG_PROMPT_TEMPLATE.format(context=c, question=q)
    out = ""
    for chunk in llm.stream(prompt, max_new_tokens=120):
        out += str(chunk)
    answers.append(out.strip())
print("Generated", len(answers), "answers")

# %%
print("Perplexity:", perplexity_score(answers, model_id="gpt2"))

# %%
print("BERTScore:", bertscore(answers, references))

# %%
sem = SemanticEvaluator()
print("Semantic:", sem.evaluate(questions, answers, contexts))

# %%
nli = NLIEvaluator()
print("NLI:", nli.evaluate(answers, contexts))

# %%
# RAGAS (needs OpenAI key by default)
# from rag_pipeline.evaluation import run_ragas
# print("RAGAS:", run_ragas(questions, answers, [[c] for c in contexts], references))
