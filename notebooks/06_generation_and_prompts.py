# %% [markdown]
# # 06 — Generation & Prompt Engineering
# Compare RAG, CoT, ReAct, ToT, zero-shot, and few-shot prompts.

# %%
from rag_pipeline.embeddings import build_embeddings
from rag_pipeline.vectorstores import load_vectorstore
from rag_pipeline.retrieval import build_retriever
from rag_pipeline.generation import build_llms, PROMPT_REGISTRY
from rag_pipeline.utils import format_docs

# %%
emb = build_embeddings({"provider": "huggingface", "model": "BAAI/bge-small-en-v1.5"})
store = load_vectorstore(emb, {"type": "faiss", "persist_dir": "indexes/faiss_arxiv"})
retriever = build_retriever(store, {"search_type": "similarity", "k": 3})

# %%
llms = build_llms({
    "models": [{
        "name": "Llama-3.1-8B-Instruct",
        "hf_path": "meta-llama/Llama-3.1-8B-Instruct",
        "max_new_tokens": 200, "temperature": 0.35, "top_p": 0.92,
        "repetition_penalty": 1.05, "load_in_4bit": True,
    }]
})
llm = llms["Llama-3.1-8B-Instruct"]

# %%
def show(prompt_name, question):
    template = PROMPT_REGISTRY[prompt_name]
    docs = retriever.invoke(question)
    ctx = format_docs(docs, max_chars=300)
    prompt = template.format(context=ctx, question=question)
    out = ""
    for c in llm.stream(prompt, max_new_tokens=200):
        out += str(c)
    print(out)

# %%
q = "What are GANs?"
for name in ["rag", "cot", "react", "tot", "few_shot"]:
    print(f"\n### {name.upper()}\n")
    show(name, q)
