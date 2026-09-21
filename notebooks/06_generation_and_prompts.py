# %% [markdown]
# # 06 — Generation & Prompt Engineering
# Load an LLM and compare RAG, CoT, ReAct, ToT, zero-shot, and few-shot prompts.

# %%
from rag_pipeline.embeddings import build_embeddings
from rag_pipeline.vectorstores import load_vectorstore
from rag_pipeline.retrieval import build_retriever
from rag_pipeline.generation import (
    build_llms,
    RAG_PROMPT_TEMPLATE, COT_PROMPT, ZERO_SHOT_PROMPT,
    FEW_SHOT_PROMPT, REACT_PROMPT, TOT_PROMPT,
)
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
        "max_new_tokens": 200,
        "temperature": 0.35,
        "top_p": 0.92,
        "repetition_penalty": 1.05,
        "load_in_4bit": True,
    }]
})
llm = llms["Llama-3.1-8B-Instruct"]

# %%
def show(prompt_template, question, k=3):
    docs = retriever.invoke(question)
    ctx = format_docs(docs, max_chars=300)
    prompt = prompt_template.format(context=ctx, question=question)
    out = ""
    for chunk in llm.stream(prompt, max_new_tokens=200):
        out += str(chunk)
    print(out)

# %%
q = "What are GANs?"

# %%
print("### RAG\n"); show(RAG_PROMPT_TEMPLATE, q)
# %%
print("### Chain-of-Thought\n"); show(COT_PROMPT, q)
# %%
print("### ReAct\n"); show(REACT_PROMPT, q)
# %%
print("### Tree-of-Thought\n"); show(TOT_PROMPT, q)
# %%
print("### Zero-shot\n"); show(ZERO_SHOT_PROMPT, q)
# %%
print("### Few-shot\n"); show(FEW_SHOT_PROMPT, q)
