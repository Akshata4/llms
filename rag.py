# main.py
from __future__ import annotations

import argparse
import os
from typing import List, TypedDict

from dotenv import load_dotenv
import bs4

from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph


# Lightweight fallback embeddings implementation for offline/testing
class SimpleFallbackEmbeddings:
    """A tiny deterministic embedding fallback used when the real embeddings
    service is unavailable. It produces small fixed-size vectors derived from
    a SHA256 hash of the input text. This preserves relative similarity for
    short demos and lets the rest of the RAG pipeline run locally.

    Not suitable for production use. Use a real embedding provider for
    meaningful search results.
    """
    def __init__(self, dim: int = 16):
        self.dim = dim

    def _text_to_vector(self, text: str):
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand/contract bytes into floats in range [-1,1]
        vec = []
        for i in range(self.dim):
            b = h[i % len(h)]
            vec.append((b / 255.0) * 2 - 1)
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t or "") for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._text_to_vector(text or "")


# ---------------------------
# Config / setup
# ---------------------------
def get_api_key() -> str:
    load_dotenv()  # loads .env from project root
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Put it in a .env file, e.g.\n"
            "OPENROUTER_API_KEY=sk-or-... "
        )
    return api_key


def build_llm_and_embeddings(api_key: str):
    # OpenRouter is API-compatible with OpenAI; just point base URL
    base = "https://openrouter.ai/api/v1"

    llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    # optional but recommended by OpenRouter:
    default_headers={
        "HTTP-Referer": "http://localhost",   # your site/app URL if you have one
        "X-Title": "LangChain RAG Script",
    },
)


    embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",   # <-- add the provider prefix
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

    return llm, embeddings


# ---------------------------
# RAG building blocks
# ---------------------------
def load_and_chunk(url: str) -> List[Document]:
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            # keep it robust: parse the whole body (Strainer is optional)
            parse_only=bs4.SoupStrainer(name=True)
        ),
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def index_docs(chunks: List[Document], embeddings: OpenAIEmbeddings):
    # Sanity-check the embeddings client before adding all docs so we can
    # surface a helpful error if the provider/credentials are misconfigured.
    sample_texts = [c.page_content for c in chunks[:2]] if chunks else []
    fallback_used = False
    try:
        # ask the embeddings object to embed a tiny sample
        sample_vectors = embeddings.embed_documents(sample_texts) if sample_texts else []
        # Basic validation of the returned embeddings
        if sample_vectors and (not isinstance(sample_vectors, list) or not hasattr(sample_vectors[0], "__len__")):
            raise TypeError("Unexpected embeddings response type: {}".format(type(sample_vectors)))
    except Exception as e:
        # Fall back to a deterministic local embedding implementation so the
        # rest of the RAG flow can run for demos/tests.
        print("WARNING: remote embeddings failed ({}). Falling back to local embeddings.".format(repr(e)))
        embeddings = SimpleFallbackEmbeddings()
        fallback_used = True

    vs = InMemoryVectorStore(embeddings)
    vs.add_documents(chunks)
    return vs


class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str


def build_graph(vector_store: InMemoryVectorStore, llm: ChatOpenAI):
    # Slim RAG prompt from the Hub
    prompt = hub.pull("rlm/rag-prompt")

    def retrieve(state: RAGState):
        retrieved = vector_store.similarity_search(state["question"], k=4)
        return {"context": retrieved}

    def generate(state: RAGState):
        ctx_text = "\n\n".join(d.page_content for d in state["context"])
        msgs = prompt.invoke({"question": state["question"], "context": ctx_text})
        out = llm.invoke(msgs)
        return {"answer": out.content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    return graph.compile()


# ---------------------------
# CLI + main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Minimal RAG with OpenRouter + LangChain")
    p.add_argument(
        "--url",
        default="https://python.langchain.com/docs/tutorials/rag/",
        help="Web page to index (default: LangChain RAG tutorial)",
    )
    p.add_argument(
        "--question",
        required=True,
        help="Your question to ask over the indexed content",
    )
    return p.parse_args()


def main():
    args = parse_args()
    api_key = get_api_key()
    llm, embeddings = build_llm_and_embeddings(api_key)

    print(f"Loading & chunking: {args.url}")
    chunks = load_and_chunk(args.url)

    print(f"Indexing {len(chunks)} chunks into in-memory vector store…")
    vs = index_docs(chunks, embeddings)

    print("Compiling graph (retrieve -> generate)…")
    graph = build_graph(vs, llm)

    result = graph.invoke({"question": args.question})
    print("\n=== QUESTION ===")
    print(args.question)
    print("\n=== ANSWER ===")
    print(result["answer"])

    # Optional: show where top contexts came from
    print("\n=== TOP CONTEXT SOURCES ===")
    for i, d in enumerate(vs.similarity_search(args.question, k=3), start=1):
        src = d.metadata.get("source") or d.metadata.get("loc") or "unknown"
        print(f"{i}. {src}")


if __name__ == "__main__":
    main()
# Load .env file
# load_dotenv()