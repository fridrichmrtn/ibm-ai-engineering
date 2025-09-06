"""RAG-style PDF QA using LangChain + OpenAI + Chroma.

- Loads a PDF, chunks it, embeds it with OpenAI, stores in Chroma, and
  answers questions with a RetrievalQA chain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LangChain >= 0.2 moved RetrievalQA here:
from langchain.chains.retrieval_qa.base import RetrievalQA  # noqa: E0611
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_TOKEN_FILE = os.getenv("OPENAI_TOKEN_FILE", "openai_token")


@dataclass(frozen=True)
class RagConfig:
    """Configuration for the RAG pipeline."""
    llm_model: str = DEFAULT_LLM_MODEL
    embedding_model: str = DEFAULT_EMBED_MODEL
    chunk_size: int = 1024
    chunk_overlap: int = 64
    mmr_k: int = 6
    mmr_lambda: float = 0.25
    temperature: float = 0.1
    max_tokens: int = 600


class RagQA:
    """Encapsulates the RetrievalQA pipeline over a single PDF."""

    def __init__(self, config: Optional[RagConfig] = None) -> None:
        self.config = config or RagConfig()
        self._chat_history: List[Tuple[str, str]] = []
        self._qa: Optional[RetrievalQA] = None

        self._ensure_openai_key()
        self._llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self._embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        self._prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "<SYSTEM_PROMPT>\n"
                "You are a helpful AI assistant. Use the provided context to answer.\n"
                "If the answer is not in the context, say you don't know.\n"
                "</SYSTEM_PROMPT>\n\n"
                "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            ),
        )

    @staticmethod
    def _ensure_openai_key() -> None:
        """Load OPENAI_API_KEY from file if present and not already set."""
        if os.getenv("OPENAI_API_KEY"):
            return
        if os.path.exists(OPENAI_TOKEN_FILE):
            with open(OPENAI_TOKEN_FILE, "r", encoding="utf-8") as f:
                os.environ["OPENAI_API_KEY"] = f.read().strip()

    def build_index_from_pdf(self, pdf_path: str) -> None:
        """Load, split, embed and build a Chroma index from a PDF."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(chunks, embedding=self._embeddings)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.config.mmr_k, "lambda_mult": self.config.mmr_lambda},
        )

        self._qa = RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            input_key="question",
            chain_type_kwargs={"prompt": self._prompt},
        )

    def ask(self, question: str) -> str:
        """Ask a question against the built index."""
        if self._qa is None:
            raise RuntimeError("Index not built. Call build_index_from_pdf(...) first.")

        output = self._qa.invoke({"question": question, "chat_history": self._chat_history})
        answer: str = output["result"]
        self._chat_history.append((question, answer))
        return answer

    @property
    def chat_history(self) -> List[Tuple[str, str]]:
        """Return the chat history as (question, answer) tuples."""
        return list(self._chat_history)


# ---------- Example usage (remove or guard under __main__ as needed) ----------
# if __name__ == "__main__":
#     rag = RagQA()
#     rag.build_index_from_pdf("manual.pdf")
#     print(rag.ask("What is the warranty period?"))
