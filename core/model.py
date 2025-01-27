import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

load_dotenv()

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that provides accurate, informative answers based on the given context.

Use the following pieces of context to answer the question at the end.

Context:
{context}

Question: {question}

Answer (provide a clear, direct answer based on the context above):"""


class RAGModel:
    def __init__(
        self,
        collection_name: str = "pdf_collection",
        temperature: float = 0.1,
        top_k: int = 4,
    ):
        self.collection_name = collection_name
        self.temperature = temperature
        self.top_k = top_k

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="retrieval_document",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=self.temperature,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        self.chain = (
            {
                "context": self._get_context,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _get_context(self, question: str) -> str:
        docs_and_scores = self._get_relevant_documents_with_scores(question)
        return "\n\n---\n\n".join(doc.page_content for doc, _ in docs_and_scores)

    def _get_relevant_documents_with_scores(
        self, question: str
    ) -> List[Tuple[Document, float]]:
        docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
            question, k=self.top_k
        )

        return [(doc, score) for doc, score in docs_and_scores if score > 0]

    def get_answer(self, question: str) -> Tuple[str, Dict]:
        docs_and_scores = self._get_relevant_documents_with_scores(question)
        answer = self.chain.invoke(question)

        source_document = None
        if docs_and_scores:
            best_doc, score = docs_and_scores[0]
            source_document = {
                "source": best_doc.metadata.get("source", "Unknown"),
                "page": best_doc.metadata.get("page", 0),
                "content": best_doc.page_content,
                "score": score,
            }

        return answer, source_document or {
            "source": None,
            "page": None,
            "content": None,
            "score": 0.0,
        }
