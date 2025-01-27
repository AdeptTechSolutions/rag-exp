import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

load_dotenv()


class DocumentIngester:
    def __init__(
        self,
        collection_name: str = "pdf_collection",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="retrieval_document",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

        self._initialize_collection()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

    def _initialize_collection(self) -> None:
        collections = self.client.get_collections()
        collection_names = [collection.name for collection in collections.collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                ),
            )

    def ingest_documents(self, data_dir: str) -> None:
        documents = []
        data_path = Path(data_dir)

        for pdf_path in data_path.glob("*.pdf"):
            loader = PyMuPDFLoader(str(pdf_path))
            documents.extend(loader.load())

        chunks = self._prepare_chunks(documents)
        self._store_chunks(chunks)

    def _prepare_chunks(self, documents: List[Document]) -> List[Document]:
        processed_chunks = []

        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            for chunk in chunks:
                if not chunk.page_content.strip():
                    continue

                chunk.metadata.update(
                    {
                        "source": os.path.basename(chunk.metadata["source"]),
                        "page": chunk.metadata.get("page", 0),
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    }
                )
                processed_chunks.append(chunk)

        return processed_chunks

    def _store_chunks(self, chunks: List[Document]) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self._initialize_collection()
        self.vector_store.add_documents(chunks)

    def get_collection_info(self) -> Optional[dict]:
        try:
            return self.client.get_collection(self.collection_name)
        except Exception:
            return None
