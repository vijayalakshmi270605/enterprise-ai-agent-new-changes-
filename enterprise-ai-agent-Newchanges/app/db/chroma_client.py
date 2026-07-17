from chromadb import Client
from chromadb.config import Settings as ChromaSettings
from app.config import settings


class ChromaClient:
    def __init__(self):
        self.client = None
        self.collection = None

    def initialize(self, collection_name: str = "enterprise_assistant"):
        self.client = Client(
            ChromaSettings(persist_directory=settings.chroma_persist_dir)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        return self.collection

    def add_documents(self, ids, documents, metadatas, embeddings):
        return self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_embeddings, n_results: int = 5):
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'],
        )


chroma_client = ChromaClient()
