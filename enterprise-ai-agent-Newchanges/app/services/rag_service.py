import logging
from app.db.chroma_client import chroma_client
from app.services.embeddings import embeddings_service
from app.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    initialized = False

    @classmethod
    def initialize(cls):
        try:
            chroma_client.initialize()
            embeddings_service.initialize()
            cls.initialized = True
            logger.info("RAG retrieval service initialized")
        except Exception as e:
            logger.warning("RAG service initialization failed: %s", str(e))
            cls.initialized = False

    @classmethod
    async def index_documents(cls, docs):
        if not cls.initialized:
            logger.warning("RAG service not initialized. Skipping document indexing.")
            return
        embeddings = embeddings_service.embed([doc["text"] for doc in docs])
        ids = [doc.get("id", str(idx)) for idx, doc in enumerate(docs)]
        documents = [doc["text"] for doc in docs]
        metadatas = [doc.get("metadata", {}) for doc in docs]
        chroma_client.add_documents(ids, documents, metadatas, embeddings)
        logger.info("Indexed %d documents into ChromaDB", len(docs))

    @classmethod
    async def retrieve(cls, query: str, top_k: int = 5):
        if not cls.initialized:
            logger.debug("RAG service not initialized. Returning empty results.")
            return []
        query_embeddings = embeddings_service.embed(query)
        results = chroma_client.query(query_embeddings=query_embeddings, n_results=top_k)
        docs = []
        for row in results["documents"]:
            docs.extend(row)
        docs = [doc for doc in docs if doc]
        logger.debug("RAG retrieved %d docs", len(docs))
        return docs


rag_service = RAGService()
