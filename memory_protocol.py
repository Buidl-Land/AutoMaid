import chromadb
import openai
import json
import os
from typing import List, Dict, Any
from chromadb.utils import embedding_functions

class MemoryProtocol:
    """
    Handles the agent's long-term memory, including ingestion, storage, and retrieval
    using a vector database (ChromaDB) and a configurable embedding model.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the MemoryProtocol with a resolved configuration dictionary.

        Args:
            config: The fully resolved 'memory_protocol' configuration section.
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        if not self.enabled:
            return

        self.client = self._get_chroma_client()
        self.embedding_function = self._get_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name="agent_memory",
            embedding_function=self.embedding_function
        )

    def _get_chroma_client(self):
        """Initializes and returns the ChromaDB client based on the config."""
        db_config = self.config.get("database", {})
        mode = db_config.get("mode", "persistent")

        if mode == "persistent":
            path = db_config.get("path", "./.rooroo/memory_db")
            os.makedirs(path, exist_ok=True)
            return chromadb.PersistentClient(path=path)
        elif mode == "http":
            host = db_config.get("host", "localhost")
            port = db_config.get("port", 8000)
            return chromadb.HttpClient(host=host, port=port)
        else:
            raise ValueError(f"Unsupported ChromaDB mode: {mode}")

    def _get_embedding_function(self):
        """
        Initializes and returns the appropriate embedding function based on the config.
        """
        embed_config = self.config.get("embedding", {})
        provider = embed_config.get("provider")

        if provider == "openai":
            api_key = embed_config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in config or OPENAI_API_KEY environment variable.")

            # The embedding function needs to be compatible with OpenAI's API signature.
            # We can use chromadb's helper or construct our own if more customization is needed.
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=embed_config.get("model_name", "text-embedding-ada-002"),
                api_base=embed_config.get("base_url") # Pass base_url if present
            )
        # Example for a future SentenceTransformer implementation
        # elif provider == "sentence-transformers":
        #     return embedding_functions.SentenceTransformerEmbeddingFunction(
        #         model_name=embed_config.get("model_name", "all-MiniLM-L6-v2")
        #     )
        else:
            raise NotImplementedError(f"Embedding provider '{provider}' is not supported.")

    def ingest_memory(self, text: str, metadata: Dict[str, Any]):
        """
        Ingests a new memory into the vector store.

        Args:
            text: The text content of the memory to be ingested and embedded.
            metadata: A dictionary of metadata associated with the memory.
                      Must include a unique 'id' for the memory.
        """
        if not self.enabled:
            return

        if 'id' not in metadata:
            raise ValueError("Metadata must contain a unique 'id' for the memory.")

        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[str(metadata['id'])]
        )
        print(f"Ingested memory with ID: {metadata['id']}")

    def retrieve_memories(self, query_text: str, n_results: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves relevant memories from the vector store based on a query.

        Args:
            query_text: The text to search for.
            n_results: The number of results to return. Defaults to the config value.

        Returns:
            A list of dictionaries, where each dictionary represents a retrieved memory.
        """
        if not self.enabled:
            return []

        k = n_results if n_results is not None else self.config.get("retrieval_k", 5)

        results = self.collection.query(
            query_texts=[query_text],
            n_results=k
        )

        # The query returns a complex structure, we need to parse it.
        retrieved_memories = []
        if results and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                retrieved_memories.append({
                    "id": doc_id,
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })

        return retrieved_memories
