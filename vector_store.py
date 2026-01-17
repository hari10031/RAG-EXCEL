"""
Vector Store - ChromaDB operations with incremental update support
- Manages the excel_rag_all collection
- Supports incremental embedding (only new rows)
- Provides query functionality
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Set, Tuple
import config
from llm_client import get_llm_client


class VectorStore:
    """ChromaDB vector store with incremental update support."""
    
    def __init__(self):
        """Initialize the vector store with persistent ChromaDB."""
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = config.CHROMA_COLLECTION_NAME
        self._collection = None
    
    @property
    def collection(self):
        """Get or create the collection lazily."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        return self._collection
    
    def get_existing_row_ids(self) -> Set[str]:
        """
        Get all row_ids currently in the collection.
        
        Returns:
            Set of existing row_ids
        """
        try:
            # Get all IDs from collection
            result = self.collection.get(include=[])
            return set(result['ids']) if result['ids'] else set()
        except Exception as e:
            print(f"Error getting existing row_ids: {e}")
            return set()
    
    def add_documents(self, documents: List[Dict], llm_client=None) -> Tuple[int, int]:
        """
        Add documents to the collection with incremental support.
        Only adds documents whose row_id doesn't already exist.
        
        Args:
            documents: List of dicts with 'row_id', 'text', 'metadata'
            llm_client: Optional LLMClient instance for embeddings
            
        Returns:
            Tuple of (added_count, skipped_count)
        """
        if not documents:
            return 0, 0
        
        # Get existing row_ids
        existing_ids = self.get_existing_row_ids()
        
        # Filter to only new documents
        new_docs = [doc for doc in documents if doc['row_id'] not in existing_ids]
        skipped_count = len(documents) - len(new_docs)
        
        if not new_docs:
            return 0, skipped_count
        
        # Get LLM client for embeddings
        if llm_client is None:
            llm_client = get_llm_client()
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for doc in new_docs:
            ids.append(doc['row_id'])
            texts.append(doc['text'])
            metadatas.append(doc.get('metadata', {}))
        
        # Generate embeddings in batch (more efficient, fewer API calls)
        embeddings = llm_client.embed_texts(texts)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        return len(new_docs), skipped_count
    
    def add_single_document(self, row_id: str, text: str, metadata: Dict, 
                           llm_client=None) -> bool:
        """
        Add a single document to the collection.
        
        Args:
            row_id: Unique identifier for the row
            text: Text content to embed
            metadata: Metadata dictionary
            llm_client: Optional LLMClient instance
            
        Returns:
            True if added, False if already exists
        """
        # Check if already exists
        existing_ids = self.get_existing_row_ids()
        if row_id in existing_ids:
            return False
        
        # Get LLM client
        if llm_client is None:
            llm_client = get_llm_client()
        
        # Generate embedding
        embedding = llm_client.embed_text(text)
        
        # Add to collection
        self.collection.add(
            ids=[row_id],
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding]
        )
        
        return True
    
    def query(self, query_text: str, top_k: int = None, 
              llm_client=None) -> List[Dict]:
        """
        Query the collection for similar documents.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            llm_client: Optional LLMClient instance
            
        Returns:
            List of result dicts with 'row_id', 'text', 'metadata', 'distance'
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        
        # Get LLM client
        if llm_client is None:
            llm_client = get_llm_client()
        
        # Generate query embedding
        query_embedding = llm_client.embed_query(query_text)
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i, row_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'row_id': row_id,
                    'text': results['documents'][0][i] if results['documents'] else '',
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0.0
                })
        
        return formatted_results
    
    def delete_document(self, row_id: str) -> bool:
        """
        Delete a document from the collection.
        
        Args:
            row_id: ID of the document to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            existing_ids = self.get_existing_row_ids()
            if row_id not in existing_ids:
                return False
            
            self.collection.delete(ids=[row_id])
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def delete_documents(self, row_ids: List[str]) -> int:
        """
        Delete multiple documents from the collection.
        
        Args:
            row_ids: List of IDs to delete
            
        Returns:
            Number of documents deleted
        """
        existing_ids = self.get_existing_row_ids()
        ids_to_delete = [rid for rid in row_ids if rid in existing_ids]
        
        if not ids_to_delete:
            return 0
        
        self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    
    def rebuild_collection(self, documents: List[Dict], llm_client=None) -> int:
        """
        Completely rebuild the collection from scratch.
        Deletes all existing documents and re-embeds everything.
        
        Args:
            documents: List of dicts with 'row_id', 'text', 'metadata'
            llm_client: Optional LLMClient instance
            
        Returns:
            Number of documents added
        """
        # Delete the collection and recreate
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # Collection might not exist
        
        # Reset the collection reference
        self._collection = None
        
        if not documents:
            return 0
        
        # Get LLM client
        if llm_client is None:
            llm_client = get_llm_client()
        
        # Prepare data
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            if not doc.get('text', '').strip():
                continue
                
            ids.append(doc['row_id'])
            texts.append(doc['text'])
            metadatas.append(doc.get('metadata', {}))
        
        if not ids:
            return 0
        
        # Generate embeddings in batch (more efficient, fewer API calls)
        embeddings = llm_client.embed_texts(texts)
        
        # Add all documents
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        return len(ids)
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'document_count': count,
                'path': str(config.CHROMA_DIR)
            }
        except Exception as e:
            return {
                'name': self.collection_name,
                'document_count': 0,
                'path': str(config.CHROMA_DIR),
                'error': str(e)
            }
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self._collection = None
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False


# Singleton instance
_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get or create the vector store instance.
    
    Returns:
        VectorStore instance
    """
    global _store_instance
    
    if _store_instance is None:
        _store_instance = VectorStore()
    
    return _store_instance
