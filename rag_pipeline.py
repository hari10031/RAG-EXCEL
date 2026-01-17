"""
RAG Pipeline - Orchestrates the retrieval-augmented generation process
- Query processing
- Context retrieval from vector store
- Answer generation with LLM (Nebius)
"""
from typing import List, Dict, Optional, Tuple
import pandas as pd
from vector_store import get_vector_store, VectorStore
from llm_client import get_llm_client, LLMClient
from excel_handler import (
    load_existing_store, 
    dataframe_to_documents,
    get_row_text_for_embedding,
    add_row_ids
)
import config


class RAGPipeline:
    """RAG Pipeline for Excel data Q&A."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None,
                 vector_store: Optional[VectorStore] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            llm_client: Optional LLMClient instance
            vector_store: Optional VectorStore instance
        """
        self.llm_client = llm_client or get_llm_client()
        self.vector_store = vector_store or get_vector_store()
    
    def build_context_from_results(self, results: List[Dict]) -> str:
        """
        Build a context string from retrieved results.
        
        Args:
            results: List of result dicts from vector store query
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result.get('text', '')
            distance = result.get('distance', 0)
            
            # Format each result with its index and relevance
            context_parts.append(f"[Result {i}] (Relevance: {1 - distance:.2f})\n{text}")
        
        return "\n\n".join(context_parts)
    
    def query(self, question: str, top_k: int = None, 
              return_sources: bool = True) -> Dict:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User's question
            top_k: Number of results to retrieve
            return_sources: Whether to include source documents in response
            
        Returns:
            Dictionary with 'answer', 'sources' (optional), 'context'
        """
        if not question or not question.strip():
            return {
                'answer': "Please provide a question.",
                'sources': [],
                'context': ""
            }
        
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        
        # Step 1: Retrieve relevant documents
        try:
            results = self.vector_store.query(
                query_text=question,
                top_k=top_k,
                llm_client=self.llm_client
            )
        except Exception as e:
            return {
                'answer': f"Error retrieving documents: {str(e)}",
                'sources': [],
                'context': ""
            }
        
        # Step 2: Build context
        context = self.build_context_from_results(results)
        
        if not context:
            return {
                'answer': "I don't know based on the uploaded data. No relevant information was found in the dataset.",
                'sources': [],
                'context': ""
            }
        
        # Step 3: Generate answer with LLM
        try:
            answer = self.llm_client.answer_with_llm(question, context)
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': results if return_sources else [],
                'context': context
            }
        
        response = {
            'answer': answer,
            'context': context
        }
        
        if return_sources:
            response['sources'] = results
        
        return response
    
    def update_index_incremental(self, df: Optional[pd.DataFrame] = None) -> Tuple[int, int]:
        """
        Incrementally update the vector index.
        Only embeds and adds rows that don't exist in the index.
        
        Args:
            df: Optional DataFrame to index. If None, loads from Excel store.
            
        Returns:
            Tuple of (added_count, skipped_count)
        """
        # Load data if not provided
        if df is None:
            df = load_existing_store()
            if df is None:
                return 0, 0
        
        # Convert to documents
        documents = dataframe_to_documents(df)
        
        # Add with incremental support
        added, skipped = self.vector_store.add_documents(
            documents=documents,
            llm_client=self.llm_client
        )
        
        return added, skipped
    
    def rebuild_index(self, df: Optional[pd.DataFrame] = None) -> int:
        """
        Completely rebuild the vector index from scratch.
        
        Args:
            df: Optional DataFrame to index. If None, loads from Excel store.
            
        Returns:
            Number of documents indexed
        """
        # Load data if not provided
        if df is None:
            df = load_existing_store()
            if df is None:
                return 0
        
        # Convert to documents
        documents = dataframe_to_documents(df)
        
        # Rebuild collection
        count = self.vector_store.rebuild_collection(
            documents=documents,
            llm_client=self.llm_client
        )
        
        return count
    
    def add_single_row_to_index(self, row_id: str, row_data: pd.Series) -> bool:
        """
        Add a single row to the vector index.
        
        Args:
            row_id: Unique row identifier
            row_data: Row data as pandas Series
            
        Returns:
            True if added, False if already exists
        """
        # Convert row to text
        text = get_row_text_for_embedding(row_data)
        
        if not text.strip():
            return False
        
        # Create metadata
        metadata = {}
        for col, value in row_data.items():
            if pd.notna(value):
                if isinstance(value, (int, float, bool)):
                    metadata[col] = value
                else:
                    metadata[col] = str(value)
        
        # Add to vector store
        return self.vector_store.add_single_document(
            row_id=row_id,
            text=text,
            metadata=metadata,
            llm_client=self.llm_client
        )
    
    def remove_row_from_index(self, row_id: str) -> bool:
        """
        Remove a row from the vector index.
        
        Args:
            row_id: Row ID to remove
            
        Returns:
            True if removed, False if not found
        """
        return self.vector_store.delete_document(row_id)
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the vector index.
        
        Returns:
            Dictionary with index statistics
        """
        return self.vector_store.get_collection_stats()
    
    def sync_index_with_store(self) -> Dict:
        """
        Synchronize the vector index with the Excel store.
        Adds missing rows and removes rows that no longer exist in Excel.
        
        Returns:
            Dictionary with sync statistics
        """
        # Load current data
        df = load_existing_store()
        if df is None:
            return {'added': 0, 'removed': 0, 'error': 'No Excel store found'}
        
        df = add_row_ids(df)
        excel_row_ids = set(df['row_id'].tolist())
        
        # Get existing vector store IDs
        vector_row_ids = self.vector_store.get_existing_row_ids()
        
        # Find rows to add (in Excel but not in vectors)
        rows_to_add = excel_row_ids - vector_row_ids
        
        # Find rows to remove (in vectors but not in Excel)
        rows_to_remove = vector_row_ids - excel_row_ids
        
        # Remove orphaned vectors
        removed_count = 0
        if rows_to_remove:
            removed_count = self.vector_store.delete_documents(list(rows_to_remove))
        
        # Add missing rows
        added_count = 0
        if rows_to_add:
            docs_to_add = dataframe_to_documents(df[df['row_id'].isin(rows_to_add)])
            added_count, _ = self.vector_store.add_documents(
                documents=docs_to_add,
                llm_client=self.llm_client
            )
        
        return {
            'added': added_count,
            'removed': removed_count,
            'total_in_excel': len(excel_row_ids),
            'total_in_vectors': self.vector_store.collection.count()
        }


# Singleton instance
_pipeline_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create the RAG pipeline instance.
    
    Returns:
        RAGPipeline instance
    """
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    
    return _pipeline_instance


def query_excel_data(question: str, top_k: int = None) -> Dict:
    """
    Convenience function to query the Excel data.
    
    Args:
        question: User's question
        top_k: Number of results to retrieve
        
    Returns:
        Query response dictionary
    """
    pipeline = get_rag_pipeline()
    return pipeline.query(question, top_k)
