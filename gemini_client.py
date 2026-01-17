"""
Gemini Client - Handles all interactions with Google Gemini API including:
- Text embeddings
- Chat completions for RAG
- Rate limit handling with retry logic
"""
import google.generativeai as genai
from typing import List, Optional
import time
import random
import config


class GeminiClient:
    """Client for interacting with Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Optional API key. If not provided, uses GEMINI_API_KEY env var.
        """
        self.api_key = api_key or config.GEMINI_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        self.chat_model_name = config.GEMINI_CHAT_MODEL
        self.embedding_model_name = config.GEMINI_EMBEDDING_MODEL
        
        # Initialize models
        self.chat_model = genai.GenerativeModel(self.chat_model_name)
        
        # Rate limiting settings
        self.max_retries = 3
        self.base_delay = 2  # seconds
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff retry on rate limit errors.
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                if '429' in error_str or 'quota' in error_str or 'rate' in error_str:
                    last_exception = e
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit hit. Waiting {delay:.1f}s before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(delay)
                else:
                    raise e
        
        raise RuntimeError(f"Rate limit exceeded after {self.max_retries} retries. Please wait a few minutes and try again.")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        def _do_embed():
            result = genai.embed_content(
                model=f"models/{self.embedding_model_name}",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        
        try:
            return self._retry_with_backoff(_do_embed)
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {str(e)}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using batch API.
        More efficient than calling embed_text multiple times.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter empty texts and track indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            return [[0.0] * config.EMBEDDING_DIMENSION] * len(texts)
        
        def _do_batch_embed():
            # Gemini supports batch embedding
            result = genai.embed_content(
                model=f"models/{self.embedding_model_name}",
                content=valid_texts,
                task_type="retrieval_document"
            )
            return result['embedding']
        
        try:
            batch_embeddings = self._retry_with_backoff(_do_batch_embed)
            
            # Reconstruct full list with zero vectors for empty texts
            embeddings = [[0.0] * config.EMBEDDING_DIMENSION] * len(texts)
            for i, idx in enumerate(valid_indices):
                embeddings[idx] = batch_embeddings[i]
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Error generating batch embeddings: {str(e)}")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query (uses retrieval_query task type).
        
        Args:
            query: Query text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not query or not query.strip():
            raise ValueError("Cannot embed empty query")
        
        def _do_embed():
            result = genai.embed_content(
                model=f"models/{self.embedding_model_name}",
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        
        try:
            return self._retry_with_backoff(_do_embed)
        except Exception as e:
            raise RuntimeError(f"Error generating query embedding: {str(e)}")
    
    def answer_with_gemini(self, query: str, context: str) -> str:
        """
        Generate an answer using Gemini given a query and context.
        
        Args:
            query: User's question
            context: Retrieved context from the vector store
            
        Returns:
            Generated answer string
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Build the prompt
        user_prompt = config.RAG_USER_PROMPT_TEMPLATE.format(
            context=context if context else "No relevant context found.",
            question=query
        )
        
        def _do_generate():
            chat = self.chat_model.start_chat(history=[])
            full_prompt = f"{config.RAG_SYSTEM_PROMPT}\n\n{user_prompt}"
            response = chat.send_message(full_prompt)
            return response.text
        
        try:
            return self._retry_with_backoff(_do_generate)
        except Exception as e:
            raise RuntimeError(f"Error generating answer: {str(e)}")
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a general response using Gemini.
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            
        Returns:
            Generated response string
        """
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.chat_model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Error generating response: {str(e)}")


# Singleton instance for convenience
_client_instance: Optional[GeminiClient] = None


def get_gemini_client(api_key: Optional[str] = None) -> GeminiClient:
    """
    Get or create a Gemini client instance.
    
    Args:
        api_key: Optional API key
        
    Returns:
        GeminiClient instance
    """
    global _client_instance
    
    if _client_instance is None or api_key:
        _client_instance = GeminiClient(api_key=api_key)
    
    return _client_instance


def embed_text(text: str, api_key: Optional[str] = None) -> List[float]:
    """
    Convenience function to embed text.
    
    Args:
        text: Text to embed
        api_key: Optional API key
        
    Returns:
        Embedding vector
    """
    client = get_gemini_client(api_key)
    return client.embed_text(text)


def embed_query(query: str, api_key: Optional[str] = None) -> List[float]:
    """
    Convenience function to embed a query.
    
    Args:
        query: Query to embed
        api_key: Optional API key
        
    Returns:
        Embedding vector
    """
    client = get_gemini_client(api_key)
    return client.embed_query(query)


def answer_with_gemini(query: str, context: str, api_key: Optional[str] = None) -> str:
    """
    Convenience function to generate an answer.
    
    Args:
        query: User's question
        context: Retrieved context
        api_key: Optional API key
        
    Returns:
        Generated answer
    """
    client = get_gemini_client(api_key)
    return client.answer_with_gemini(query, context)
