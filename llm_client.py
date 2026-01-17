"""
LLM Client - Handles all interactions with Nebius API including:
- Text embeddings (Qwen3-Embedding-8B)
- Chat completions (openai/gpt-oss-120b)
- Rate limit handling with retry logic
"""
import os
from openai import OpenAI
from typing import List, Optional
import time
import random
import config


class LLMClient:
    """Client for interacting with Nebius API (OpenAI-compatible)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            api_key: Optional API key. If not provided, uses NEBIUS_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("NEBIUS_API_KEY", "")
        
        if not self.api_key:
            raise ValueError(
                "Nebius API key is required. Set NEBIUS_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize OpenAI client with Nebius base URL
        self.client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=self.api_key
        )
        
        self.chat_model_name = config.LLM_CHAT_MODEL
        self.embedding_model_name = config.LLM_EMBEDDING_MODEL
        
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
        
        raise RuntimeError(f"Rate limit exceeded after {self.max_retries} retries. Please wait and try again.")
    
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
            response = self.client.embeddings.create(
                model=self.embedding_model_name,
                input=text
            )
            return response.data[0].embedding
        
        try:
            return self._retry_with_backoff(_do_embed)
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {str(e)}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using batch API.
        
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
            response = self.client.embeddings.create(
                model=self.embedding_model_name,
                input=valid_texts
            )
            return [item.embedding for item in response.data]
        
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
        Generate embedding for a query.
        
        Args:
            query: Query text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # For query embedding, we use the same method
        return self.embed_text(query)
    
    def answer_with_llm(self, query: str, context: str) -> str:
        """
        Generate an answer using LLM given a query and context.
        
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
            response = self.client.chat.completions.create(
                model=self.chat_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": config.RAG_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            return response.choices[0].message.content
        
        try:
            return self._retry_with_backoff(_do_generate)
        except Exception as e:
            raise RuntimeError(f"Error generating answer: {str(e)}")
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a general response using LLM.
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            
        Returns:
            Generated response string
        """
        def _do_generate():
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.chat_model_name,
                messages=messages
            )
            return response.choices[0].message.content
        
        try:
            return self._retry_with_backoff(_do_generate)
        except Exception as e:
            raise RuntimeError(f"Error generating response: {str(e)}")


# Singleton instance for convenience
_client_instance: Optional[LLMClient] = None


def get_llm_client(api_key: Optional[str] = None) -> LLMClient:
    """
    Get or create an LLM client instance.
    
    Args:
        api_key: Optional API key
        
    Returns:
        LLMClient instance
    """
    global _client_instance
    
    if _client_instance is None or api_key:
        _client_instance = LLMClient(api_key=api_key)
    
    return _client_instance


def embed_text(text: str, api_key: Optional[str] = None) -> List[float]:
    """Convenience function to embed text."""
    client = get_llm_client(api_key)
    return client.embed_text(text)


def embed_query(query: str, api_key: Optional[str] = None) -> List[float]:
    """Convenience function to embed a query."""
    client = get_llm_client(api_key)
    return client.embed_query(query)


def answer_with_llm(query: str, context: str, api_key: Optional[str] = None) -> str:
    """Convenience function to generate an answer."""
    client = get_llm_client(api_key)
    return client.answer_with_llm(query, context)
