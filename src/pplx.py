"""
API client module for embedding and text generation services.
"""

import os
import sys
import requests
import time
import logging
import docx
from hashlib import md5
import numpy as np

# Import constants
from src.constants import (
    DEFAULT_PPLX_API_KEY,
    DEFAULT_PPLX_MODEL,
    GENERATION_PPLX_MODEL,
    PHYSICS_QUESTION_GENERATION_PROMPT,
    PHYSICS_QUESTION_SYSTEM_PROMPT
)

# Set up logging
logger = logging.getLogger("API")

class EmbeddingProvider:
    """Base class for embedding providers"""
    
    def embed(self, text):
        """Generate embeddings for text"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _fallback_embedding(self, text, dimension=1536):
        """
        Generate a deterministic embedding as a fallback method.
        
        Parameters:
        - text: The text to embed
        - dimension: Desired embedding dimension
        
        Returns:
        - Embedding vector (list of floats)
        """
        # Create a deterministic embedding
        embedding = []
        text_bytes = text.encode('utf-8')
        
        # Calculate how many md5 hashes we need
        hash_size = 16  # md5 produces 16 bytes
        num_hashes = (dimension + hash_size - 1) // hash_size  # Ceiling division
        
        for i in range(num_hashes):
            hash_input = text_bytes + str(i).encode('utf-8')
            hash_bytes = md5(hash_input).digest()
            
            for byte in hash_bytes:
                embedding.append((byte / 128.0) - 1.0)
        
        # Truncate to the desired dimension
        embedding = embedding[:dimension]
        
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        return embedding


class HuggingFaceEmbedding(EmbeddingProvider):
    """Client for Hugging Face embedding API using InferenceClient"""
    
    def __init__(self, api_key=None, model="jinaai/jina-embeddings-v2-base-en"):
        """
        Initialize the Hugging Face embedding client.
        
        Parameters:
        - api_key: Hugging Face API key (defaults to HF_API_KEY environment variable)
        - model: Model to use for embeddings
        """
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        self.model = model
        
        if not self.api_key:
            logger.warning("HF_API_KEY not found in environment variables. Will use fallback method if needed.")
    
    def embed(self, text, max_retries=3, retry_delay=10):
        """
        Generate embeddings for text using the Hugging Face InferenceClient.
        
        Parameters:
        - text: The text to embed
        - max_retries: Maximum number of retries on failure
        - retry_delay: Delay between retries in seconds
        
        Returns:
        - Embedding vector (list of floats)
        """
        if not self.api_key:
            logger.warning("No Hugging Face API key available. Using fallback method.")
            return self._fallback_embedding(text)
        
        try:
            # Import the InferenceClient
            from huggingface_hub import InferenceClient
            
            # Create the client
            client = InferenceClient(
                model=self.model,
                token=self.api_key
            )
            
            # Increase backoff for each retry
            current_delay = retry_delay
            
            # Truncate text if it's too long
            truncated_text = text[:4000]  # Reduced limit to avoid overloading the API
            
            for attempt in range(max_retries):
                try:
                    # Use the feature_extraction method for embeddings
                    embedding = None
                    
                    # Try the direct REST API first as it's more reliable
                    logger.info(f"Attempt {attempt+1}: Using direct REST API for embeddings")
                    embedding = self._use_rest_api(truncated_text, 1, 0)  # Single attempt, no delay
                    
                    if embedding is not None:
                        return embedding
                    
                    # If REST API failed, try InferenceClient
                    logger.info(f"Attempt {attempt+1}: Using InferenceClient for embeddings")
                    
                    # Different models have different parameter names
                    try:
                        # First try with 'text' parameter
                        result = client.feature_extraction(text=truncated_text)
                        
                        # Process the result
                        if isinstance(result, np.ndarray):
                            # Convert numpy array to list
                            embedding = result.tolist()
                        elif isinstance(result, list):
                            embedding = result
                        elif hasattr(result, 'embeddings') and isinstance(result.embeddings, list):
                            # Some models return an object with embeddings attribute
                            embedding = result.embeddings[0] if result.embeddings else None
                        
                        if embedding:
                            # If it's a list of lists (batch mode), take the first one
                            if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                                embedding = embedding[0]
                            
                            logger.info(f"Generated embedding with dimension {len(embedding)}")
                            return embedding
                        
                    except TypeError:
                        # Try with 'inputs' parameter
                        try:
                            result = client.feature_extraction(inputs=truncated_text)
                            
                            # Process the result
                            if isinstance(result, np.ndarray):
                                embedding = result.tolist()
                            elif isinstance(result, list):
                                embedding = result
                            elif hasattr(result, 'embeddings') and isinstance(result.embeddings, list):
                                embedding = result.embeddings[0] if result.embeddings else None
                            
                            if embedding:
                                # If it's a list of lists (batch mode), take the first one
                                if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                                    embedding = embedding[0]
                                
                                logger.info(f"Generated embedding with dimension {len(embedding)}")
                                return embedding
                        except:
                            pass
                    
                    # If we got here, both methods failed
                    logger.warning(f"Both InferenceClient and REST API failed on attempt {attempt+1}")
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= 2
                    else:
                        logger.warning("All embedding methods failed. Using fallback method.")
                        return self._fallback_embedding(text)
                    
                except Exception as e:
                    logger.warning(f"Embedding generation failed (attempt {attempt+1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= 2
                    else:
                        return self._fallback_embedding(text)
            
            return self._fallback_embedding(text)
            
        except ImportError:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
            # Fall back to REST API
            return self._use_rest_api(text, max_retries, retry_delay)
        
        except Exception as e:
            logger.error(f"Unexpected error in embed method: {e}")
            return self._fallback_embedding(text)
    
    def _use_rest_api(self, text, max_retries=3, retry_delay=10):
        """
        Fallback method using the REST API directly.
        
        Parameters:
        - text: The text to embed
        - max_retries: Maximum number of retries on failure
        - retry_delay: Delay between retries in seconds
        
        Returns:
        - Embedding vector (list of floats)
        """
        # Use the REST API
        api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Increase backoff for each retry
        current_delay = retry_delay
        
        for attempt in range(max_retries):
            try:
                # Make request to Hugging Face API
                response = requests.post(
                    api_url,
                    headers=headers,
                    json={"inputs": text},
                    timeout=30
                )
                
                # Check for specific error codes
                if response.status_code == 503:
                    logger.warning(f"Hugging Face API is temporarily unavailable (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= 2
                        continue
                    else:
                        return self._fallback_embedding(text)
                
                response.raise_for_status()
                
                # Parse the response
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list):
                    if len(result) > 0:
                        if isinstance(result[0], list):
                            embedding = result[0]
                        else:
                            embedding = result
                        
                        logger.info(f"Generated embedding with dimension {len(embedding)}")
                        return embedding
                
                # If we got here, the format is unexpected
                logger.warning(f"Unexpected response format: {result}")
                if attempt < max_retries - 1:
                    time.sleep(current_delay)
                    current_delay *= 2
                else:
                    return self._fallback_embedding(text)
                
            except Exception as e:
                logger.warning(f"REST API embedding generation failed (attempt {attempt+1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(current_delay)
                    current_delay *= 2
                else:
                    return self._fallback_embedding(text)
        
        return self._fallback_embedding(text)


class PerplexityAPI:
    """Client for the Perplexity API for text generation."""
    
    def __init__(self, api_key=None, model=None):
        """
        Initialize the Perplexity client.
        
        Parameters:
        - api_key: Perplexity API key (defaults to PPLX_API_KEY environment variable or DEFAULT_PPLX_API_KEY)
        - model: Model to use (defaults to DEFAULT_PPLX_MODEL)
        """
        self.api_key = api_key or os.environ.get("PPLX_API_KEY") or DEFAULT_PPLX_API_KEY
        self.model = model or DEFAULT_PPLX_MODEL
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, prompt, system_prompt=None, max_retries=3, retry_delay=5):
        """
        Generate a response from the Perplexity API.
        
        Parameters:
        - prompt: The prompt to send to the API
        - system_prompt: Optional system prompt to set context
        - max_retries: Maximum number of retries on failure
        - retry_delay: Delay between retries in seconds
        
        Returns:
        - Generated text
        """
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,  # Low temperature for more deterministic outputs
            "max_tokens": 4000
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=self.headers, json=data)
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt+1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Giving up.")
                    raise


# For backward compatibility
class Perplexity:
    """
    Legacy class that combines Perplexity API for text generation
    and Hugging Face API for embeddings.
    """
    
    def __init__(self, api_key=None, model=None):
        """
        Initialize the combined client.
        
        Parameters:
        - api_key: Perplexity API key
        - model: Model to use
        """
        self.perplexity = PerplexityAPI(api_key=api_key, model=model)
        self.embedding_provider = HuggingFaceEmbedding()
    
    def generate(self, prompt, system_prompt=None, max_retries=3, retry_delay=5):
        """Proxy to PerplexityAPI.generate"""
        return self.perplexity.generate(prompt, system_prompt, max_retries, retry_delay)
    
    def embed(self, text, max_retries=3, retry_delay=10):
        """Proxy to HuggingFaceEmbedding.embed"""
        return self.embedding_provider.embed(text, max_retries, retry_delay)
    
    def _fallback_embedding(self, text):
        """Proxy to HuggingFaceEmbedding._fallback_embedding"""
        return self.embedding_provider._fallback_embedding(text)


# Utility functions
def process_examples(examples_path):
    """Process the examples file (DOCX) and extract few-shot examples"""
    if not examples_path or not os.path.exists(examples_path):
        return "Examples file not found."
    
    try:
        # Check if it's a DOCX file
        if examples_path.lower().endswith('.docx'):
            # Parse DOCX file
            doc = docx.Document(examples_path)
            # Extract text from paragraphs
            examples_content = "\n".join([para.text for para in doc.paragraphs if para.text])
            return examples_content
        else:
            # Handle as text file
            with open(examples_path, 'r', encoding='utf-8') as file:
                examples_content = file.read()
            return examples_content
    except Exception as e:
        return f"Error reading examples file: {e}"


def process_guide(guide_path):
    """Process the guide.txt file and extract key information"""
    if not guide_path or not os.path.exists(guide_path):
        return "Guide file not found."
    
    try:
        with open(guide_path, 'r', encoding='utf-8') as file:
            guide_content = file.read()
        return guide_content
    except Exception as e:
        return f"Error reading guide file: {e}"


def generate_physics_question(topic, difficulty, question_type, guide_content=None, examples_content=None, rag_examples=None):
    """
    Generate a physics question using the Perplexity API with RAG-enhanced examples.
    
    Parameters:
    - topic: The physics topic
    - difficulty: Difficulty level
    - question_type: Type of question
    - guide_content: Content from the guide file
    - examples_content: Content from the examples file
    - rag_examples: Similar questions retrieved from RAG system
    
    Returns:
    - Generated question
    """
    # Load guide and examples if not provided
    if guide_content is None:
        guide_content = process_guide("guide.txt")
    if examples_content is None:
        examples_content = process_examples("examples.docx")
    
    # If RAG examples are not provided, try to get them
    if rag_examples is None:
        try:
            # Import RAG system
            from src.RAG import PhysicsRAG
            
            # Initialize RAG
            rag = PhysicsRAG()
            
            # Search for similar questions
            search_query = f"{topic} {difficulty} {question_type} physics question"
            rag_examples = rag.search(search_query, top_k=5)
            
            logger.info(f"Retrieved {len(rag_examples)} similar questions using RAG")
        except Exception as e:
            logger.warning(f"Failed to retrieve RAG examples: {e}")
            rag_examples = []
    
    # Format RAG examples for the prompt
    rag_examples_text = ""
    if rag_examples:
        rag_examples_text = "\n\nHere are some similar questions from past exams:\n\n"
        for i, example in enumerate(rag_examples):
            rag_examples_text += f"EXAMPLE {i+1}:\n"
            rag_examples_text += f"Question: {example['question']}\n\n"
            if example['mark_scheme']:
                rag_examples_text += f"Mark Scheme: {example['mark_scheme']}\n\n"
            rag_examples_text += f"Year: {example['year']}, Level: {example['level']}, Marks: {example['marks']}\n"
            rag_examples_text += "-" * 50 + "\n\n"
    
    # Create the prompt with RAG examples
    prompt = PHYSICS_QUESTION_GENERATION_PROMPT.format(
        guide_content=guide_content,
        examples_content=examples_content + rag_examples_text,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type
    )
    
    try:
        # Initialize the Perplexity client
        perplexity = PerplexityAPI(model=GENERATION_PPLX_MODEL)
        
        # Call the API
        response = perplexity.generate(
            prompt=prompt,
            system_prompt=PHYSICS_QUESTION_SYSTEM_PROMPT
        )
        
        # Save the raw output to a temporary file in the logs folder
        from src.utils import ensure_log_directory
        import datetime
        import os
        
        log_dir = ensure_log_directory()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"question_generation_{topic}_{difficulty}_{timestamp}.txt"
        filepath = os.path.join(log_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"TOPIC: {topic}\n")
            f.write(f"DIFFICULTY: {difficulty}\n")
            f.write(f"QUESTION TYPE: {question_type}\n")
            f.write(f"TIMESTAMP: {datetime.datetime.now().isoformat()}\n")
            f.write(f"RAG EXAMPLES USED: {len(rag_examples) if rag_examples else 0}\n")
            f.write("\n" + "="*50 + "\n")
            f.write("RAW GENERATED OUTPUT:\n")
            f.write(response)
        
        logger.info(f"Raw question generation output saved to {filepath}")
        
        # Return the generated question
        return response
    except Exception as e:
        return f"Error generating question: {e}"


# Example usage
if __name__ == "__main__":
    guide_content = process_guide("guide.txt")
    examples_content = process_examples("examples.docx")
    
    # You can call the function with specific parameters
    result = generate_physics_question(
        topic="waves", 
        difficulty="medium", 
        question_type="calculation, short answer, application, long answer",
        guide_content=guide_content,
        examples_content=examples_content
    )
    
    if result:
        print("\n=== GENERATED QUESTION ===\n")
        print(result)
