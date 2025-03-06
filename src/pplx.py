import os
import sys
from openai import OpenAI
import docx  # Add this import
import requests
import time
import logging
from src.constants import (
    DEFAULT_PPLX_API_KEY,
    DEFAULT_PPLX_MODEL,
    GENERATION_PPLX_MODEL,
    PHYSICS_QUESTION_GENERATION_PROMPT,
    PHYSICS_QUESTION_SYSTEM_PROMPT
)

logger = logging.getLogger("Perplexity")

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

def generate_physics_question(topic, difficulty, question_type, guide_content=None, examples_content=None):
    """
    Generate a physics question using the Perplexity API.
    
    Parameters:
    - topic: The physics topic
    - difficulty: Difficulty level
    - question_type: Type of question
    - guide_content: Content from the guide file
    - examples_content: Content from the examples file
    
    Returns:
    - Generated question
    """
    # Load guide and examples if not provided
    if guide_content is None:
        guide_content = process_guide("guide.txt")
    if examples_content is None:
        examples_content = process_examples("examples.docx")
    
    # Create the prompt
    prompt = PHYSICS_QUESTION_GENERATION_PROMPT.format(
        guide_content=guide_content,
        examples_content=examples_content,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type
    )
    
    try:
        # Initialize the Perplexity client
        perplexity = Perplexity(model=GENERATION_PPLX_MODEL)
        
        # Call the API
        response = perplexity.generate(
            prompt=prompt,
            system_prompt=PHYSICS_QUESTION_SYSTEM_PROMPT
        )
        
        # Return the generated question
        return response
    except Exception as e:
        return f"Error generating question: {e}"

class Perplexity:
    """
    Client for the Perplexity API.
    """
    
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
