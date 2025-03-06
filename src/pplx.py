import os
import sys
from openai import OpenAI
import docx  # Add this import
import requests
import time
import logging

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

def generate_physics_question(topic=None, difficulty=None, question_type=None, guide_path=None, guide_content=None, examples_path=None, examples_content=None):
    """
    Generate Edexcel A-level Physics exam-style questions using Perplexity API
    
    Parameters:
    - topic: Physics topic (e.g., "mechanics", "electricity")
    - difficulty: Question difficulty (e.g., "easy", "medium", "hard")
    - question_type: Type of question (e.g., "short-answer", "calculation")
    - guide_path: Path to the guide.txt file
    - guide_content: Content of the guide.txt file
    - examples_path: Path to the examples file
    - examples_content: Content of the examples file
    
    Returns:
    - The generated question with mark scheme
    """
    # Access the API key from environment variables
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    api_key = "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"
    
    # Check if API key exists
    if not api_key:
        api_key = input("Please enter your Perplexity API key: ")
        # Optionally save it for the current session
        os.environ["PERPLEXITY_API_KEY"] = api_key
    
    # Build the prompt with any provided parameters
    prompt = """
    **Context:**  
    You are an expert in creating authentic Edexcel A-level Physics examination questions.

    """
    
    # Add guide content if available
    if guide_content:
        prompt += f"\n\nHere is the guide document detailing the exact structure, style, methodology, and marking principles:\n\n{guide_content}\n\n"
    else:
        prompt += "\n\nI would normally provide a guide document (guide.txt) detailing the exact structure, style, methodology, and marking principles of Edexcel A-level Physics exam questions, but it's not available. Please use your knowledge of Edexcel A-level Physics exam standards.\n\n"
    
    # Add examples content if available
    if examples_content:
        prompt += f"\n\n**Few-Shot Examples:**\nHere are some examples of well-formatted questions and mark schemes to follow:\n\n{examples_content}\n\n"
    
    # Add the rest of the prompt
    prompt += """
**Objective:**  
Generate Edexcel A-level Physics exam-style questions that precisely match the official Edexcel standards described in the attached document.

**Specification Alignment:**  
- Clearly align with the current Edexcel A-level Physics specification.
- Cover a balanced range of specification topics (e.g., mechanics, electricity, waves, thermodynamics, fields, nuclear physics).

**Question Types (User-specified):**  
Generate only the types of questions explicitly requested by the user from the following categories:
- Short-answer questions (1–2 marks; definitions, equations, diagrams).
- Calculation questions (structured clearly with marks allocated for substitution, rearrangement, evaluation).
- Application questions (requiring analysis of physics principles in novel or real-world contexts).
- Extended-response (6-mark) questions (assessing depth of understanding and coherent scientific arguments).

**Mark Scheme Structure:**  
- Provide concise, short bullet points for each marking point.
- Each bullet point must represent exactly one distinct idea, step, or calculation stage.
- Clearly indicate marks awarded for knowledge, application, analysis, calculation steps (substitution/rearrangement/evaluation), and quality of written communication where relevant.
- Follow exactly the style and format demonstrated in the attached guide and examples.

**Mathematical Integration:**  
- Integrate mathematics seamlessly into physics contexts.
- Include relevant mathematical skills such as plotting data graphs, rates of change calculations, areas under curves, uncertainties handling, and use of trigonometric functions.

**Practical Skills Assessment:**  
- Include practical assessment elements where appropriate.
- Frame practical-related questions using phrases like "explain how," "describe how," or "outline a procedure," assessing experimental techniques and data analysis skills.

**Style:**  
- Use clear and precise language consistent with official Edexcel examinations. Follow exactly the formatting conventions demonstrated in your attached document.

**Tone:**  
- Maintain a formal academic tone consistent with official Edexcel examination materials. Ensure clarity and neutrality while challenging students appropriately.

**Audience:**  
- A-level students preparing specifically for Edexcel Physics examinations who have completed their full course.

**Structure:**  
Strictly structure your output exactly as follows:

```
<question> - (question text)
(question 1 description)

<mark scheme>
(question 1 mark scheme)

<question> - (question text)
(question 2 description)

<mark scheme>
(question 2 mark scheme)

<question> - (question text)
(question 3 description)

<mark scheme>
(question 3 mark scheme)
```

Specifically:

1. Clearly state each question with necessary context or diagrams.  
2. Indicate explicit mark allocations clearly for each sub-part.  
3. Provide detailed mark schemes precisely matching the structure and style described in your attached document:
    - For calculation questions: clearly award marks separately for substitution (1 mark), rearrangement (1 mark), evaluation with correct significant figures (1 mark), plus any additional intermediate steps or unit handling as appropriate.
    - For explanation/application/extended-response questions: award marks explicitly for knowledge points made, correct application of concepts to contexts provided, analytical reasoning clearly shown step-by-step, and quality of written communication where applicable.
4. Explicitly reference examples from your attached document when explaining your approach to ensure consistency with official Edexcel standards.

Before generating any new content, carefully analyze your attached document (guide.txt) to fully understand its detailed guidance on question structure, style, marking principles, mathematical integration, practical skills assessment methods, and content focus.
    """

    # Add specific parameters if provided
    if topic:
        prompt += f"\n\nPlease generate a question on the topic of {topic}."
    if difficulty:
        prompt += f"\nThe difficulty level should be {difficulty}."
    if question_type:
        prompt += f"\nPlease make this include {question_type} type questions."
    
    messages = [
        {
            "role": "system",
            "content": "You are an artificial intelligence assistant specializing in creating authentic Edexcel A-level Physics examination questions."
        },
        {   
            "role": "user",
            "content": prompt
        },
    ]
    
    try:
        # Set default headers with the API key
        client = OpenAI(
            api_key=api_key, 
            base_url="https://api.perplexity.ai",
            default_headers={"Authorization": f"Bearer {api_key}"}
        )
    
        # chat completion without streaming
        response = client.chat.completions.create(
            model="sonar-reasoning",  
            messages=messages,
        )
        
        # Return the generated content
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nPossible issues:")
        print("1. The API key may be incorrect")
        print("2. The model name may be incorrect (try 'mistral-7b-instruct' or 'llama-3-8b-instruct')")
        print("3. There might be network connectivity issues")
        print("4. Your Perplexity account may not have API access enabled")
        return None

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

# Example usage
if __name__ == "__main__":
    guide_content = process_guide("guide.txt")
    examples_content = process_examples("examples.docx")  # Changed to .docx
    print(examples_content)
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
class Perplexity:
    """
    Client for the Perplexity API.
    """
    
    def __init__(self, api_key=None, model="r1-1776"):
        """
        Initialize the Perplexity client.
        
        Parameters:
        - api_key: Perplexity API key (defaults to PPLX_API_KEY environment variable)
        - model: Model to use (default: llama-3-sonar-large-32k-online)
        """
        # self.api_key = api_key or os.environ.get("PPLX_API_KEY")
        # if not self.api_key:
        #     raise ValueError("Perplexity API key not provided. Set PPLX_API_KEY environment variable or pass api_key parameter.")
        self.api_key = "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"

        self.model = model
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, prompt, max_retries=3, retry_delay=5):
        """
        Generate a response from the Perplexity API.
        
        Parameters:
        - prompt: The prompt to send to the API
        - max_retries: Maximum number of retries on failure
        - retry_delay: Delay between retries in seconds
        
        Returns:
        - Generated text
        """
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
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
