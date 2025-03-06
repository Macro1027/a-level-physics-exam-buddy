"""
Physics Question RAG (Retrieval-Augmented Generation) System
"""

import os
import sys
import logging
import time
import glob
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
from src.utils import ensure_log_directory
log_dir = ensure_log_directory()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "rag.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAG")

# Import Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    logger.error("Pinecone import error. Please install with: pip install pinecone-client")
    sys.exit(1)

# Import embedding provider
from src.pplx import HuggingFaceEmbedding

class PhysicsRAG:
    """
    Physics Question Retrieval-Augmented Generation (RAG) System
    
    This class handles:
    1. Loading and processing physics questions
    2. Creating and managing a vector database using Pinecone
    3. Generating embeddings for questions
    4. Searching for similar questions
    """
    
    def __init__(self, 
                 questions_dir="examples/questions", 
                 pinecone_api_key=None, 
                 pinecone_index_name="physics-questions",
                 embedding_model="jinaai/jina-embeddings-v2-base-en"):
        """
        Initialize the Physics RAG system.
        
        Parameters:
        - questions_dir: Directory containing question files
        - pinecone_api_key: Pinecone API key (defaults to PINECONE_API_KEY environment variable)
        - pinecone_index_name: Name of the Pinecone index
        - embedding_model: Model to use for embeddings
        """
        self.questions_dir = questions_dir
        self.pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        self.pinecone_index_name = pinecone_index_name
        self.embedding_model = embedding_model
        
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not provided. Set PINECONE_API_KEY environment variable or pass pinecone_api_key parameter.")
        
        # Initialize embedding provider
        self.embedding_provider = HuggingFaceEmbedding(model=embedding_model)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Force embedding dimension to 768 for consistency
        embedding_dimension = 768
        logger.info(f"Using embedding dimension: {embedding_dimension}")
        
        # Create or update index
        self._setup_pinecone_index(embedding_dimension)
        
        # Connect to the index
        self.index = self.pc.Index(self.pinecone_index_name)
    
    def _setup_pinecone_index(self, dimension):
        """
        Create or update the Pinecone index.
        
        Parameters:
        - dimension: Dimension of the embeddings
        """
        # Check if index exists
        if self.pinecone_index_name not in [idx.name for idx in self.pc.list_indexes()]:
            logger.info(f"Creating new Pinecone index: {self.pinecone_index_name} with dimension {dimension}")
            self.pc.create_index(
                name=self.pinecone_index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            # Check if the existing index has the right dimension
            try:
                index_info = self.pc.describe_index(self.pinecone_index_name)
                existing_dimension = index_info.dimension
                
                if existing_dimension != dimension:
                    logger.warning(f"Existing index has dimension {existing_dimension}, but embeddings have dimension {dimension}")
                    logger.warning(f"Deleting and recreating index {self.pinecone_index_name}")
                    
                    # Delete the existing index
                    self.pc.delete_index(self.pinecone_index_name)
                    
                    # Create a new index with the correct dimension
                    self.pc.create_index(
                        name=self.pinecone_index_name,
                        dimension=dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
            except Exception as e:
                logger.error(f"Error checking index dimension: {e}")
    
    def get_embedding(self, text):
        """
        Get embedding for a text and ensure it has 768 dimensions.
        
        Parameters:
        - text: Text to embed
        
        Returns:
        - Embedding vector with 768 dimensions
        """
        try:
            # Get the embedding from the provider
            embedding = self.embedding_provider.embed(text)
            
            if not embedding:
                logger.error("Failed to get embedding")
                return None
            
            # Resize to 768 dimensions if needed
            if len(embedding) != 768:
                logger.info(f"Resizing embedding from {len(embedding)} to 768 dimensions")
                
                # If smaller than 768, repeat the vector
                if len(embedding) < 768:
                    repeats = 768 // len(embedding) + 1
                    extended = embedding * repeats
                    embedding = extended[:768]
                # If larger, truncate
                else:
                    embedding = embedding[:768]
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def load_question_file(self, file_path):
        """
        Load a question file and extract its content.
        
        Parameters:
        - file_path: Path to the question file
        
        Returns:
        - Dictionary with question information
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from filename
            filename = os.path.basename(file_path)
            match = os.path.splitext(filename)[0].split('_')
            
            if len(match) >= 4:
                year, level, paper_number, question_number = match
            else:
                # Handle case where filename doesn't match expected format
                year, level, paper_number, question_number = "XX", "XX", "X", "X"
            
            # Split content into question and mark scheme
            parts = content.split("MARK SCHEME:")
            
            if len(parts) > 1:
                question_text = parts[0].strip()
                mark_scheme = parts[1].strip()
            else:
                question_text = content.strip()
                mark_scheme = ""
            
            # Extract marks if available
            marks_match = content.find("MARKS:")
            marks = None
            if marks_match != -1:
                marks_line = content[marks_match:].split("\n")[0]
                try:
                    marks = int(marks_line.replace("MARKS:", "").strip())
                except:
                    pass
            
            return {
                "id": f"{year}_{level}_{paper_number}_{question_number}",
                "year": year,
                "level": level,
                "paper_number": paper_number,
                "question_number": question_number,
                "question_text": question_text,
                "mark_scheme": mark_scheme,
                "marks": marks,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error loading question file {file_path}: {e}")
            return None
    
    def create_database(self):
        """
        Create the RAG database by processing all question files and storing them in Pinecone.
        
        Returns:
        - Number of questions processed
        """
        # Get all question files
        question_files = glob.glob(os.path.join(self.questions_dir, "*.txt"))
        
        if not question_files:
            logger.warning(f"No question files found in {self.questions_dir}")
            return 0
        
        logger.info(f"Found {len(question_files)} question files to process")
        
        # Process each question file
        processed_count = 0
        batch_size = 30  # Process in batches
        
        for i in tqdm(range(0, len(question_files), batch_size), desc="Processing question batches"):
            batch_files = question_files[i:i+batch_size]
            vectors_to_upsert = []
            
            for file_path in batch_files:
                # Load question data
                question_data = self.load_question_file(file_path)
                if not question_data:
                    continue
                
                # Get embeddings for question text
                question_embedding = self.get_embedding(question_data["question_text"])
                if not question_embedding:
                    logger.warning(f"Failed to get embedding for {file_path}")
                    continue
                
                # Prepare metadata - ensure all values are valid for Pinecone
                metadata = {
                    "year": question_data["year"],
                    "level": question_data["level"],
                    "paper_number": question_data["paper_number"],
                    "question_number": question_data["question_number"],
                    "question_text": question_data["question_text"][:1000],  # Truncate for metadata
                    "mark_scheme": question_data["mark_scheme"][:1000] if question_data["mark_scheme"] else "",
                    "file_path": question_data["file_path"]
                }
                
                # Handle marks - convert None to 0
                if question_data["marks"] is not None:
                    metadata["marks"] = question_data["marks"]
                else:
                    metadata["marks"] = 0
                
                # Prepare vector for Pinecone
                vector = {
                    "id": question_data["id"],
                    "values": question_embedding,
                    "metadata": metadata
                }
                
                vectors_to_upsert.append(vector)
            
            # Upsert vectors to Pinecone in smaller batches
            if vectors_to_upsert:
                sub_batch_size = 5
                for j in range(0, len(vectors_to_upsert), sub_batch_size):
                    sub_batch = vectors_to_upsert[j:j+sub_batch_size]
                    try:
                        self.index.upsert(vectors=sub_batch)
                        processed_count += len(sub_batch)
                        logger.info(f"Upserted {len(sub_batch)} vectors to Pinecone")
                        time.sleep(0.5)  # Small delay between sub-batches
                except Exception as e:
                    logger.error(f"Error upserting vectors to Pinecone: {e}")
                        if hasattr(e, 'response') and hasattr(e.response, 'text'):
                            logger.error(f"Response details: {e.response.text}")
            
            # Sleep to avoid rate limits
            time.sleep(1)
        
        logger.info(f"Processed {processed_count} questions in total")
        return processed_count
    
    def search(self, query_text, top_k=5, filter=None):
        """
        Search for similar questions.
        
        Parameters:
        - query_text: The query text
        - top_k: Number of results to return
        - filter: Optional filter for metadata
        
        Returns:
        - List of search results
        """
        # Get embedding for the query
        query_embedding = self.get_embedding(query_text)
        
        if not query_embedding:
            logger.error("Failed to get embedding for query")
            return []
        
        # Search the index
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter
            )
            
            # Process and return results
            processed_results = []
            for match in results.matches:
                processed_results.append({
                    "id": match.id,
                    "score": match.score,
                    "question": match.metadata.get("question_text", ""),
                    "mark_scheme": match.metadata.get("mark_scheme", ""),
                    "year": match.metadata.get("year", ""),
                    "level": match.metadata.get("level", ""),
                    "marks": match.metadata.get("marks", 0)
                })
            
            return processed_results
        
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return []
    
    def answer_with_rag(self, query_text, top_k=3, filter=None, model="sonar-medium-online"):
        """
        Answer a question using RAG (Retrieval-Augmented Generation).
        
        Parameters:
        - query_text: The question to answer
        - top_k: Number of similar questions to retrieve
        - filter: Optional filter for metadata
        - model: Perplexity model to use for generation
        
        Returns:
        - Generated answer with context from similar questions
        """
        # First, search for similar questions
        similar_questions = self.search(query_text, top_k=top_k, filter=filter)
        
        if not similar_questions:
            logger.warning("No similar questions found for RAG")
            return {
                "answer": "I couldn't find any similar questions to help answer this query.",
                "context": [],
                "query": query_text
            }
        
        # Format the similar questions as context
        context = []
        context_text = ""
        
        for i, question in enumerate(similar_questions):
            # Format each question and its mark scheme
            q_text = question["question"]
            ms_text = question["mark_scheme"] if question["mark_scheme"] else "No mark scheme available."
            score = question["score"]
            
            # Add to context list
            context.append({
                "question": q_text,
                "mark_scheme": ms_text,
                "score": score,
                "year": question["year"],
                "level": question["level"]
            })
            
            # Add to context text for the prompt
            context_text += f"\nEXAMPLE {i+1} (Similarity: {score:.2f}):\n"
            context_text += f"Question: {q_text}\n\n"
            context_text += f"Mark Scheme: {ms_text}\n\n"
            context_text += "-" * 50 + "\n"
        
        # Create the prompt for the LLM
        prompt = f"""You are a physics teacher helping students with exam questions.

I'll provide you with a new question and several similar example questions with their mark schemes.
Use these examples to guide your answer to the new question.

SIMILAR EXAMPLES:
{context_text}

NEW QUESTION:
{query_text}

Please provide a detailed answer to the new question, following these guidelines:
1. Use the mark schemes from similar questions as a guide for the level of detail required
2. Include relevant equations and explain how to apply them
3. Show step-by-step working for calculation questions
4. Explain key physics concepts clearly
5. Format your answer in a way that would score full marks in an exam

ANSWER:
"""
        
        try:
            # Initialize Perplexity API
            from src.pplx import PerplexityAPI
            perplexity = PerplexityAPI(model=model)
            
            # Generate the answer
            answer = perplexity.generate(prompt)
            
            return {
                "answer": answer,
                "context": context,
                "query": query_text
            }
        except Exception as e:
            logger.error(f"Error generating answer with RAG: {e}")
                    return {
                "answer": f"Error generating answer: {str(e)}",
                "context": context,
                "query": query_text
            }
    
    def recreate_index(self):
        """
        Delete and recreate the Pinecone index.
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            logger.info(f"Deleting existing index: {self.pinecone_index_name}")
            self.pc.delete_index(self.pinecone_index_name)
            
            logger.info(f"Creating new index: {self.pinecone_index_name} with dimension 768")
            self.pc.create_index(
                name=self.pinecone_index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
            # Reconnect to the index
            self.index = self.pc.Index(self.pinecone_index_name)
            return True
        except Exception as e:
            logger.error(f"Error recreating index: {e}")
            return False
    
    def list_indexes(self):
        """
        List all Pinecone indexes.
        
        Returns:
        - List of index information
        """
        try:
            indexes = self.pc.list_indexes()
            return [{"name": idx.name, "dimension": idx.dimension, "status": idx.status} for idx in indexes]
        except Exception as e:
            logger.error(f"Error listing indexes: {e}")
            return []


# Command-line interface
def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Physics RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create the RAG vector store")
    create_parser.add_argument("--questions_dir", default="examples/questions", help="Directory containing question files")
    create_parser.add_argument("--index_name", default="physics-questions", help="Name of the Pinecone index")
    create_parser.add_argument("--recreate", action="store_true", help="Recreate the index if it already exists")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar questions")
    search_parser.add_argument("query", help="Query text to search for")
    search_parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--filter", help="Filter in JSON format (e.g., '{\"year\": \"19\"}')")
    search_parser.add_argument("--index_name", default="physics-questions", help="Name of the Pinecone index")
    search_parser.add_argument("--show_mark_scheme", action="store_true", help="Show mark schemes in results")
    search_parser.add_argument("--output", help="Output file to save results (JSON format)")
    
    # Answer command
    answer_parser = subparsers.add_parser("answer", help="Generate an answer using RAG")
    answer_parser.add_argument("query", help="Question to answer")
    answer_parser.add_argument("--top_k", type=int, default=3, help="Number of similar questions to use")
    answer_parser.add_argument("--filter", help="Filter in JSON format (e.g., '{\"year\": \"19\"}')")
    answer_parser.add_argument("--index_name", default="physics-questions", help="Name of the Pinecone index")
    answer_parser.add_argument("--model", default="sonar-medium-online", help="Perplexity model to use")
    answer_parser.add_argument("--show_context", action="store_true", help="Show context sources")
    answer_parser.add_argument("--output", help="Output file to save result (JSON format)")
    
    # List indexes command
    subparsers.add_parser("list-indexes", help="List all Pinecone indexes")
    
    return parser.parse_args()

def main():
    """Main entry point for the command-line interface"""
    args = parse_args()
    
    if args.command == "create":
        # Initialize RAG
        rag = PhysicsRAG(
            questions_dir=args.questions_dir,
            pinecone_index_name=args.index_name
        )
        
        # Recreate index if requested
        if args.recreate:
            rag.recreate_index()
        
        # Create the database
        rag.create_database()
    
    elif args.command == "search":
        # Initialize RAG
        rag = PhysicsRAG(pinecone_index_name=args.index_name)
        
        # Parse filter if provided
        filter_dict = None
        if args.filter:
            try:
                filter_dict = json.loads(args.filter)
            except json.JSONDecodeError:
                logger.error(f"Invalid filter JSON: {args.filter}")
                return
        
        # Search for similar questions
        results = rag.search(args.query, top_k=args.top_k, filter=filter_dict)
        
        if not results:
            print("No results found.")
            return
        
        # Print results
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
            print(f"ID: {result['id']}")
            print(f"Year: {result['year']}, Level: {result['level']}, Marks: {result['marks']}")
            print(f"Question: {result['question'][:200]}..." if len(result['question']) > 200 else f"Question: {result['question']}")
            if args.show_mark_scheme and result['mark_scheme']:
                print(f"Mark Scheme: {result['mark_scheme'][:200]}..." if len(result['mark_scheme']) > 200 else f"Mark Scheme: {result['mark_scheme']}")
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
    
    elif args.command == "answer":
        # Initialize RAG
        rag = PhysicsRAG(pinecone_index_name=args.index_name)
        
        # Parse filter if provided
        filter_dict = None
        if args.filter:
            try:
                filter_dict = json.loads(args.filter)
            except json.JSONDecodeError:
                logger.error(f"Invalid filter JSON: {args.filter}")
                return
        
        # Generate answer
        result = rag.answer_with_rag(
            args.query, 
            top_k=args.top_k, 
            filter=filter_dict,
            model=args.model
        )
        
        # Print answer
        print("\n" + "="*50)
        print("QUESTION:")
        print(args.query)
        print("\n" + "="*50)
        print("ANSWER:")
        print(result["answer"])
        print("\n" + "="*50)
        
        # Print context sources if requested
        if args.show_context:
            print("CONTEXT SOURCES:")
            for i, ctx in enumerate(result["context"]):
                print(f"\nSource {i+1} (Score: {ctx['score']:.4f}):")
                print(f"Year: {ctx['year']}, Level: {ctx['level']}")
                print(f"Question: {ctx['question'][:100]}...")
        
        # Save result to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to {args.output}")
    
    elif args.command == "list-indexes":
        # Initialize RAG
        rag = PhysicsRAG()
        
        # List indexes
        indexes = rag.list_indexes()
        
        if not indexes:
            print("No indexes found.")
            return
        
        print(f"Found {len(indexes)} indexes:")
        for idx in indexes:
            print(f"- {idx['name']} (Dimension: {idx['dimension']}, Status: {idx['status']})")
    
    else:
        print("Please specify a command. Use --help for more information.")


# Example usage
if __name__ == "__main__":
    # If run as a script, use the command-line interface
    if len(sys.argv) > 1:
        main() 
    else:
        # Default behavior when imported
        print("Physics RAG System")
        print("Use 'python -m src.RAG <command>' to run commands")
        print("Available commands: create, search, answer, list-indexes") 