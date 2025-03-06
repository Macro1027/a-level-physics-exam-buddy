import os
import re
import fitz  # PyMuPDF
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx  # python-docx for .docx files
import glob
import json
import logging
import argparse
import time
from tqdm import tqdm
import pinecone
from pplx import Perplexity
from src.constants import EMBEDDING_GENERATION_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAGProcessor")

class PhysicsExamProcessor:
    """
    Process Edexcel A-Level Physics exam papers, extract questions and mark schemes,
    and organize them by topic for retrieval-augmented generation.
    """
    
    def __init__(self, examples_dir="examples"):
        """Initialize with path to the directory containing exam questions"""
        self.examples_dir = examples_dir
        self.topics = [
            "Mechanics - Motion",
            "Mechanics - Forces",
            "Electricity - Circuits",
            "Electricity - Fields",
            "Waves - Properties",
            "Waves - Optics",
            "Nuclear Physics",
            "Thermodynamics",
            "Particle Physics",
            "Magnetic Fields"
        ]
        self.topic_keywords = {
            "Mechanics - Motion": ["velocity", "acceleration", "displacement", "motion", "kinematics", "projectile", "trajectory"],
            "Mechanics - Forces": ["force", "newton", "momentum", "impulse", "collision", "energy", "work", "power", "conservation"],
            "Electricity - Circuits": ["circuit", "current", "voltage", "resistance", "ohm", "capacitor", "resistor", "kirchhoff"],
            "Electricity - Fields": ["electric field", "charge", "coulomb", "potential", "capacitance"],
            "Waves - Properties": ["wave", "frequency", "wavelength", "amplitude", "interference", "diffraction", "doppler"],
            "Waves - Optics": ["light", "refraction", "reflection", "lens", "mirror", "optical", "polarization"],
            "Nuclear Physics": ["nucleus", "radioactive", "decay", "half-life", "isotope", "radiation", "alpha", "beta", "gamma"],
            "Thermodynamics": ["temperature", "heat", "thermal", "entropy", "gas law", "pressure", "volume"],
            "Particle Physics": ["quark", "lepton", "hadron", "boson", "fermion", "standard model", "particle"],
            "Magnetic Fields": ["magnetic", "field", "flux", "induction", "electromagnet", "solenoid", "fleming"]
        }
        self.questions_db = []
        self.vectorizer = None
        self.question_vectors = None
    
    def extract_text_from_pdf(self, file_path):
        """Extract all text from a PDF file"""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def extract_text_from_docx(self, file_path):
        """Extract all text from a DOCX file"""
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    def extract_text_from_file(self, file_path):
        """Extract text from a file based on its extension"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_path)
        else:
            print(f"Unsupported file type: {ext}")
            return ""
    
    def get_topic_from_filename(self, file_path):
        """Extract topic from filename or infer from content"""
        # Extract filename without extension
        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]
        
        # Try to match filename with known topics
        for topic in self.topics:
            if topic.lower() in file_name.lower():
                return topic
        
        # If no match, return a generic name based on the filename
        return file_name.replace("_", " ").title()
    
    def extract_questions_and_mark_schemes(self, text, topic):
        """Extract questions and their corresponding mark schemes from text"""
        # Clean the text first
        text = self.clean_text(text)
        
        all_questions = []
        
        # Updated pattern to match "Q1." format
        question_pattern = r"(?:Question|Q)[\s]*(\d+)\.?"
        mark_scheme_pattern = r"(?:Mark[\s]*Scheme|MS)[\s]*(\d+)\.?"
        
        # Also look for mark scheme indicators
        ms_indicators = ["Mark scheme", "Marking scheme", "MS", "Answer", "Answers"]
        ms_pattern = "|".join(ms_indicators)
        ms_section_match = re.search(f"({ms_pattern})", text, re.IGNORECASE)
        
        # If we found a mark scheme section, split the text
        if ms_section_match:
            questions_text = text[:ms_section_match.start()]
            mark_schemes_text = text[ms_section_match.start():]
        else:
            # If no mark scheme section found, assume the whole text is questions
            questions_text = text
            mark_schemes_text = ""
        
        # Find all question starts in the questions section
        question_matches = list(re.finditer(question_pattern, questions_text, re.IGNORECASE))
        
        # Extract questions
        questions = []
        for i, match in enumerate(question_matches):
            q_num = match.group(1)
            start_pos = match.end()
            
            # Determine end position (next question or end of content)
            if i < len(question_matches) - 1:
                end_pos = question_matches[i+1].start()
            else:
                end_pos = len(questions_text)
            
            q_text = questions_text[start_pos:end_pos].strip()
            questions.append((q_num, q_text))
        
        # Find all mark scheme starts in the mark schemes section
        mark_scheme_matches = list(re.finditer(question_pattern, mark_schemes_text, re.IGNORECASE))
        
        # Extract mark schemes
        mark_schemes = []
        for i, match in enumerate(mark_scheme_matches):
            ms_num = match.group(1)
            start_pos = match.end()
            
            # Determine end position
            if i < len(mark_scheme_matches) - 1:
                end_pos = mark_scheme_matches[i+1].start()
            else:
                end_pos = len(mark_schemes_text)
            
            ms_text = mark_schemes_text[start_pos:end_pos].strip()
            mark_schemes.append((ms_num, ms_text))
        
        # Match questions with mark schemes
        for q_num, q_text in questions:
            # Find corresponding mark scheme
            ms_text = ""
            for ms_num, ms_content in mark_schemes:
                if ms_num == q_num:
                    ms_text = ms_content
                    break
            
            # Determine the actual topic based on content if not provided
            detected_topic = topic
            if topic == "Unknown":
                detected_topic = self.classify_topic(q_text)
            
            all_questions.append({
                "file_topic": topic,
                "detected_topic": detected_topic,
                "question_number": q_num,
                "question_text": q_text,
                "mark_scheme": ms_text
            })
        
        return all_questions
    
    def classify_topic(self, text):
        """Classify the question into one of the predefined topics based on keywords"""
        text = text.lower()
        scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            scores[topic] = score
        
        # Return the topic with the highest score, or "Unknown" if all scores are 0
        max_score = max(scores.values())
        if max_score == 0:
            return "Unknown"
        
        # Get all topics with the max score
        top_topics = [topic for topic, score in scores.items() if score == max_score]
        return top_topics[0]  # Return the first one in case of ties
    
    def process_directory(self):
        """Process all files in the examples directory"""
        print(f"Processing files in {self.examples_dir}...")
        
        # Get all PDF and DOCX files in the directory
        pdf_files = glob.glob(os.path.join(self.examples_dir, "*.pdf"))
        docx_files = glob.glob(os.path.join(self.examples_dir, "*.docx"))
        all_files = pdf_files + docx_files
        
        if not all_files:
            print(f"No PDF or DOCX files found in {self.examples_dir}")
            return []
        
        # Process each file
        for file_path in all_files:
            print(f"Processing {file_path}...")
            
            # Extract text from the file
            text = self.extract_text_from_file(file_path)
            if not text:
                print(f"Could not extract text from {file_path}")
                continue
            
            # Clean the text
            text = self.clean_text(text)
            
            # Get topic from filename
            topic = self.get_topic_from_filename(file_path)
            
            # Extract questions and mark schemes
            file_questions = self.extract_questions_and_mark_schemes(text, topic)
            
            # Add to the database
            self.questions_db.extend(file_questions)
            
            print(f"Extracted {len(file_questions)} questions from {file_path}")
        
        print(f"Total questions extracted: {len(self.questions_db)}")
        return self.questions_db
    
    def save_to_csv(self, output_path="physics_questions_db.csv"):
        """Save the extracted questions to a CSV file"""
        if not self.questions_db:
            self.process_directory()
        
        df = pd.DataFrame(self.questions_db)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} questions to {output_path}")
        return output_path
    
    def build_search_index(self):
        """Build a search index for the questions using TF-IDF"""
        if not self.questions_db:
            self.process_directory()
        
        if not self.questions_db:
            print("No questions to index.")
            return
        
        # Combine question text and mark scheme for better matching
        texts = [f"{q['question_text']} {q['mark_scheme']}" for q in self.questions_db]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.question_vectors = self.vectorizer.fit_transform(texts)
        
        print(f"Built search index with {len(texts)} documents.")
    
    def search_similar_questions(self, query, topic=None, n=5):
        """
        Search for similar questions based on a query
        
        Parameters:
        - query: The search query
        - topic: Optional topic filter
        - n: Number of results to return
        
        Returns:
        - List of similar questions with similarity scores
        """
        # Clean the query
        query = self.clean_text(query)
        
        if not self.vectorizer:
            self.build_search_index()
        
        if not self.questions_db or not self.vectorizer:
            print("No questions indexed. Run process_directory() and build_search_index() first.")
            return []
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        
        # Get indices of top n similar questions
        if topic:
            # Filter by topic first
            topic_indices = [i for i, q in enumerate(self.questions_db) 
                            if q['detected_topic'] == topic]
            if not topic_indices:
                print(f"No questions found for topic: {topic}")
                return []
            
            topic_similarities = [(i, similarities[i]) for i in topic_indices]
            top_indices = sorted(topic_similarities, key=lambda x: x[1], reverse=True)[:n]
        else:
            top_indices = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:n]
        
        # Return the top n questions with their similarity scores
        results = []
        for idx, score in top_indices:
            question = self.questions_db[idx].copy()
            question['similarity'] = float(score)
            results.append(question)
        
        return results
    
    def get_questions_by_topic(self, topic):
        """Get all questions for a specific topic"""
        if not self.questions_db:
            self.process_directory()
        
        return [q for q in self.questions_db if q['detected_topic'] == topic]
    
    def get_random_question(self, topic=None, difficulty=None):
        """Get a random question, optionally filtered by topic and difficulty"""
        if not self.questions_db:
            self.process_directory()
        
        filtered_questions = self.questions_db
        
        if topic:
            filtered_questions = [q for q in filtered_questions if q['detected_topic'] == topic]
        
        if difficulty:
            # This would require difficulty to be determined or stored
            # For now, we'll just return a random question from the filtered list
            pass
        
        if filtered_questions:
            return np.random.choice(filtered_questions)
        else:
            return None
    
    def format_examples_for_prompt(self, examples, max_length=2000):
        """Format examples for inclusion in a prompt, with length limit"""
        if not examples:
            return ""
        
        formatted_text = "**Few-Shot Examples:**\n\n"
        total_length = len(formatted_text)
        
        for i, example in enumerate(examples):
            q_text = example['question_text'].strip()
            ms_text = example['mark_scheme'].strip()
            
            example_text = f"**Example {i+1}:**\n\n"
            example_text += f"**Question {example['question_number']}:** {q_text}\n\n"
            example_text += f"**Mark Scheme:** {ms_text}\n\n"
            
            # Check if adding this example would exceed the max length
            if total_length + len(example_text) > max_length:
                # If we already have at least one example, stop here
                if i > 0:
                    break
                # Otherwise, truncate this example
                available_space = max_length - total_length - len("... (truncated)")
                example_text = example_text[:available_space] + "... (truncated)"
            
            formatted_text += example_text
            total_length += len(example_text)
        
        return formatted_text
    
    def clean_text(self, text):
        """
        Clean the text by removing special characters and unwanted text
        
        Parameters:
        - text: The text to clean
        
        Returns:
        - Cleaned text
        """
        if not text:
            return ""
        
        # Remove registered trademark symbols, periods, and other special characters
        text = re.sub(r'[®©™\.]', '', text)
        
        # Remove various forms of "PhysicsAndMathsTutor.com"
        patterns = [
            r'Physics\s*And\s*Maths\s*Tutor\s*com',
            r'Physics\s*&\s*Maths\s*Tutor\s*com',
            r'PhysicsAndMathsTutor\s*com',
            r'www\s*physicsandmathstutor\s*com',
            r'physicsandmathstutor\s*com'
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove other common unwanted text
        unwanted_texts = [
            "Edexcel",
            "Pearson",
            "GCE",
            "A Level",
            "AS Level",
            "Turn over",
            "Please turn over",
            "Page \d+ of \d+",
            "Continue on the next page",
            "TOTAL FOR PAPER"
        ]
        
        for unwanted in unwanted_texts:
            text = re.sub(unwanted, '', text, flags=re.IGNORECASE)
        
        # Remove unusual Unicode characters (keeping basic punctuation and symbols)
        text = re.sub(r'[^\x00-\x7F\u2013\u2014\u2018\u2019\u201C\u201D\u2022\u2026\u2032\u2033\u2212\u00B0\u00B1\u00D7\u00F7\u03B1-\u03C9\u0394\u03A9\u03C0\u03BC\u03C3]', '', text)
        
        # Remove standalone periods (not part of numbers or abbreviations)
        # This preserves decimal numbers (e.g., 3.14) and common abbreviations (e.g., e.g., i.e.)
        text = re.sub(r'(?<!\w)\.(?!\w)|(?<=\w)\.(?!\w)|(?<=\s)\.(?=\s)', '', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple newlines with a maximum of two
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

class PhysicsQuestionRAG:
    """
    Create and manage a RAG database for physics questions using Pinecone.
    """
    
    def __init__(self, questions_dir="examples/questions", 
                 pinecone_api_key=None, pinecone_index_name="physics-questions",
                 perplexity_api_key=None, embedding_model="r1-1776"):
        """
        Initialize the RAG processor.
        
        Parameters:
        - questions_dir: Directory containing question files
        - pinecone_api_key: Pinecone API key (defaults to PINECONE_API_KEY environment variable)
        - pinecone_index_name: Name of the Pinecone index
        - perplexity_api_key: Perplexity API key (defaults to PPLX_API_KEY environment variable)
        - embedding_model: Perplexity model to use for embeddings
        """
        self.questions_dir = questions_dir
        self.pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        self.pinecone_index_name = pinecone_index_name
        self.embedding_model = embedding_model
        
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not provided. Set PINECONE_API_KEY environment variable or pass pinecone_api_key parameter.")
        
        # Initialize Perplexity client
        self.perplexity = Perplexity(api_key=perplexity_api_key, model=embedding_model)
        
        # Initialize Pinecone
        pinecone.init(api_key=self.pinecone_api_key, environment="gcp-starter")
        
        # Create index if it doesn't exist
        if self.pinecone_index_name not in pinecone.list_indexes():
            logger.info(f"Creating new Pinecone index: {self.pinecone_index_name}")
            pinecone.create_index(
                name=self.pinecone_index_name,
                dimension=1536,  # Dimension for embeddings
                metric="cosine"
            )
        
        # Connect to the index
        self.index = pinecone.Index(self.pinecone_index_name)
    
    def get_embedding(self, text):
        """
        Get embedding for a text using Perplexity's Sonar model.
        
        Parameters:
        - text: Text to embed
        
        Returns:
        - Embedding vector
        """
        try:
            # Create a prompt for embedding
            prompt = EMBEDDING_GENERATION_PROMPT.format(text=text)
            
            # Call the Perplexity API
            response = self.perplexity.generate(prompt)
            
            # Parse the response to extract the embedding
            # This is a simplified approach - in a real implementation,
            # you would need to use a specific embedding endpoint or parse the response properly
            try:
                # Try to extract a JSON array from the response
                import re
                embedding_match = re.search(r'\[[\d\.\-\,\s]+\]', response)
                if embedding_match:
                    embedding_str = embedding_match.group(0)
                    embedding = json.loads(embedding_str)
                    
                    # Ensure it's the right dimension (1536)
                    if len(embedding) != 1536:
                        # If not, pad or truncate
                        if len(embedding) < 1536:
                            embedding.extend([0.0] * (1536 - len(embedding)))
                        else:
                            embedding = embedding[:1536]
                    
                    return embedding
                else:
                    # If no embedding found, use a simpler approach
                    # Convert the text to a simple vector based on character codes
                    simple_embedding = []
                    for char in text[:1536]:
                        simple_embedding.append(ord(char) / 255.0)  # Normalize to [0,1]
                    
                    # Pad to 1536 dimensions
                    if len(simple_embedding) < 1536:
                        simple_embedding.extend([0.0] * (1536 - len(simple_embedding)))
                    
                    return simple_embedding
            except:
                logger.error("Failed to parse embedding from response")
                return None
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
            if len(parts) == 2:
                question_text = parts[0].replace("QUESTION:", "").strip()
                mark_scheme = parts[1].strip()
            else:
                question_text = content
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
        batch_size = 100  # Process in batches to avoid rate limits
        
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
                
                # Prepare vector for Pinecone
                vector = {
                    "id": question_data["id"],
                    "values": question_embedding,
                    "metadata": {
                        "year": question_data["year"],
                        "level": question_data["level"],
                        "paper_number": question_data["paper_number"],
                        "question_number": question_data["question_number"],
                        "marks": question_data["marks"],
                        "question_text": question_data["question_text"][:1000],  # Truncate for metadata
                        "mark_scheme": question_data["mark_scheme"][:1000],  # Truncate for metadata
                        "file_path": question_data["file_path"]
                    }
                }
                
                vectors_to_upsert.append(vector)
            
            # Upsert vectors to Pinecone
            if vectors_to_upsert:
                try:
                    self.index.upsert(vectors=vectors_to_upsert)
                    processed_count += len(vectors_to_upsert)
                    logger.info(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone")
                except Exception as e:
                    logger.error(f"Error upserting vectors to Pinecone: {e}")
            
            # Sleep to avoid rate limits
            time.sleep(1)
        
        logger.info(f"Processed {processed_count} questions in total")
        return processed_count
    
    def search_similar_questions(self, query, top_k=5, filter_params=None):
        """
        Search for similar questions in the database.
        
        Parameters:
        - query: Query text
        - top_k: Number of results to return
        - filter_params: Dictionary of filter parameters (e.g., {"level": "AS"})
        
        Returns:
        - List of similar questions with metadata
        """
        # Get embedding for query
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            logger.error("Failed to get embedding for query")
            return []
        
        # Prepare filter if provided
        filter_dict = {}
        if filter_params:
            for key, value in filter_params.items():
                if key in ["year", "level", "paper_number", "question_number", "marks"]:
                    filter_dict[key] = value
        
        # Search Pinecone
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []
    
    def get_question_by_id(self, question_id):
        """
        Get a question by its ID.
        
        Parameters:
        - question_id: Question ID
        
        Returns:
        - Question data or None if not found
        """
        try:
            # Fetch from Pinecone
            result = self.index.fetch(ids=[question_id])
            
            if question_id in result.vectors:
                vector = result.vectors[question_id]
                
                # Load full question from file
                file_path = vector.metadata.get("file_path")
                if file_path and os.path.exists(file_path):
                    return self.load_question_file(file_path)
                else:
                    # Return metadata if file not found
                    return {
                        "id": question_id,
                        "year": vector.metadata.get("year"),
                        "level": vector.metadata.get("level"),
                        "paper_number": vector.metadata.get("paper_number"),
                        "question_number": vector.metadata.get("question_number"),
                        "question_text": vector.metadata.get("question_text"),
                        "mark_scheme": vector.metadata.get("mark_scheme"),
                        "marks": vector.metadata.get("marks")
                    }
            else:
                logger.warning(f"Question ID {question_id} not found in database")
                return None
        except Exception as e:
            logger.error(f"Error fetching question {question_id}: {e}")
            return None

def create_rag_database():
    """
    Create a RAG database from the extracted questions.
    """
    logger.info("Creating RAG database from extracted questions")
    rag = PhysicsQuestionRAG(questions_dir="examples/questions")
    processed_count = rag.create_database()
    
    logger.info(f"Created RAG database with {processed_count} questions")
    return processed_count

def main():
    """Main function to create and query the RAG database"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create and query a RAG database for physics questions")
    parser.add_argument("--create", action="store_true", help="Create the RAG database")
    parser.add_argument("--query", type=str, help="Query to search for similar questions")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--filter", type=str, help="Filter parameters in JSON format (e.g., '{\"level\": \"AS\"}')")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Initialize RAG
    rag = PhysicsQuestionRAG(questions_dir="examples/questions")
    
    # Create database if requested
    if args.create:
        processed_count = rag.create_database()
        logger.info(f"Created RAG database with {processed_count} questions")
    
    # Query if requested
    if args.query:
        filter_params = None
        if args.filter:
            try:
                filter_params = json.loads(args.filter)
            except json.JSONDecodeError:
                logger.error(f"Invalid filter JSON: {args.filter}")
        
        results = rag.search_similar_questions(args.query, top_k=args.top_k, filter_params=filter_params)
        
        print(f"\nTop {len(results)} similar questions for query: '{args.query}'")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Score: {result['score']:.4f}")
            print(f"   ID: {result['id']}")
            print(f"   Year: {result['metadata']['year']}, Level: {result['metadata']['level']}")
            print(f"   Question: {result['metadata']['question_text'][:200]}...")
    
    end_time = time.time()
    
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 