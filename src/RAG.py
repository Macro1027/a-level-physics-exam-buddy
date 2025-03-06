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

def main():
    """Main function to demonstrate usage"""
    processor = PhysicsExamProcessor()
    
    # Process all files in the examples directory
    processor.process_directory()
    processor.save_to_csv()
    
    # Build search index
    processor.build_search_index()
    
    # Example: Search for questions about momentum
    results = processor.search_similar_questions("Calculate the momentum of a particle", n=3)
    print("\nSearch results for 'momentum':")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Similarity: {result['similarity']:.2f}):")
        print(f"Topic: {result['detected_topic']}")
        print(f"Question: {result['question_text'][:200]}...")
    
    # Example: Get questions by topic
    mechanics_questions = processor.get_questions_by_topic("Mechanics - Forces")
    print(f"\nFound {len(mechanics_questions)} questions on Mechanics - Forces")
    
    # Example: Get a random question
    random_q = processor.get_random_question(topic="Particle Physics")
    if random_q:
        print("\nRandom question on Waves:")
        print(f"Question: {random_q['question_text'][:200]}...")
        print(f"Mark Scheme: {random_q['mark_scheme'][:200]}...")
    
    # Example: Format examples for a prompt
    if results:
        prompt_examples = processor.format_examples_for_prompt(results[:2])
        print("\nFormatted examples for prompt:")
        print(prompt_examples)

if __name__ == "__main__":
    main() 