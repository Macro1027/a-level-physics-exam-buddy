import os
import glob
import re
import shutil
import argparse
import time
import logging
import json
from tqdm import tqdm
import fitz  # PyMuPDF
from pplx import Perplexity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ExamProcessor")

class ExamPaperProcessor:
    """
    Process and rename exam papers according to specific rules.
    """
    
    def __init__(self, input_dir="raw examples", output_dir="examples/papers"):
        """
        Initialize the exam paper processor.
        
        Parameters:
        - input_dir: Directory containing raw PDF files
        - output_dir: Directory to save processed files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_year_paper_info(self, filename):
        """
        Extract year and paper number from filename with fuzzy matching.
        
        Parameters:
        - filename: Original filename
        
        Returns:
        - Tuple of (year, paper_number, is_ms, level)
        """
        # Check if it's a mark scheme (allow for variations like "mark scheme", "markscheme", "ms", etc.)
        is_ms = any(marker in filename.lower() for marker in ["ms", "mark scheme", "markscheme", "marking scheme"])
        
        # Extract year (YYYY format) - look for any 4-digit number starting with 20
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            year = year_match.group(1)
            # Convert to YY format
            year = year[-2:]
        else:
            # Try to find a 2-digit year
            year_match = re.search(r'(?<!\d)(\d{2})(?!\d)', filename)
            if year_match and int(year_match.group(1)) > 10:  # Assume years after 2010
                year = year_match.group(1)
            else:
                year = "XX"  # Unknown year
        
        # Extract paper number with more flexible pattern matching
        # Look for patterns like "Paper 1", "Paper1", "P1", "Paper 2A", etc.
        paper_patterns = [
            r'Paper\s*(\d[A-Z]?)',  # Paper 1, Paper1, Paper 2A
            r'P\s*(\d[A-Z]?)',      # P1, P 2A
            r'Paper\s*Number\s*(\d[A-Z]?)',  # Paper Number 1
            r'(?<![A-Za-z])(\d[A-Z]?)(?:\s*Paper)',  # 1 Paper, 2A Paper
        ]
        
        for pattern in paper_patterns:
            paper_match = re.search(pattern, filename, re.IGNORECASE)
            if paper_match:
                paper_number = paper_match.group(1)
                break
        else:
            # If no match found, try to find any standalone digit
            digit_match = re.search(r'(?<!\d)(\d)(?!\d)', filename)
            if digit_match:
                paper_number = digit_match.group(1)
            else:
                paper_number = "X"  # Unknown paper
        
        # Determine if it's AS or A2 level with fuzzy matching
        if "as" in filename.lower() or "as-level" in filename.lower() or "as level" in filename.lower():
            level = "AS"
        elif "a2" in filename.lower() or "a2-level" in filename.lower() or "a2 level" in filename.lower():
            level = "A2"
        elif "a-level" in filename.lower() or "a level" in filename.lower():
            # If just "A-level" is mentioned, default to AS unless specified otherwise
            level = "AS"
        else:
            # Default to AS if unclear
            level = "AS"
        
        return year, paper_number, is_ms, level
    
    def get_new_filename(self, original_filename):
        """
        Generate new filename based on original filename with improved robustness.
        
        Parameters:
        - original_filename: Original filename
        
        Returns:
        - New filename
        """
        # Get just the filename without path
        filename = os.path.basename(original_filename)
        
        # Check if it's a specimen paper
        if "specimen" in filename.lower() or "sample" in filename.lower():
            logger.info(f"Skipping specimen paper: {filename}")
            return None  # Skip specimen papers
        
        # Extract information with fuzzy matching
        year, paper_number, is_ms, level = self.extract_year_paper_info(filename)
        
        # Log the extracted information for debugging
        logger.debug(f"Extracted from '{filename}': Year={year}, Paper={paper_number}, MS={is_ms}, Level={level}")
        
        # Create new filename
        paper_type = "MS" if is_ms else "QP"
        new_filename = f"{year}_{level}_{paper_number}_{paper_type}.pdf"
        
        return new_filename
    
    def process_pdf(self, pdf_path):
        """
        Process a PDF file: rename and remove unwanted pages.
        
        Parameters:
        - pdf_path: Path to the PDF file
        
        Returns:
        - Path to the processed file or None if skipped
        """
        try:
            # Get the original filename
            original_filename = os.path.basename(pdf_path)
            
            # Generate new filename
            new_filename = self.get_new_filename(original_filename)
            if not new_filename:
                logger.info(f"Skipping file: {original_filename}")
                return None
            
            # Create output path
            output_path = os.path.join(self.output_dir, new_filename)
            
            # Check if the file already exists
            if os.path.exists(output_path):
                logger.info(f"File already exists: {output_path}")
                return output_path
            
            # Extract information to determine pages to remove
            year, paper_number, is_ms, level = self.extract_year_paper_info(original_filename)
            
            # Open the PDF
            pdf_document = fitz.open(pdf_path)
            
            # Determine how many pages to skip
            if is_ms:
                # For mark schemes
                if int(year) >= 17:  # 2017 and beyond
                    pages_to_skip = 5
                else:  # 2016 and earlier
                    pages_to_skip = 4
            else:
                # For question papers
                pages_to_skip = 1
            
            # Create a new PDF with skipped pages
            new_pdf = fitz.open()
            for i in range(pages_to_skip, pdf_document.page_count):
                new_pdf.insert_pdf(pdf_document, from_page=i, to_page=i)
            
            # Save the new PDF
            new_pdf.save(output_path)
            new_pdf.close()
            pdf_document.close()
            
            logger.info(f"Processed: {original_filename} -> {new_filename} (removed first {pages_to_skip} pages)")
            
            return output_path
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return None
    
    def process_directory(self):
        """
        Process all PDF files in the input directory.
        
        Returns:
        - List of paths to processed files
        """
        # Get all PDF files in the input directory
        pdf_files = glob.glob(os.path.join(self.input_dir, "*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF file
        processed_files = []
        for pdf_path in pdf_files:
            output_path = self.process_pdf(pdf_path)
            if output_path:
                processed_files.append(output_path)
        
        logger.info(f"Processed {len(processed_files)} files")
        return processed_files


class QuestionExtractor:
    """
    Extract question-mark scheme pairs from processed exam papers.
    """
    
    def __init__(self, examples_dir="examples/papers", output_dir="examples/questions", 
                 progress_file="extraction_progress.json", use_llm=False):
        """
        Initialize the question extractor.
        
        Parameters:
        - examples_dir: Directory containing processed exam papers
        - output_dir: Directory to save extracted questions
        - progress_file: File to track extraction progress
        - use_llm: Whether to use LLM for extraction
        """
        self.examples_dir = examples_dir
        self.output_dir = output_dir
        self.progress_file = progress_file
        self.use_llm = use_llm
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Perplexity client if using LLM
        if self.use_llm:
            try:
                self.perplexity = Perplexity()
                # Load progress if it exists
                self.progress = self.load_progress()
            except Exception as e:
                logger.error(f"Error initializing Perplexity client: {e}")
                self.use_llm = False
    
    def load_progress(self):
        """
        Load extraction progress from file.
        
        Returns:
        - Dictionary of completed papers
        """
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading progress file: {e}")
        
        return {"completed_papers": []}
    
    def save_progress(self):
        """
        Save extraction progress to file.
        """
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f)
        except Exception as e:
            logger.error(f"Error saving progress file: {e}")
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file and clean it.
        
        Parameters:
        - pdf_path: Path to the PDF file
        
        Returns:
        - Extracted and cleaned text
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Clean the text to remove consecutive periods
            cleaned_text = self.clean_text_for_llm(text)
            
            return cleaned_text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text):
        """
        Clean the extracted text by removing unwanted elements.
        
        Parameters:
        - text: Text to clean
        
        Returns:
        - Cleaned text
        """
        # Instead of removing all periods, only remove standalone periods
        # This preserves decimal numbers and abbreviations
        text = re.sub(r'(?<!\w)\.(?!\w)|(?<=\s)\.(?=\s)', '', text)
        
        # Remove 'DO NOT WRITE IN THIS AREA' and 'BLANK PAGE' sentences
        text = re.sub(r'DO NOT WRITE IN THIS AREA.*?(\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'BLANK PAGE.*?(\n|$)', '', text, flags=re.IGNORECASE)
        
        # Ignore barcodes (typically appear as patterns like P12345A or similar)
        text = re.sub(r'(?<!\w)[A-Z]\d{5}[A-Z](?!\w)', '', text)
        
        # Remove other common unwanted text in exam papers
        # Use word boundaries to avoid removing parts of legitimate text
        unwanted_texts = [
            r'\bTurn over\b',
            r'\bTOTAL FOR PAPER\b',
            r'\bPearson Edexcel\b',
            r'\bEdexcel\b',
            r'\bGCE\b',
            r'\bAdvanced Level\b',
            r'\bAdvanced Subsidiary\b',
            r'\bPhysics\b',
            r'© 20\d\d Pearson Education Ltd',
            r'© Pearson Education Limited 20\d\d',
            r'Pearson Education Limited',
            r'www\.edexcel\.com',
        ]
        
        for pattern in unwanted_texts:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # More carefully remove paper codes
        text = re.sub(r'(?<!\w)P\d{5}[A-Z](?!\w)', '', text)  # Paper codes like P43567A
        text = re.sub(r'(?<!\w)\*P\d{5}[A-Z]\*(?!\w)', '', text)  # Paper codes with asterisks
        
        # Clean up excessive whitespace but preserve paragraph structure
        text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
        
        return text.strip()
    
    def truncate_at_data_section(self, text):
        """
        Truncate text at 'List of data, formulae and relationships' section.
        
        Parameters:
        - text: Text to truncate
        
        Returns:
        - Truncated text
        """
        # 4. Stop processing at 'List of data, formulae and relationships'
        data_section_patterns = [
            r'List of data, formulae and relationships',
            r'Data, Formulae and Relationships Booklet',
            r'FORMULAE SHEET',
            r'FORMULA SHEET',
            r'FORMULAE AND DATA SHEET',
            r'PHYSICS DATA SHEET'
        ]
        
        for pattern in data_section_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Truncate the text at this point
                return text[:match.start()]
        
        return text
    
    def extract_marks_from_text(self, text):
        """
        Extract the number of marks from text.
        
        Parameters:
        - text: Text to extract marks from
        
        Returns:
        - Number of marks or None if not found
        """
        # Look for patterns like "Total for Question X = Y marks"
        # Handle both with and without periods
        marks_patterns = [
            r"Total for Question \d+\s*=\s*(\d+)\s*marks?",  # Without period
            r"Total for Question \d+\s*=\s*(\d+)\s*marks?\.",  # With period
            r"\(Total for Question \d+\s*=\s*(\d+)\s*marks?\)",  # In parentheses
            r"\(Total:\s*(\d+)\s*marks?\)"  # Simplified format
        ]
        
        for pattern in marks_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def extract_questions_from_qp(self, qp_text):
        """
        Extract questions from question paper text using a sliding window approach
        with a running counter that only updates when "Total for Question X" is seen.
        
        Parameters:
        - qp_text: Question paper text
        
        Returns:
        - List of (question_number, question_text, marks) tuples
        """
        # Clean and truncate the text first
        qp_text = self.clean_text(qp_text)
        qp_text = self.truncate_at_data_section(qp_text)
        
        questions = []
        current_text = ""
        current_question = "1"  # Start with question 1
        
        # Split the text into lines for processing
        lines = qp_text.split('\n')
        
        # Pattern to match question numbers at the start of a line
        question_start_pattern = r"^(\d+)[\.\s]+(.+)$"
        
        # Pattern to match "Total for Question X = Y marks"
        total_pattern = r".*Total for Question (\d+)\s*=\s*(\d+)\s*marks?.*"
        
        for line in lines:
            line = line.strip()
            if not line:
                # Add empty line to current question text if we're tracking a question
                if current_question:
                    current_text += "\n\n"
                continue
            
            # Check if this line contains a "Total for Question X" marker
            total_match = re.match(total_pattern, line, re.IGNORECASE)
            if total_match:
                question_num = total_match.group(1)
                marks = int(total_match.group(2))
                
                # Add this line to the current text
                current_text += line + "\n"
                
                # If this total marker matches our current question
                if question_num == current_question:
                    # Commit this question
                    questions.append((current_question, current_text.strip(), marks))
                    
                    # Move to the next question
                    current_question = str(int(current_question) + 1)
                    current_text = ""
                continue
            
            # Check if this line starts a new question
            question_match = re.match(question_start_pattern, line)
            if question_match:
                question_num = question_match.group(1)
                question_line = question_match.group(2)
                
                # Only process valid question numbers (1 and above)
                if question_num.isdigit() and int(question_num) > 0:
                    # If this is a new question number
                    if question_num != current_question:
                        # If we have content for the current question, commit it
                        if current_text and len(current_text.strip()) > 10:
                            # Try to extract marks from the current text
                            extracted_marks = self.extract_marks_from_text(current_text)
                            questions.append((current_question, current_text.strip(), extracted_marks))
                        
                        # Update to the new question number
                        current_question = question_num
                        current_text = line + "\n"
                    else:
                        # This is still part of the current question
                        current_text += line + "\n"
            else:
                # Add this line to the current question text
                current_text += line + "\n"
        
        # Handle the last question if there is one
        if current_text and len(current_text.strip()) > 10:
            # Try to extract marks from the current text
            extracted_marks = self.extract_marks_from_text(current_text)
            questions.append((current_question, current_text.strip(), extracted_marks))
        
        # Post-process: merge any fragments of the same question
        merged_questions = {}
        for q_num, q_text, q_marks in questions:
            if q_num not in merged_questions:
                merged_questions[q_num] = {"text": q_text, "marks": q_marks}
            else:
                # Append text
                merged_questions[q_num]["text"] += "\n\n" + q_text
                # Use the marks if we didn't have them before
                if merged_questions[q_num]["marks"] is None and q_marks is not None:
                    merged_questions[q_num]["marks"] = q_marks
        
        # Convert back to list format
        result = [(q_num, data["text"], data["marks"]) 
                  for q_num, data in merged_questions.items()]
        
        # Sort by question number
        result.sort(key=lambda x: int(x[0]))
        
        return result
    
    def extract_mark_schemes_from_ms(self, ms_text):
        """
        Extract mark schemes from mark scheme text using a sliding window approach
        with a running counter that only updates when "Total for Question X" is seen.
        
        Parameters:
        - ms_text: Mark scheme text
        
        Returns:
        - List of (question_number, mark_scheme_text, marks) tuples
        """
        # Clean the text first
        ms_text = self.clean_text(ms_text)
        
        mark_schemes = []
        current_text = ""
        current_question = "1"  # Start with question 1
        
        # Split the text into lines for processing
        lines = ms_text.split('\n')
        
        # Pattern to match question numbers at the start of a line
        # This handles both simple numbers and numbers with subparts like 1(a) or 1(a)(i)
        question_start_pattern = r"^(\d+(?:\([a-z]+\)(?:\([ivx]+\))?)?)[\.\s:]+(.+)$"
        
        # Pattern to match "Total for Question X = Y marks"
        total_pattern = r".*Total for Question (\d+)\s*=\s*(\d+)\s*marks?.*"
        
        for line in lines:
            line = line.strip()
            if not line:
                # Add empty line to current mark scheme text if we're tracking one
                if current_question:
                    current_text += "\n\n"
                continue
            
            # Check if this line contains a "Total for Question X" marker
            total_match = re.match(total_pattern, line, re.IGNORECASE)
            if total_match:
                question_num = total_match.group(1)
                marks = int(total_match.group(2))
                
                # Add this line to the current text
                current_text += line + "\n"
                
                # If this total marker matches our current question
                if question_num == current_question:
                    # Commit this mark scheme
                    mark_schemes.append((current_question, current_text.strip(), marks))
                    
                    # Move to the next question
                    current_question = str(int(current_question) + 1)
                    current_text = ""
                continue
            
            # Check if this line starts a new mark scheme item
            question_match = re.match(question_start_pattern, line)
            if question_match:
                question_num = question_match.group(1)
                question_line = question_match.group(2)
                
                # Extract the main question number
                main_question_number = re.match(r"(\d+)", question_num).group(1)
                
                # Only process valid question numbers (1 and above)
                if main_question_number.isdigit() and int(main_question_number) > 0:
                    # If this is a new main question
                    if main_question_number != current_question:
                        # If we have content for the current question, commit it
                        if current_text and len(current_text.strip()) > 10:
                            # Try to extract marks from the current text
                            extracted_marks = self.extract_marks_from_text(current_text)
                            mark_schemes.append((current_question, current_text.strip(), extracted_marks))
                        
                        # Update to the new question number
                        current_question = main_question_number
                        current_text = line + "\n"
                    else:
                        # This is still part of the current question
                        current_text += line + "\n"
            else:
                # Add this line to the current mark scheme text
                current_text += line + "\n"
        
        # Handle the last mark scheme if there is one
        if current_text and len(current_text.strip()) > 10:
            # Try to extract marks from the current text
            extracted_marks = self.extract_marks_from_text(current_text)
            mark_schemes.append((current_question, current_text.strip(), extracted_marks))
        
        # Post-process: merge any fragments of the same question
        merged_mark_schemes = {}
        for q_num, ms_text, ms_marks in mark_schemes:
            if q_num not in merged_mark_schemes:
                merged_mark_schemes[q_num] = {"text": ms_text, "marks": ms_marks}
            else:
                # Append text
                merged_mark_schemes[q_num]["text"] += "\n\n" + ms_text
                # Use the marks if we didn't have them before
                if merged_mark_schemes[q_num]["marks"] is None and ms_marks is not None:
                    merged_mark_schemes[q_num]["marks"] = ms_marks
        
        # Convert back to list format
        result = [(q_num, data["text"], data["marks"]) 
                  for q_num, data in merged_mark_schemes.items()]
        
        # Sort by question number
        result.sort(key=lambda x: int(x[0]))
        
        return result
    
    def match_questions_with_mark_schemes(self, questions, mark_schemes):
        """
        Match questions with their corresponding mark schemes.
        
        Parameters:
        - questions: List of (question_number, question_text, marks) tuples
        - mark_schemes: List of (question_number, mark_scheme_text, marks) tuples
        
        Returns:
        - List of (question_number, question_text, mark_scheme_text, marks) tuples
        """
        # Group mark schemes by question number
        ms_by_question = {}
        for ms_num, ms_text, ms_marks in mark_schemes:
            if ms_num not in ms_by_question:
                ms_by_question[ms_num] = []
            ms_by_question[ms_num].append((ms_text, ms_marks))
        
        # Match questions with mark schemes
        matched_pairs = []
        
        # Ensure we're only dealing with valid question numbers (1 and above)
        valid_questions = [(q_num, q_text, q_marks) for q_num, q_text, q_marks in questions 
                           if q_num.isdigit() and int(q_num) > 0]
        
        for q_num, q_text, q_marks in valid_questions:
            if q_num in ms_by_question:
                # Get all mark schemes for this question
                ms_entries = ms_by_question[q_num]
                
                # Combine all mark scheme text for this question
                combined_ms_text = "\n\n".join([ms_text for ms_text, _ in ms_entries])
                
                # Get the total marks from the mark schemes
                ms_total_marks = sum([ms_marks for _, ms_marks in ms_entries if ms_marks is not None])
                
                # Validate by comparing marks
                if q_marks is None or ms_total_marks == 0 or q_marks == ms_total_marks:
                    matched_pairs.append((q_num, q_text, combined_ms_text, q_marks or ms_total_marks))
                else:
                    logger.warning(f"Mark mismatch for question {q_num}: QP={q_marks}, MS={ms_total_marks}")
                    # Still add it, but with a warning
                    matched_pairs.append((q_num, q_text, combined_ms_text, q_marks or ms_total_marks))
        
        return matched_pairs
    
    def save_question_pair(self, question_number, question_text, mark_scheme_text, marks, paper_info):
        """
        Save a question-mark scheme pair to a file.
        
        Parameters:
        - question_number: Question number
        - question_text: Question text
        - mark_scheme_text: Mark scheme text
        - marks: Number of marks
        - paper_info: Paper information (year, level, paper_number)
        
        Returns:
        - Path to the saved file
        """
        year, level, paper_number = paper_info
        
        # Create filename
        filename = f"{year}_{level}_{paper_number}_Q{question_number}.txt"
        file_path = os.path.join(self.output_dir, filename)
        
        # Format the content
        content = f"Question {question_number} ({marks} marks)\n"
        content += "=" * 50 + "\n\n"
        content += question_text + "\n\n"
        content += "Mark Scheme\n"
        content += "=" * 50 + "\n\n"
        content += mark_scheme_text
        
        # Save to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return file_path
    
    def find_paper_pairs(self):
        """
        Find question paper and mark scheme pairs in the examples directory.
        
        Returns:
        - List of (qp_path, ms_path) tuples
        """
        # Get all PDFs in the examples directory
        pdf_files = glob.glob(os.path.join(self.examples_dir, "*.pdf"))
        
        # Group by year, level, and paper number
        paper_groups = {}
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            match = re.match(r"(\d+)_([A-Z]+)_(\d[A-Z]?)_([A-Z]+)", filename)
            if match:
                year, level, paper_number, paper_type = match.groups()
                key = (year, level, paper_number)
                if key not in paper_groups:
                    paper_groups[key] = {"QP": None, "MS": None}
                paper_groups[key][paper_type] = pdf_path
        
        # Find pairs where both QP and MS exist
        pairs = []
        for key, paths in paper_groups.items():
            if paths["QP"] and paths["MS"]:
                pairs.append((paths["QP"], paths["MS"]))
            else:
                logger.warning(f"Incomplete pair for {key}: QP={paths['QP']}, MS={paths['MS']}")
        
        return pairs
    
    def process_all_papers_batch(self):
        """
        Process all question paper and mark scheme pairs in the examples directory using batch processing.
        First extracts all questions, then all mark schemes, then combines them.
        
        Returns:
        - List of paths to saved question-mark scheme pairs
        """
        # Find paper pairs
        pairs = self.find_paper_pairs()
        
        logger.info(f"Found {len(pairs)} question paper and mark scheme pairs")
        
        # Step 1: Extract all questions
        all_questions = {}  # Dictionary to store questions by paper key
        for qp_path, _ in pairs:
            paper_key, questions = self.extract_questions_from_paper(qp_path)
            if paper_key:
                all_questions[paper_key] = questions
        
        logger.info(f"Extracted questions from {len(all_questions)} papers")
        
        # Step 2: Extract all mark schemes
        all_mark_schemes = {}  # Dictionary to store mark schemes by paper key
        for _, ms_path in pairs:
            paper_key, mark_schemes = self.extract_mark_schemes_from_paper(ms_path)
            if paper_key:
                all_mark_schemes[paper_key] = mark_schemes
        
        logger.info(f"Extracted mark schemes from {len(all_mark_schemes)} papers")
        
        # Step 3: Combine questions and mark schemes into matching pairs
        all_saved_files = []
        
        for paper_key in all_questions.keys():
            if paper_key not in all_mark_schemes:
                logger.warning(f"No mark schemes found for paper {paper_key}")
                continue
            
            year, level, paper_number = paper_key
            questions = all_questions[paper_key]
            mark_schemes = all_mark_schemes[paper_key]
            
            # Match questions with mark schemes
            matched_pairs = self.match_questions_with_mark_schemes(questions, mark_schemes)
            
            logger.info(f"Paper {year} {level} {paper_number}: Matched {len(matched_pairs)} question-mark scheme pairs")
            
            # Save each question-mark scheme pair
            paper_info = (year, level, paper_number)
            for q_num, q_text, ms_text, marks in matched_pairs:
                file_path = self.save_question_pair(q_num, q_text, ms_text, marks, paper_info)
                all_saved_files.append(file_path)
        
        logger.info(f"Saved {len(all_saved_files)} question-mark scheme pairs in total")
        
        return all_saved_files

    def extract_questions_from_paper(self, qp_path):
        """
        Extract questions from a question paper.
        
        Parameters:
        - qp_path: Path to the question paper
        
        Returns:
        - Tuple of (paper_key, questions) where questions is a list of (question_number, question_text, marks) tuples
        """
        # Extract paper information from filename
        qp_filename = os.path.basename(qp_path)
        match = re.match(r"(\d+)_([A-Z]+)_(\d[A-Z]?)_QP", qp_filename)
        if not match:
            logger.warning(f"Could not extract paper information from {qp_filename}")
            return None, []
        
        year, level, paper_number = match.groups()
        paper_key = (year, level, paper_number)
        
        # Extract text from PDF
        qp_text = self.extract_text_from_pdf(qp_path)
        
        if not qp_text:
            logger.error(f"Failed to extract text from {qp_path}")
            return paper_key, []
        
        # Extract questions
        questions = self.extract_questions_from_qp(qp_text)
        
        logger.info(f"Paper {year} {level} {paper_number}: Extracted {len(questions)} questions")
        
        return paper_key, questions

    def extract_mark_schemes_from_paper(self, ms_path):
        """
        Extract mark schemes from a mark scheme paper.
        
        Parameters:
        - ms_path: Path to the mark scheme
        
        Returns:
        - Tuple of (paper_key, mark_schemes) where mark_schemes is a list of (question_number, mark_scheme_text, marks) tuples
        """
        # Extract paper information from filename
        ms_filename = os.path.basename(ms_path)
        match = re.match(r"(\d+)_([A-Z]+)_(\d[A-Z]?)_MS", ms_filename)
        if not match:
            logger.warning(f"Could not extract paper information from {ms_filename}")
            return None, []
        
        year, level, paper_number = match.groups()
        paper_key = (year, level, paper_number)
        
        # Extract text from PDF
        ms_text = self.extract_text_from_pdf(ms_path)
        
        if not ms_text:
            logger.error(f"Failed to extract text from {ms_path}")
            return paper_key, []
        
        # Extract mark schemes
        mark_schemes = self.extract_mark_schemes_from_ms(ms_text)
        
        logger.info(f"Paper {year} {level} {paper_number}: Extracted {len(mark_schemes)} mark schemes")
        
        return paper_key, mark_schemes

    def extract_questions_with_llm(self, text, is_mark_scheme=False):
        """
        Extract questions or mark schemes using the Perplexity LLM.
        
        Parameters:
        - text: Text to extract from
        - is_mark_scheme: Whether the text is from a mark scheme
        
        Returns:
        - List of extracted questions or mark schemes
        """
        # Clean the text first to remove consecutive periods
        cleaned_text = self.clean_text_for_llm(text)
        
        # Prepare the prompt based on whether it's a question paper or mark scheme
        if is_mark_scheme:
            prompt = """
            You are an expert at analyzing physics exam mark schemes. I'll provide you with the text of a mark scheme.
            
            Your task is to:
            1. Identify each question's mark scheme in the document
            2. Extract the complete mark scheme for each question, including all subparts
            3. Format each mark scheme with the question number as a header
            4. Include the total marks for each question
            
            For each question, output in this exact format:
            
            QUESTION_START: {question_number}
            {full mark scheme text for this question}
            TOTAL_MARKS: {number of marks}
            QUESTION_END
            
            Make sure to:
            - Keep all original formatting and content for each mark scheme
            - Include all subparts (a, b, c, etc.) under the main question number
            - Extract the total marks from phrases like "Total for Question X = Y marks"
            - Separate each question with the QUESTION_START and QUESTION_END markers
            
            Here is the mark scheme text:
            """
        else:
            prompt = """
            You are an expert at analyzing physics exam papers. I'll provide you with the text of an exam paper.
            
            Your task is to:
            1. Identify each question in the document
            2. Extract the complete text for each question, including all subparts
            3. Format each question with the question number as a header
            4. Include the total marks for each question
            
            For each question, output in this exact format:
            
            QUESTION_START: {question_number}
            {full question text}
            TOTAL_MARKS: {number of marks}
            QUESTION_END
            
            Make sure to:
            - Keep all original formatting and content for each question
            - Include all subparts (a, b, c, etc.) under the main question number
            - Extract the total marks from phrases like "Total for Question X = Y marks"
            - Separate each question with the QUESTION_START and QUESTION_END markers
            - Ignore any data sheets, formula pages, or other non-question content
            
            Here is the exam paper text:
            """
        
        # Call the Perplexity API
        try:
            response = self.perplexity.generate(prompt + "\n\n" + cleaned_text)
            return self.parse_llm_response(response)
        except Exception as e:
            logger.error(f"Error calling Perplexity API: {e}")
            return []
    
    def parse_llm_response(self, response):
        """
        Parse the LLM response to extract questions or mark schemes.
        
        Parameters:
        - response: LLM response text
        
        Returns:
        - List of (question_number, text, marks) tuples
        """
        results = []
        
        # Pattern to match the question blocks
        pattern = r"QUESTION_START: (\d+)\s+(.*?)TOTAL_MARKS: (\d+)\s+QUESTION_END"
        matches = re.finditer(pattern, response, re.DOTALL)
        
        for match in matches:
            question_number = match.group(1)
            question_text = match.group(2).strip()
            marks = int(match.group(3))
            
            results.append((question_number, question_text, marks))
        
        return results
    
    def process_paper_pair_with_llm(self, qp_path, ms_path, paper_key):
        """
        Process a question paper and mark scheme pair using LLM.
        
        Parameters:
        - qp_path: Path to the question paper
        - ms_path: Path to the mark scheme
        - paper_key: Paper key (year, level, paper_number)
        
        Returns:
        - List of paths to saved question-mark scheme pairs
        """
        year, level, paper_number = paper_key
        paper_id = f"{year}_{level}_{paper_number}"
        
        logger.info(f"Processing paper pair with LLM: {paper_id}")
        
        # Extract text from PDFs
        qp_text = self.extract_text_from_pdf(qp_path)
        ms_text = self.extract_text_from_pdf(ms_path)
        
        if not qp_text or not ms_text:
            logger.error(f"Failed to extract text from PDFs for {paper_id}")
            return []
        
        # Extract questions and mark schemes using LLM
        questions = self.extract_questions_with_llm(qp_text, is_mark_scheme=False)
        mark_schemes = self.extract_questions_with_llm(ms_text, is_mark_scheme=True)
        
        logger.info(f"Paper {paper_id}: Extracted {len(questions)} questions and {len(mark_schemes)} mark schemes")
        
        # Match questions with mark schemes
        matched_pairs = self.match_questions_with_mark_schemes(questions, mark_schemes)
        
        logger.info(f"Paper {paper_id}: Matched {len(matched_pairs)} question-mark scheme pairs")
        
        # Save each question-mark scheme pair
        saved_files = []
        for q_num, q_text, ms_text, marks in matched_pairs:
            file_path = self.save_question_pair(q_num, q_text, ms_text, marks, paper_key)
            saved_files.append(file_path)
        
        # Mark this paper as completed
        self.progress["completed_papers"].append(paper_id)
        self.save_progress()
        
        return saved_files
    
    def process_all_papers(self):
        """
        Process all question paper and mark scheme pairs.
        
        Returns:
        - List of paths to saved question-mark scheme pairs
        """
        # Find paper pairs
        pairs = self.find_paper_pairs()
        
        logger.info(f"Found {len(pairs)} question paper and mark scheme pairs")
        
        # Choose the appropriate processing method
        if self.use_llm:
            return self.process_all_papers_with_llm(pairs)
        else:
            return self.process_all_papers_batch()
    
    def process_all_papers_with_llm(self, pairs):
        """
        Process all question paper and mark scheme pairs using LLM.
        
        Parameters:
        - pairs: List of (qp_path, ms_path, paper_key) tuples
        
        Returns:
        - List of paths to saved question-mark scheme pairs
        """
        # Filter out already completed papers
        remaining_pairs = []
        for qp_path, ms_path in pairs:
            paper_key = self.extract_paper_key(qp_path)
            if not paper_key:
                continue
                
            year, level, paper_number = paper_key
            paper_id = f"{year}_{level}_{paper_number}"
            
            if paper_id in self.progress["completed_papers"]:
                logger.info(f"Skipping already processed paper: {paper_id}")
            else:
                remaining_pairs.append((qp_path, ms_path, paper_key))
        
        logger.info(f"{len(remaining_pairs)} papers remaining to process")
        
        # Process each pair with progress bar
        all_saved_files = []
        for qp_path, ms_path, paper_key in tqdm(remaining_pairs, desc="Processing papers with LLM"):
            saved_files = self.process_paper_pair_with_llm(qp_path, ms_path, paper_key)
            all_saved_files.extend(saved_files)
        
        logger.info(f"Saved {len(all_saved_files)} question-mark scheme pairs in total")
        
        return all_saved_files
    
    def extract_paper_key(self, pdf_path):
        """
        Extract paper key from PDF path.
        
        Parameters:
        - pdf_path: Path to the PDF file
        
        Returns:
        - Tuple of (year, level, paper_number) or None if not found
        """
        filename = os.path.basename(pdf_path)
        match = re.match(r"(\d+)_([A-Z]+)_(\d[A-Z]?)_([A-Z]+)", filename)
        if match:
            year, level, paper_number, _ = match.groups()
            return (year, level, paper_number)
        return None

    def clean_text_for_llm(self, text):
        """
        Clean text before sending to LLM by removing large chunks of consecutive periods.
        
        Parameters:
        - text: Text to clean
        
        Returns:
        - Cleaned text
        """
        # Remove chunks of 3 or more consecutive periods
        cleaned_text = re.sub(r'\.{3,}', ' ', text)
        
        # Remove any remaining excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()


def process_exam_papers():
    """
    Process all exam papers in the raw examples directory.
    This function handles the renaming and page removal.
    """
    # Process and rename the papers
    logger.info("Processing and renaming exam papers")
    processor = ExamPaperProcessor(input_dir="raw examples", output_dir="examples/papers")
    processed_files = processor.process_directory()
    
    logger.info("Exam paper processing complete!")
    return processed_files


def extract_questions():
    """
    Extract question-mark scheme pairs from processed exam papers.
    """
    logger.info("Extracting questions from processed exam papers")
    extractor = QuestionExtractor(examples_dir="examples/papers", output_dir="examples/questions")
    saved_files = extractor.process_all_papers()
    
    logger.info(f"Extracted {len(saved_files)} question-mark scheme pairs")
    return saved_files


def extract_questions_with_llm():
    """
    Extract question-mark scheme pairs from processed exam papers using LLM.
    """
    logger.info("Extracting questions from processed exam papers using LLM")
    extractor = QuestionExtractor(
        examples_dir="examples/papers", 
        output_dir="examples/questions",
        use_llm=True
    )
    saved_files = extractor.process_all_papers()
    
    logger.info(f"Extracted {len(saved_files)} question-mark scheme pairs")
    return saved_files


def main():
    """Main function to run the exam paper processor and question extractor"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process exam papers and extract questions")
    parser.add_argument("--input", default="raw examples", help="Input directory containing PDF files")
    parser.add_argument("--output", default="examples/papers", help="Output directory for processed files")
    parser.add_argument("--extract", action="store_true", help="Extract questions from processed papers")
    parser.add_argument("--process", action="store_true", help="Process raw exam papers")
    parser.add_argument("--llm", action="store_true", help="Use LLM for question extraction")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Process exam papers if requested
    if args.process:
        processor = ExamPaperProcessor(input_dir=args.input, output_dir=args.output)
        processor.process_directory()
    
    # Extract questions if requested
    if args.extract:
        if args.llm:
            extract_questions_with_llm()
        else:
            extract_questions()
    
    # If no specific action is requested, do both processing and extraction
    if not args.process and not args.extract:
        # Process new papers
        processor = ExamPaperProcessor(input_dir=args.input, output_dir=args.output)
        processor.process_directory()
        
        # Extract questions
        if args.llm:
            extract_questions_with_llm()
        else:
            extract_questions()
    
    end_time = time.time()
    
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 