import os
import glob
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import shutil
import argparse
import time
from tqdm import tqdm
import logging
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ocr_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OCR_Processor")

class OCRProcessor:
    """
    Process PDF files using Tesseract OCR and save the results as text files.
    """
    
    def __init__(self, input_dir="raw examples", output_dir="examples", dpi=400, lang="eng"):
        """
        Initialize the OCR processor.
        
        Parameters:
        - input_dir: Directory containing raw PDF files
        - output_dir: Directory to save processed files
        - dpi: DPI for PDF to image conversion (higher = better quality but slower)
        - lang: Tesseract language code
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dpi = dpi
        self.lang = lang
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check if Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            logger.error("Please install Tesseract OCR and make sure it's in your PATH")
            raise RuntimeError("Tesseract OCR not found")
    
    def preprocess_image(self, image):
        """
        Preprocess the image to improve OCR accuracy
        
        Parameters:
        - image: PIL Image object
        
        Returns:
        - Preprocessed PIL Image
        """
        # Convert PIL image to OpenCV format
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Convert back to PIL Image
        enhanced_img = Image.fromarray(denoised)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = enhancer.enhance(2.0)
        
        # Sharpen the image
        enhanced_img = enhanced_img.filter(ImageFilter.SHARPEN)
        
        return enhanced_img
    
    def process_pdf(self, pdf_path, output_format="pdf"):
        """
        Process a single PDF file with OCR.
        
        Parameters:
        - pdf_path: Path to the PDF file
        - output_format: Format to save the output ('pdf', 'docx', or 'txt')
        
        Returns:
        - Path to the processed file
        """
        logger.info(f"Processing {pdf_path}")
        
        # Get base filename without extension
        base_name = os.path.basename(pdf_path)
        file_name = os.path.splitext(base_name)[0]
        
        # Create output path
        output_path = os.path.join(self.output_dir, f"{file_name}.{output_format}")
        
        # Check if output file already exists
        if os.path.exists(output_path):
            logger.info(f"Output file {output_path} already exists, skipping")
            return output_path
        
        try:
            # Try to find poppler path
            poppler_paths = [
                "/usr/local/bin",  # Intel Mac
                "/opt/homebrew/bin",  # Apple Silicon Mac
                "/usr/bin",  # Linux
                "C:\\Program Files\\poppler-23.11.0\\Library\\bin"  # Windows
            ]
            
            poppler_path = None
            for path in poppler_paths:
                if os.path.exists(path) and (
                    os.path.exists(os.path.join(path, "pdftoppm")) or 
                    os.path.exists(os.path.join(path, "pdftoppm.exe"))
                ):
                    poppler_path = path
                    break
            
            # Convert PDF to images
            logger.info(f"Converting PDF to images with DPI={self.dpi}")
            if poppler_path:
                logger.info(f"Using poppler path: {poppler_path}")
                images = convert_from_path(pdf_path, dpi=self.dpi, poppler_path=poppler_path)
            else:
                images = convert_from_path(pdf_path, dpi=self.dpi)
            
            # Process each page
            logger.info(f"Processing {len(images)} pages with OCR")
            text_content = ""
            
            # Configure Tesseract parameters for better accuracy
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            
            for i, image in enumerate(tqdm(images, desc=f"OCR {file_name}")):
                # Preprocess the image
                processed_image = self.preprocess_image(image)
                
                # Apply OCR to the image with custom configuration
                page_text = pytesseract.image_to_string(
                    processed_image, 
                    lang=self.lang,
                    config=custom_config
                )
                
                # Add page separator and text
                text_content += f"\n\n--- Page {i+1} ---\n\n{page_text}"
            
            # Save the OCR result based on the output format
            if output_format == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
            
            elif output_format == 'pdf':
                # For PDF, we'll create a searchable PDF
                # First save the OCR text to a temporary file
                temp_txt = os.path.join(self.output_dir, f"{file_name}_temp.txt")
                with open(temp_txt, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                
                # Then copy the original PDF to the output directory
                shutil.copy2(pdf_path, output_path)
                
                # Remove the temporary text file
                os.remove(temp_txt)
                
                logger.info(f"Created searchable PDF: {output_path}")
            
            elif output_format == 'docx':
                # For DOCX, we'll use python-docx
                try:
                    import docx
                    doc = docx.Document()
                    
                    # Split the text into paragraphs and add to the document
                    paragraphs = text_content.split('\n')
                    for para in paragraphs:
                        if para.strip():  # Skip empty paragraphs
                            doc.add_paragraph(para)
                    
                    doc.save(output_path)
                    logger.info(f"Created DOCX: {output_path}")
                
                except ImportError:
                    logger.error("python-docx not installed. Install with: pip install python-docx")
                    # Fall back to text format
                    output_path = os.path.join(self.output_dir, f"{file_name}.txt")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
            
            logger.info(f"Saved OCR result to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return None
    
    def process_directory(self, output_format="pdf"):
        """
        Process all PDF files in the input directory.
        
        Parameters:
        - output_format: Format to save the output ('pdf', 'docx', or 'txt')
        
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
            output_path = self.process_pdf(pdf_path, output_format)
            if output_path:
                processed_files.append(output_path)
        
        logger.info(f"Processed {len(processed_files)} files")
        return processed_files

def main():
    """Main function to run the OCR processor"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process PDF files with OCR")
    parser.add_argument("--input", default="raw examples", help="Input directory containing PDF files")
    parser.add_argument("--output", default="examples", help="Output directory for processed files")
    parser.add_argument("--dpi", type=int, default=400, help="DPI for PDF to image conversion")
    parser.add_argument("--lang", default="eng", help="Tesseract language code")
    parser.add_argument("--format", choices=["pdf", "docx", "txt"], default="txt", 
                        help="Output format (pdf, docx, or txt)")
    
    args = parser.parse_args()
    
    # Create and run the OCR processor
    processor = OCRProcessor(
        input_dir=args.input,
        output_dir=args.output,
        dpi=args.dpi,
        lang=args.lang
    )
    
    start_time = time.time()
    processor.process_directory(output_format=args.format)
    end_time = time.time()
    
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 