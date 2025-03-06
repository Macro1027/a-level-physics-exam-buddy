# A-Level Physics Exam Buddy

An AI-powered tool for generating custom A-Level Physics exam questions with explanations and solutions, designed specifically for Edexcel A-Level Physics students.

## Features

- **Custom Question Generation**: Create physics questions based on specific topics, difficulty levels, and question types
- **Interactive UI**: User-friendly interface with topic-specific icons and styling
- **RAG (Retrieval-Augmented Generation)**: Uses real exam questions to generate authentic practice materials
- **OCR Processing**: Convert PDF exam papers into searchable text for the RAG system
- **Analytics Dashboard**: Track your progress and identify areas for improvement
- **User Profiles**: Save your history and track your progress over time
- **Interactive Diagrams**: Visualize physics concepts with interactive plots

## Topics Covered

- Mechanics - Motion
- Mechanics - Forces
- Electricity - Circuits
- Electricity - Fields
- Waves - Properties
- Waves - Optics
- Nuclear Physics
- Thermodynamics
- Particle Physics
- Magnetic Fields

## Project Structure

- `app.py`: Main Streamlit application
- `src/RAG.py`: Retrieval-Augmented Generation system for processing exam questions
- `src/ocr.py`: OCR processing for converting PDF exam papers to text
- `src/analytics_dashboard.py`: Analytics and visualization tools
- `src/styles.py`: UI styling and theme components
- `src/pplx.py`: Integration with Perplexity AI for question generation

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install additional requirements:
   - **Tesseract OCR**: Required for OCR processing
     - For macOS: `brew install tesseract`
     - For Windows: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
     - For Linux: `sudo apt-get install tesseract-ocr`
   - **Poppler**: Required for PDF processing
     - For macOS: `brew install poppler`
     - For Windows: Download from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)
     - For Linux: `sudo apt-get install poppler-utils`

4. Set up your Perplexity API key:
   - Create a `.env` file in the project root
   - Add your API key: `PERPLEXITY_API_KEY=your_api_key_here`

5. Run the application:
   ```
   streamlit run app.py
   ```

## OCR Processing

To process PDF exam papers for the RAG system:

1. Create a folder called `raw examples`
2. Place your PDF files in the `raw examples` folder
3. Run the OCR processor:
   ```
   python src/ocr.py
   ```
4. Processed files will be saved in the `examples` folder

## Using the Application

1. **Generate Questions**: Select a topic, difficulty, and question type to generate custom questions
2. **View History**: See all your previously generated questions
3. **Profile**: View your statistics and topic preferences
4. **Analytics**: Explore detailed analytics about your question generation activity
5. **Settings**: Customize the application and manage your account

## Technologies Used

- **Streamlit**: Web interface and application framework
- **Perplexity AI**: Advanced question generation
- **Tesseract OCR**: PDF text extraction
- **Plotly**: Interactive diagrams and visualizations
- **scikit-learn**: Text processing and similarity search
- **PyMuPDF & python-docx**: Document processing
- **OpenCV**: Image preprocessing for OCR

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Edexcel A-Level Physics curriculum
- Perplexity AI for providing the question generation capabilities
- Streamlit for the interactive web framework

---

© 2025 A-Level Physics Exam Buddy | Marco
