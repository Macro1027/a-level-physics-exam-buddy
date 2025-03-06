# Physics Exam Processing and Retrieval-Augmented Generation (RAG) System

This document provides a detailed technical overview of the implemented system for processing physics exam papers, extracting structured questions and mark schemes using LLMs, and creating a Retrieval-Augmented Generation (RAG) database using Pinecone and Perplexity's API.

---

## 1. Raw Exam Paper Processing (`src/utils.py`)

### Overview:
- Processes raw PDF exam papers from the `raw examples` directory.
- Renames files into a standardized format: `YY_LEVEL_PAPER_TYPE.pdf` (e.g., `22_AS_1_QP.pdf`).
- Removes unnecessary pages based on paper type and year.

### Key Functionalities:
- **Fuzzy Filename Parsing**: Uses regex patterns to handle variations and spelling errors in filenames.
- **Page Removal Logic**:
  - Question Papers: Removes the first page.
  - Mark Schemes: Removes the first 4 pages for years 2016 and 2019; removes the first 5 pages for all other years.
- **Logging**: Detailed logging of processing steps and errors.

### Usage:
```bash
python src/utils.py --process
```

---

## 2. Question and Mark Scheme Extraction with LLM (`src/utils.py`)

### Overview

Extracts structured questions and mark schemes from processed PDFs using Perplexity's LLM (`r1-1776` model).

### Workflow:

- **Text Extraction**: Extracts text from processed PDFs using PyMuPDF (`fitz`).
- **Text Cleaning**: Removes large chunks of consecutive periods (`...`) and excessive whitespace.
- **LLM Prompting**: Sends cleaned text to Perplexity API with structured prompts to extract questions and mark schemes.
- **Structured Output**: Parses LLM responses into structured question-mark scheme pairs, clearly delimited by markers (`QUESTION_START`, `QUESTION_END`, `TOTAL_MARKS`).

### Progress Tracking:
- Uses `extraction_progress.json` to track completed papers, allowing resumption after interruptions.

---

## 3. Retrieval-Augmented Generation (RAG) Database (`src/RAG.py`)

### Overview:

Creates a semantic search database of physics questions using Pinecone for vector storage and Perplexity's Sonar model (`r1-1776`) for embeddings.

### Components:

#### Pinecone Vector Database:
- **Initialization**: Connects to Pinecone using API key (`PINECONE_API_KEY` environment variable).
- **Index Creation**: Automatically creates a Pinecone index (`physics-questions`) with 1536-dimensional embeddings and cosine similarity metric.

### Embedding Generation:

- **Perplexity Sonar Model**: Generates embeddings by prompting the Perplexity API with:
  ```
  "Generate an embedding for the following text: {text}"
  ```
- **Embedding Parsing**: Extracts embeddings from the LLM response using regex to find JSON arrays. If parsing fails, falls back to a simple character-based embedding (normalized ASCII values).

### Database Creation Workflow:

- **Question Loading**: Reads structured question files from `examples/questions`.
- **Metadata Extraction**: Parses filenames (`YY_LEVEL_PAPER_QUESTION.txt`) to extract metadata (year, level, paper number, question number).
- **Vector Upsertion**: Embeddings and metadata are batched and upserted into Pinecone.

### Semantic Search:

- **Query Embedding**: Generates embedding for user queries using Perplexity Sonar.
- **Similarity Search**: Queries Pinecone for top-k similar questions based on cosine similarity.
- **Filtering**: Supports metadata-based filtering (e.g., by year, level).

---

## 4. Command-Line Interface (CLI)

### Processing Raw Papers:

```bash
python src/utils.py --process
```

### Extracting Questions with LLM:

```bash
python src/utils.py --extract --llm
```

### Creating RAG Database:

```bash
python src/RAG.py --create
```

### Querying RAG Database:

```bash
python src/RAG.py --query "Explain Newton's second law" --top_k 5
```

---

## 4. Technical Stack and Dependencies

### Python Libraries:
- **PyMuPDF (`fitz`)**: PDF text extraction and manipulation.
- **Perplexity API (`pplx.py`)**: For LLM-based extraction and embedding generation.
- **Pinecone**: Vector database for semantic search.
- **TQDM**: Progress bars for processing visibility.
- **Logging**: Detailed logs for debugging and monitoring.

### Environment Variables:
- `PPLX_API_KEY`: Perplexity API key.
- `PINECONE_API_KEY`: Pinecone API key.

---

## 4. Directory Structure:

```
project-root/
├── raw examples/          # Raw PDF exam papers
├── examples/papers/       # Processed PDFs (renamed, pages removed)
├── examples/questions/    # Extracted question-mark scheme pairs
├── extraction_progress.json  # Tracks extraction progress
├── src/
│   ├── utils.py           # Paper processing and extraction logic
│   ├── pplx.py            # Perplexity API client
│   ├── RAG.py             # RAG database creation and querying
│   ├── analytics_dashboard.py # Dashboard for analytics (unchanged)
│   └── styles.py          # Dashboard styling
├── processing.log         # Logs for paper processing
└── rag_processing.log     # Logs for RAG database processing

---

## 4. Error Handling and Logging

- **Robust Error Handling**: All API calls and file operations are wrapped in try-except blocks with detailed logging.
- **Fallback Mechanisms**: If embedding extraction fails, a simple character-based embedding is generated as a fallback.

---

## 4. Future Improvements and Considerations

- **Dedicated Embedding Endpoint**: If Perplexity provides a dedicated embedding endpoint, replace the current workaround with direct API calls.
- **Enhanced Metadata**: Include additional metadata (e.g., topic, difficulty) for improved filtering and querying.
- **Automated Testing**: Implement unit tests and integration tests for robustness.
- **Rate Limit Handling**: Implement exponential backoff strategies for API rate limits.

---

This document provides a comprehensive overview of the implemented system, detailing each component's functionality, integration points, and technical considerations. The system leverages advanced LLM capabilities and semantic search to provide a powerful tool for physics exam question management and retrieval. 