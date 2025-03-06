"""
Constants for the physics exam processing system.
Contains API keys, model names, and prompt templates.
"""

# API Keys (these should be overridden by environment variables in production)
DEFAULT_PPLX_API_KEY = "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"
# Model names
DEFAULT_PPLX_MODEL = "r1-1776"
GENERATION_PPLX_MODEL = "sonar-reasoning"

# Prompt templates
QUESTION_EXTRACTION_PROMPT = """
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

MARK_SCHEME_EXTRACTION_PROMPT = """
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

EMBEDDING_GENERATION_PROMPT = """
Generate an embedding for the following text: {text}
"""

PHYSICS_QUESTION_GENERATION_PROMPT = """
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

Here are some examples of good physics questions:
{examples_content}

Now, please generate a new physics question with the following parameters:
- Topic: {topic}
- Difficulty: {difficulty}
- Question type: {question_type}
"""

PHYSICS_QUESTION_SYSTEM_PROMPT = "You are a physics teacher creating a level edexcel exam questions." 