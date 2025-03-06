import unittest
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from src directory
from src.pplx import Perplexity, generate_physics_question, process_guide, process_examples

class IntegrationTests(unittest.TestCase):
    """Integration tests for the Physics Question Generator application"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create test paths
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.test_guide_path = self.test_dir / "test_data" / "test_guide.txt"
        self.test_examples_path = self.test_dir / "test_data" / "test_examples.docx"
        
        # Create test directory if it doesn't exist
        os.makedirs(self.test_dir / "test_data", exist_ok=True)
        
        # Create test guide file
        with open(self.test_guide_path, 'w', encoding='utf-8') as f:
            f.write("This is a test guide for Edexcel A-level Physics questions.")
        
        # Mock API responses
        self.mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "<question>Question 1: Test Question\nCalculate something.\n\nMark Scheme:\n- Point 1 (1)\n- Point 2 (1)</question>"
                    }
                }
            ]
        }
    
    def tearDown(self):
        """Clean up test fixtures after each test method"""
        # Remove test files
        if self.test_guide_path.exists():
            os.remove(self.test_guide_path)
        
        # Remove test directory if empty
        try:
            os.rmdir(self.test_dir / "test_data")
        except OSError:
            pass  # Directory not empty or doesn't exist
    
    @patch('requests.post')
    def test_perplexity_api_integration(self, mock_post):
        """Test integration with Perplexity API"""
        # Configure the mock
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Create Perplexity client
        client = Perplexity(api_key="test_key", model="test_model")
        
        # Test generate method
        result = client.generate("Test prompt", system_prompt="Test system prompt")
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertIn("Question 1", result)
        
        # Verify API was called with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        self.assertIn("json", call_args)
        self.assertEqual(call_args["json"]["model"], "test_model")
        self.assertEqual(len(call_args["json"]["messages"]), 2)
    
    @patch('src.pplx.Perplexity.generate')
    def test_generate_physics_question_integration(self, mock_generate):
        """Test integration between generate_physics_question and Perplexity"""
        # Configure the mock
        mock_generate.return_value = "<question>Question 1: Test Question\nCalculate something.\n\nMark Scheme:\n- Point 1 (1)\n- Point 2 (1)</question>"
        
        # Test generate_physics_question
        result = generate_physics_question(
            topic="waves",
            difficulty="medium",
            question_type="calculation",
            guide_content="Test guide content",
            examples_content="Test examples content"
        )
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertIn("Question 1", result)
        
        # Verify Perplexity.generate was called with correct parameters
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args
        self.assertIn("prompt", call_args[1])
        self.assertIn("waves", call_args[1]["prompt"])
        self.assertIn("medium", call_args[1]["prompt"])
        self.assertIn("calculation", call_args[1]["prompt"])
    
    def test_process_guide_integration(self):
        """Test integration between process_guide and file system"""
        # Test process_guide with test file
        result = process_guide(self.test_guide_path)
        
        # Verify the result
        self.assertEqual(result, "This is a test guide for Edexcel A-level Physics questions.")
    
    def test_process_guide_file_not_found(self):
        """Test process_guide with non-existent file"""
        # Test process_guide with non-existent file
        result = process_guide("non_existent_file.txt")
        
        # Verify the result
        self.assertEqual(result, "Guide file not found.")
    
    @patch('docx.Document')
    def test_process_examples_docx_integration(self, mock_document):
        """Test integration between process_examples and docx library"""
        # Configure the mock
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text="Example 1"), MagicMock(text="Example 2")]
        mock_document.return_value = mock_doc
        
        # Create a dummy docx file path - use string instead of Path object
        docx_path = str(self.test_dir / "test_data" / "test_examples.docx")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(docx_path), exist_ok=True)
        
        # Create an empty file
        with open(docx_path, 'w') as f:
            f.write("")
        
        # Test process_examples with docx file
        result = process_examples(docx_path)
        
        # Verify the result
        self.assertEqual(result, "Example 1\nExample 2")
        
        # Clean up
        os.remove(docx_path)

class StreamlitIntegrationTests(unittest.TestCase):
    """Integration tests for Streamlit components"""
    
    @patch('streamlit.markdown')
    @patch('streamlit.expander')
    @patch('streamlit.tabs')
    def test_question_display_integration(self, mock_tabs, mock_expander, mock_markdown):
        """Test integration of question display components"""
        # This is a placeholder for Streamlit integration tests
        # In a real implementation, you would use a testing framework that can handle Streamlit components
        # or mock the Streamlit functions to verify they're called correctly
        pass

if __name__ == '__main__':
    unittest.main() 