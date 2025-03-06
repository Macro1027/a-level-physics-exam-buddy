#!/usr/bin/env python3
import unittest
import os
import sys
import traceback
import argparse
import logging
import warnings
import io

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Completely suppress all output during imports
if '--quiet' in sys.argv or '-q' in sys.argv:
    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Redirect to null device
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

# Completely suppress Streamlit warnings
class StreamlitWarningFilter(logging.Filter):
    def filter(self, record):
        return not (record.name.startswith('streamlit') or 
                   (hasattr(record, 'msg') and isinstance(record.msg, str) and 'streamlit' in record.msg.lower()))

# Apply the filter to the root logger
root_logger = logging.getLogger()
root_logger.addFilter(StreamlitWarningFilter())

# Also suppress warnings from the warnings module
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", module="streamlit")

# Set streamlit logger to ERROR level
logging.getLogger("streamlit").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.CRITICAL)

# Import test modules
from tests.test_integration import IntegrationTests, StreamlitIntegrationTests

def run_test_file(file_path, verbosity=2):
    """Run a single test file and report any errors"""
    print(f"\n=== Running tests from {file_path} ===\n")
    try:
        # Import the test module
        module_name = file_path.replace('/', '.').replace('.py', '')
        __import__(module_name)
        
        # Run the tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(module_name)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except Exception as e:
        print(f"Error running tests from {file_path}:")
        traceback.print_exc()
        return False

def run_all_tests(verbosity=2, debug_mode=False):
    """Run all test suites"""
    if debug_mode:
        # Run each test file separately for better error reporting
        test_files = [
            'tests/test_integration.py',
            'tests/test_unit.py',
            'tests/test_ui.py',
            'tests/test_e2e.py'
        ]
        
        success = True
        for test_file in test_files:
            if not run_test_file(test_file, verbosity):
                success = False
        
        return success
    else:
        # Create test suite with all test cases
        test_suite = unittest.TestSuite()
        
        # Add test cases
        # Integration tests
        test_suite.addTest(unittest.makeSuite(IntegrationTests))
        test_suite.addTest(unittest.makeSuite(StreamlitIntegrationTests))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(test_suite)
        
        return result.wasSuccessful()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tests for the Physics Question Generator")
    parser.add_argument("--file", type=str, help="Run a specific test file (e.g., tests/test_unit.py)")
    parser.add_argument("--debug", action="store_true", help="Run tests in debug mode (one file at a time)")
    parser.add_argument("--verbose", "-v", action="count", default=1, help="Increase verbosity (can be used multiple times)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress Streamlit warnings")
    
    args = parser.parse_args()
    
    # Set verbosity level
    verbosity = min(args.verbose + 1, 3)  # 1, 2, or 3
    
    # Run tests
    if args.file:
        success = run_test_file(args.file, verbosity)
    else:
        success = run_all_tests(verbosity, args.debug)
    
    # Exit with non-zero code if tests failed
    sys.exit(0 if success else 1)

# Restore stdout/stderr if they were redirected
if '--quiet' in sys.argv or '-q' in sys.argv:
    sys.stdout = original_stdout
    sys.stderr = original_stderr 