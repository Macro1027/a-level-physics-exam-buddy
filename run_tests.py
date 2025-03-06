#!/usr/bin/env python3
import unittest
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import test modules
from tests.test_integration import IntegrationTests, StreamlitIntegrationTests

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(IntegrationTests))
    test_suite.addTest(unittest.makeSuite(StreamlitIntegrationTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful()) 