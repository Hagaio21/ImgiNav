#!/usr/bin/env python3
"""
Test runner script for ImgiNav comprehensive test suite.
Runs all tests with proper setup and reporting.
"""
import os
import sys
import unittest
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests(test_pattern=None, verbose=2, failfast=False):
    """Run tests with specified options."""
    # Discover tests
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    
    if test_pattern:
        suite = loader.loadTestsFromName(test_pattern)
    else:
        suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=verbose,
        failfast=failfast,
        buffer=True
    )
    
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print(f"  Success: {result.wasSuccessful()}")
    print(f"{'='*60}")
    
    return result.wasSuccessful()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ImgiNav test suite")
    parser.add_argument(
        "--pattern", "-p",
        help="Test pattern to run (e.g., 'test_datasets', 'test_autoencoder')"
    )
    parser.add_argument(
        "--verbose", "-v",
        type=int,
        default=2,
        help="Verbosity level (0-2)"
    )
    parser.add_argument(
        "--failfast", "-f",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available test modules"
    )
    
    args = parser.parse_args()
    
    if args.list:
        # List available test modules
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob("test_*.py"))
        print("Available test modules:")
        for test_file in sorted(test_files):
            print(f"  {test_file.stem}")
        return 0
    
    # Run tests
    success = run_tests(
        test_pattern=args.pattern,
        verbose=args.verbose,
        failfast=args.failfast
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
