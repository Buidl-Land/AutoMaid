"""
Test runner for the messaging system.

This script runs all tests for the messaging system components.
"""

import sys
import os
import pytest
import asyncio
from pathlib import Path

# Add the messaging system to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_all_tests():
    """Run all tests for the messaging system."""
    test_dir = Path(__file__).parent
    
    # Configure pytest arguments
    pytest_args = [
        str(test_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "-x",  # Stop on first failure
        "--asyncio-mode=auto",  # Auto async mode
    ]
    
    # Add coverage if available
    try:
        import pytest_cov
        pytest_args.extend([
            "--cov=messaging_system",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    except ImportError:
        print("pytest-cov not available, running without coverage")
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    return exit_code


def run_specific_tests(test_pattern: str):
    """
    Run specific tests matching a pattern.
    
    Args:
        test_pattern: Pattern to match test files or test names
    """
    test_dir = Path(__file__).parent
    
    pytest_args = [
        str(test_dir),
        "-v",
        "-k", test_pattern,
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    exit_code = pytest.main(pytest_args)
    return exit_code


def run_client_tests():
    """Run only client tests."""
    return run_specific_tests("test_clients")


def run_trigger_tests():
    """Run only trigger tests."""
    return run_specific_tests("test_triggers")


def run_core_tests():
    """Run only core component tests."""
    return run_specific_tests("test_core")


def run_integration_tests():
    """Run only integration tests."""
    return run_specific_tests("test_integration")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run messaging system tests")
    parser.add_argument(
        "--component",
        choices=["all", "clients", "triggers", "core", "integration"],
        default="all",
        help="Which component tests to run"
    )
    parser.add_argument(
        "--pattern",
        help="Specific test pattern to match"
    )
    
    args = parser.parse_args()
    
    if args.pattern:
        exit_code = run_specific_tests(args.pattern)
    elif args.component == "all":
        exit_code = run_all_tests()
    elif args.component == "clients":
        exit_code = run_client_tests()
    elif args.component == "triggers":
        exit_code = run_trigger_tests()
    elif args.component == "core":
        exit_code = run_core_tests()
    elif args.component == "integration":
        exit_code = run_integration_tests()
    else:
        exit_code = run_all_tests()
    
    sys.exit(exit_code)
