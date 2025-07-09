#!/usr/bin/env python3
"""
Setup script for AgenticMaid Triggers package.
"""

import os
import re
from setuptools import setup, find_packages

# Read the version from the package __init__.py
def get_version():
    """Extract version from agenticmaid_triggers/__init__.py"""
    init_file = os.path.join(os.path.dirname(__file__), 'agenticmaid_triggers', '__init__.py')
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()
        version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', content, re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError('Unable to find version string.')

# Read the README file for long description
def get_long_description():
    """Read README.md for long description"""
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "AgenticMaid Triggers - Event trigger implementations for AgenticMaid framework."

setup(
    name="agenticmaid-triggers",
    version=get_version(),
    author="AgenticMaid Team",
    author_email="contact@agenticmaid.com",
    description="Event trigger implementations for AgenticMaid framework",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/agenticmaid/agenticmaid-triggers",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Office/Business :: Scheduling",
    ],
    python_requires=">=3.8",
    install_requires=[
        "asyncio>=3.4.3",
        "pydantic>=2.0.0",
        "python-dotenv>=0.19.0",
        "schedule>=1.2.0",
    ],
    extras_require={
        "solana": [
            "solana>=0.30.0",
            "solders>=0.18.0",
            "httpx>=0.24.0",
        ],
        "file_watcher": [
            "watchdog>=2.1.0",
        ],
        "webhook": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.20.0",
            "aiohttp>=3.8.0",
        ],
        "crypto": [
            "ccxt>=3.0.0",
            "websockets>=10.0",
        ],
        "core": [
            "agenticmaid-core>=1.0.0",
        ],
        "integration": [
            "agenticmaid-core>=1.0.0",
            "solana>=0.30.0",
            "solders>=0.18.0",
        ],
        "all": [
            "solana>=0.30.0",
            "solders>=0.18.0",
            "httpx>=0.24.0",
            "watchdog>=2.1.0",
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.20.0",
            "aiohttp>=3.8.0",
            "ccxt>=3.0.0",
            "websockets>=10.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    include_package_data=True,
    package_data={
        "agenticmaid_triggers": [
            "*.json",
            "*.md",
            "*.txt",
        ],
    },
    keywords=[
        "agenticmaid", "triggers", "events", "automation", "monitoring",
        "solana", "blockchain", "webhook", "scheduler", "file-watcher",
        "async", "reactive"
    ],
    project_urls={
        "Bug Reports": "https://github.com/agenticmaid/agenticmaid-triggers/issues",
        "Source": "https://github.com/agenticmaid/agenticmaid-triggers",
        "Documentation": "https://github.com/agenticmaid/agenticmaid-triggers#readme",
        "Main Project": "https://github.com/agenticmaid/agenticmaid",
    },
)
