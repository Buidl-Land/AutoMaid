#!/usr/bin/env python3
"""
Setup script for AgenticMaid Core package.
"""

import os
import re
from setuptools import setup, find_packages

# Read the version from the package __init__.py
def get_version():
    """Extract version from agenticmaid/__init__.py"""
    init_file = os.path.join(os.path.dirname(__file__), 'agenticmaid', '__init__.py')
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
    return "AgenticMaid Core - The core framework for building reactive, multi-agent systems."

setup(
    name="agenticmaid-core",
    version=get_version(),
    author="AgenticMaid Team",
    author_email="contact@agenticmaid.com",
    description="Core framework for AgenticMaid - A Python framework for building reactive, multi-agent systems",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/agenticmaid/agenticmaid-core",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications :: Chat",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv>=0.19.0",
        "langchain-mcp-adapters>=0.1.0",
        "langgraph>=0.1.0",
        "schedule>=1.2.0",
        "langchain-core>=0.1.0",
        "langchain-openai>=0.1.0",
        "langchain-anthropic>=0.1.0",
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
        "uvicorn[standard]>=0.20.0",
        "chromadb>=0.4.0",
        "aiofiles>=23.0.0",
        "httpx>=0.24.0",
        "asyncio-mqtt>=0.13.0",
    ],
    extras_require={
        "clients": [
            "agenticmaid-clients>=1.0.0",
        ],
        "triggers": [
            "agenticmaid-triggers>=1.0.0",
        ],
        "legacy": [
            "agenticmaid-legacy>=1.0.0",
        ],
        "messaging": [
            "agenticmaid-clients[telegram]>=1.0.0",
            "agenticmaid-triggers[all]>=1.0.0",
        ],
        "full": [
            "agenticmaid-clients[all]>=1.0.0",
            "agenticmaid-triggers[all]>=1.0.0",
            "agenticmaid-legacy>=1.0.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "agenticmaid=agenticmaid.cli:main",
            "agenticmaid-api=agenticmaid.api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "agenticmaid": [
            "*.json",
            "*.md",
            "*.txt",
        ],
    },
    keywords=[
        "ai", "agents", "multi-agent", "mcp", "chatbot", "automation",
        "llm", "openai", "anthropic", "google", "gemini", "claude",
        "reactive", "async", "framework", "core"
    ],
    project_urls={
        "Bug Reports": "https://github.com/agenticmaid/agenticmaid-core/issues",
        "Source": "https://github.com/agenticmaid/agenticmaid-core",
        "Documentation": "https://github.com/agenticmaid/agenticmaid-core#readme",
        "Main Project": "https://github.com/agenticmaid/agenticmaid",
    },
)
