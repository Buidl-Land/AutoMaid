#!/usr/bin/env python3
"""
Setup script for AgenticMaid package.
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
    return "AgenticMaid - A Python framework for building reactive, multi-agent systems."

# Read requirements from requirements.txt
def get_requirements():
    """Read requirements from requirements.txt"""
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="agenticmaid",
    version=get_version(),
    author="AgenticMaid Team",
    author_email="contact@agenticmaid.com",
    description="A Python framework for building reactive, multi-agent systems",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/agenticmaid/agenticmaid",
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
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "telegram": [
            "python-telegram-bot>=20.0",
        ],
        "solana": [
            "solana>=0.30.0",
            "solders>=0.18.0",
        ],
        "all": [
            "python-telegram-bot>=20.0",
            "solana>=0.30.0",
            "solders>=0.18.0",
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
        "telegram", "messaging", "reactive", "async", "framework"
    ],
    project_urls={
        "Bug Reports": "https://github.com/agenticmaid/agenticmaid/issues",
        "Source": "https://github.com/agenticmaid/agenticmaid",
        "Documentation": "https://github.com/agenticmaid/agenticmaid#readme",
    },
)
