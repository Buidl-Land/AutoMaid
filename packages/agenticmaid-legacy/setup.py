#!/usr/bin/env python3
"""
Setup script for AgenticMaid Legacy package.
"""

import os
import re
from setuptools import setup, find_packages

# Read the version from the package __init__.py
def get_version():
    """Extract version from agenticmaid_legacy/__init__.py"""
    init_file = os.path.join(os.path.dirname(__file__), 'agenticmaid_legacy', '__init__.py')
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
    return "AgenticMaid Legacy - Backward compatibility and legacy API methods for AgenticMaid framework."

setup(
    name="agenticmaid-legacy",
    version=get_version(),
    author="AgenticMaid Team",
    author_email="contact@agenticmaid.com",
    description="Backward compatibility and legacy API methods for AgenticMaid framework",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/agenticmaid/agenticmaid-legacy",
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
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: System :: Archiving :: Backup",
    ],
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv>=0.19.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "mcp": [
            "langchain-mcp-adapters>=0.1.0",
        ],
        "core": [
            "agenticmaid-core>=1.0.0",
        ],
        "clients": [
            "agenticmaid-clients>=1.0.0",
        ],
        "triggers": [
            "agenticmaid-triggers>=1.0.0",
        ],
        "full": [
            "agenticmaid-core>=1.0.0",
            "agenticmaid-clients[all]>=1.0.0",
            "agenticmaid-triggers[all]>=1.0.0",
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
            "agenticmaid-migrate=agenticmaid_legacy:migrate_to_modern",
        ],
    },
    include_package_data=True,
    package_data={
        "agenticmaid_legacy": [
            "*.json",
            "*.md",
            "*.txt",
        ],
    },
    keywords=[
        "agenticmaid", "legacy", "backward-compatibility", "migration",
        "deprecated", "api", "compatibility", "upgrade"
    ],
    project_urls={
        "Bug Reports": "https://github.com/agenticmaid/agenticmaid-legacy/issues",
        "Source": "https://github.com/agenticmaid/agenticmaid-legacy",
        "Documentation": "https://github.com/agenticmaid/agenticmaid-legacy#readme",
        "Main Project": "https://github.com/agenticmaid/agenticmaid",
        "Migration Guide": "https://github.com/agenticmaid/agenticmaid-legacy/blob/main/MIGRATION.md",
    },
)
