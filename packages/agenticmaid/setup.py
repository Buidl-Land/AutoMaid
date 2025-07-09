#!/usr/bin/env python3
"""
Setup script for AgenticMaid meta-package.

This is a meta-package that installs the complete AgenticMaid ecosystem.
"""

from setuptools import setup

setup(
    name="agenticmaid",
    version="1.0.0",
    author="AgenticMaid Team",
    author_email="contact@agenticmaid.com",
    description="Complete AgenticMaid ecosystem - A Python framework for building reactive, multi-agent systems",
    long_description="""
# AgenticMaid

Complete AgenticMaid ecosystem meta-package. This package installs all AgenticMaid components:

- **agenticmaid-core**: Core framework and agent system
- **agenticmaid-clients**: Messaging clients for various platforms
- **agenticmaid-triggers**: Event triggers for automation
- **agenticmaid-legacy**: Backward compatibility and migration tools

## Installation

```bash
# Install complete ecosystem
pip install agenticmaid

# Install with specific features
pip install agenticmaid[telegram]  # With Telegram support
pip install agenticmaid[solana]    # With Solana monitoring
pip install agenticmaid[all]       # Everything
```

## Quick Start

```python
from agenticmaid import AgenticMaid

config = {
    "ai_services": {
        "default_service": {
            "provider": "Google",
            "model": "gemini-2.5-pro",
            "api_key": "your_api_key"
        }
    }
}

agent = AgenticMaid(config)
await agent.async_initialize()
```

For detailed documentation, visit: https://github.com/agenticmaid/agenticmaid
""",
    long_description_content_type="text/markdown",
    url="https://github.com/agenticmaid/agenticmaid",
    packages=[],  # Meta-package has no code
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
        "agenticmaid-core>=1.0.0",
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
        "telegram": [
            "agenticmaid-core[clients]>=1.0.0",
            "agenticmaid-clients[telegram]>=1.0.0",
        ],
        "discord": [
            "agenticmaid-core[clients]>=1.0.0",
            "agenticmaid-clients[discord]>=1.0.0",
        ],
        "solana": [
            "agenticmaid-core[triggers]>=1.0.0",
            "agenticmaid-triggers[solana]>=1.0.0",
        ],
        "messaging": [
            "agenticmaid-core[messaging]>=1.0.0",
            "agenticmaid-clients[all]>=1.0.0",
            "agenticmaid-triggers[all]>=1.0.0",
        ],
        "all": [
            "agenticmaid-core[full]>=1.0.0",
            "agenticmaid-clients[all]>=1.0.0",
            "agenticmaid-triggers[all]>=1.0.0",
            "agenticmaid-legacy>=1.0.0",
        ],
        "dev": [
            "agenticmaid-core[dev]>=1.0.0",
            "agenticmaid-clients[dev]>=1.0.0",
            "agenticmaid-triggers[dev]>=1.0.0",
            "agenticmaid-legacy[dev]>=1.0.0",
        ],
    },
    keywords=[
        "ai", "agents", "multi-agent", "mcp", "chatbot", "automation", 
        "llm", "openai", "anthropic", "google", "gemini", "claude",
        "telegram", "discord", "solana", "blockchain", "reactive", 
        "async", "framework", "ecosystem"
    ],
    project_urls={
        "Bug Reports": "https://github.com/agenticmaid/agenticmaid/issues",
        "Source": "https://github.com/agenticmaid/agenticmaid",
        "Documentation": "https://github.com/agenticmaid/agenticmaid#readme",
        "Core Package": "https://github.com/agenticmaid/agenticmaid-core",
        "Clients Package": "https://github.com/agenticmaid/agenticmaid-clients",
        "Triggers Package": "https://github.com/agenticmaid/agenticmaid-triggers",
        "Legacy Package": "https://github.com/agenticmaid/agenticmaid-legacy",
    },
)
