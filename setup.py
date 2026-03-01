"""
TorchAgentic - A PyTorch-based library for building AI agents and agentic workflows.

This library provides a comprehensive framework for creating, managing, and orchestrating
AI agents with support for tool calling, memory management, and multi-agent workflows.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read version from package
version = "0.1.0"

setup(
    name="torchagentic",
    version=version,
    author="Liodon AI",
    author_email="contact@liodon.ai",
    description="A PyTorch-based library for building AI agents and agentic workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liodon-ai/torchagentic",
    project_urls={
        "Bug Tracker": "https://github.com/liodon-ai/torchagentic/issues",
        "Documentation": "https://github.com/liodon-ai/torchagentic#readme",
        "Source Code": "https://github.com/liodon-ai/torchagentic",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "llm": [
            "transformers>=4.35.0",
            "accelerate>=0.24.0",
            "sentencepiece>=0.1.99",
        ],
        "embeddings": [
            "sentence-transformers>=2.2.0",
        ],
        "full": [
            "transformers>=4.35.0",
            "accelerate>=0.24.0",
            "sentencepiece>=0.1.99",
            "sentence-transformers>=2.2.0",
            "aiohttp>=3.9.0",  # For async HTTP requests
            "pyyaml>=6.0",     # For YAML config files
        ],
    },
    keywords=[
        "ai",
        "agents",
        "pytorch",
        "llm",
        "machine-learning",
        "deep-learning",
        "agentic",
        "workflow",
        "automation",
        "tool-calling",
        "function-calling",
    ],
    include_package_data=True,
    package_data={
        "torchagentic": ["py.typed"],
    },
    zip_safe=False,
)
