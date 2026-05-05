"""
setup.py
--------
Package installer for the Organic Acid Leaching Efficiency ML Pipeline.

Install in editable mode (recommended for development):
    pip install -e .
"""

from setuptools import setup, find_packages


def get_requirements(path: str) -> list[str]:
    """Read requirements.txt, skip comments and editable installs."""
    with open(path) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith(("#", "-e"))
        ]


setup(
    name="ml_leaching_pipeline",
    version="0.1.0",
    author="Your Name",
    author_email="your@email.com",
    description=(
        "End-to-end ML pipeline for predicting organic acid leaching "
        "efficiency of metals from spent lithium-ion batteries."
    ),
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=get_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "train=src.pipeline.training_pipeline:main",
        ]
    },
)
