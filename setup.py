from setuptools import setup, find_packages
from pathlib import Path


# Dynamically generate install_requires from requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with Path(filename).open() as req_file:
        return [
            line.strip()
            for line in req_file
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="research_library",
    version="0.0.1",
    description="A project for processing order book data and building predictive models.",
    author="Ricardo Roche",
    author_email="reroche@pm.me",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
