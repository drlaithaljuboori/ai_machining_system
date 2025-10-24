# setup.py
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ai-machining-system",
    version="1.0.0",
    description="AI-Powered Geometric Analysis for Metal Machining",
    author="Your Name",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)
