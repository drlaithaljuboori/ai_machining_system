# setup.py
from setuptools import setup, find_packages

def get_requirements():
    """Read requirements from requirements.txt"""
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    
    # Filter out any problematic lines
    filtered_requirements = []
    for req in requirements:
        req = req.strip()
        if req and not req.startswith('--') and not req.startswith('#'):
            filtered_requirements.append(req)
    
    return filtered_requirements

setup(
    name="ai-machining-system",
    version="1.0.0",
    description="AI-Powered Geometric Analysis & Curvature Classification for Metal Machining",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires=">=3.8,<3.12",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="ai machining cad cam geometry analysis",
)
