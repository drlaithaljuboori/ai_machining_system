# setup.py
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Check CUDA availability
torch_version = torch.__version__
cuda_available = torch.cuda.is_available()

def get_requirements():
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    return requirements

setup(
    name="ai-machining-system",
    version="1.0.0",
    description="AI-Powered Geometric Analysis & Curvature Classification for Metal Machining",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires=">=3.8,<3.11",  # Kaolin works with Python 3.8-3.10
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="ai machining cad cam geometry analysis",
    include_package_data=True,
)
