# Dockerfile
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install -r requirements.txt

# Copy project files
COPY . .

# Install the package in development mode
RUN pip3 install -e .

# Set entry point
CMD ["python3", "main.py", "--help"]
