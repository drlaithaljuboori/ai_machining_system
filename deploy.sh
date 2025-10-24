#!/bin/bash

# Deploy to Railway/Render/Hugging Face Spaces
echo "Deploying AI Machining System..."

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port $PORT
