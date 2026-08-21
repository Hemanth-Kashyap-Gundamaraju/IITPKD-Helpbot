# tests/conftest.py
import os

# Set dummy values before any project modules import config.py
os.environ["PINECONE_API_KEY"] = "fake-key-for-testing"