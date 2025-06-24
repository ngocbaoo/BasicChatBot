# Vietnamese PDF RAG Chatbot

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload a **Vietnamese PDF document**, ask natural language questions, and receive contextually accurate answers.

## Built with:
- **LangChain** for RAG orchestration
- **HuggingFace Embeddings** for semantic understanding (Vietnamese bi-encoder)
- **Vicuna 7B** as the local language model (quantized)
- **Chroma** vector store for efficient document retrieval
- **Streamlit** for a clean, interactive user interface

## Features
- Upload and process your own PDF file
- Ask questions in Vietnamese
- Get accurate, document-grounded answers
- All done locally with your own models

## Installation
Create a virtual environment and install dependencies:
```bash
conda create -n rag-chatbot python=3.10
conda activate rag-chatbot
pip install -r requirements.txt
