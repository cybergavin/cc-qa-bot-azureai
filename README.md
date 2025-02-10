# Confluence Cloud RAG System

This repository contains a Retrieval-Augmented Generation (RAG) system for indexing and querying Confluence wiki pages using Azure AI and Qdrant vector database.

## Quick Demo


https://github.com/user-attachments/assets/822ffc48-0d65-4c26-9d66-f995b7f7947d


## Features
- **Confluence Page Extraction**: Retrieves and processes pages from a Confluence space.
- **Text Chunking**: Splits content into manageable chunks for efficient embedding.
- **Vector Storage**: Stores embeddings in Qdrant for fast retrieval.
- **Azure AI Integration**: Uses Azure OpenAI models for embeddings and text generation.
- **Web Interface**: Provides a FastAPI-based web application for querying indexed data.

## Installation
### Prerequisites
- Python 3.12+
- `pip` and `venv` or `uv`
- Confluence API token
- Azure AI API key
- Qdrant server

### Setup
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository>
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy the sample config and update configuration:
   ```sh
   cp config.toml.sample config.toml
   nano config.toml  # Edit with your details
   ```
4. Copy the sample env file and update environment variables:
   ```sh
   cp env.sample .env
   nano .env  # Edit with your details
   ```

### Launch Qdrant Database
Before running the system, create a local directory for Qdrant storage and start the database:
```sh
mkdir -p qdrant_storage
podman run -d -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```
Access the Qdrant dashboard at `http://localhost:6333/dashboard`

## Usage
### Index Confluence Pages
Run the script to fetch and store Confluence page embeddings in Qdrant:
```sh
python cc_index.py
```

### Start the Web Application
Launch the FastAPI-based web UI:
```sh
uvicorn cc_webapp:app --host 0.0.0.0 --port 8000
```
Access the web UI at `http://localhost:8000`

### Query the System via API
Send a POST request to retrieve answers from indexed documents:
```sh
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" \
-d '{"query": "What is the company policy on remote work?", "model": "<model deployment name>"}'
```
**NOTE:** Replace &lt;model deployment name&gt; in the above `curl` command.

## Configuration
Edit `config.toml` to update parameters like:
- **Confluence credentials**
- **Chunking strategy**
- **Azure AI model settings**
- **Qdrant database connection**

## Architecture
1. **Data Ingestion**
   - Fetches Confluence pages
   - Cleans and tokenizes text
   - Splits into chunks
   - Creates embeddings for chunks via Azure AI
   - Stores in Qdrant
2. **Query Processing**
   - Receives user query
   - Creates embeddings for query via Azure AI
   - Retrieves relevant documents from Qdrant
   - Uses Azure AI to generate a response

## License
MIT License

## Contributors
- Cybergavin
