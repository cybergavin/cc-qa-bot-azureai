import os
import re
import time
import toml
import uuid
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from itertools import islice
from atlassian import Confluence
from bs4 import BeautifulSoup
from types import MappingProxyType
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load TOML config and return an immutable MappingProxyType
with open("config.toml", "r", encoding="utf-8") as file:
    CONFIG = MappingProxyType(toml.load(file))

# Constants for batch execution
BATCH_SIZE = 10  
MAX_WORKERS = 5  # Parallel execution

# Load and retrieve environment variables from .env file
load_dotenv()
confluence_api_token = os.getenv('CONFLUENCE_API_TOKEN')
azure_ai_key = os.getenv('AZURE_AI_KEY')

# Create Confluence instance
confluence = Confluence(
        url=CONFIG["confluence"]["wiki_url"],
        username=CONFIG["confluence"]["username"],
        password=confluence_api_token
    )   

def get_confluence_pages(confluence: Confluence, space: str, excluded_folders: set = None) -> List[Dict]:
    """
    Retrieve pages from a specific Confluence space, excluding specified folders efficiently.
    """
    confluence_page_limit = CONFIG["confluence"]["page_limit"]
    confluence_retry_delay = CONFIG["confluence"]["retry_delay"]
    if excluded_folders is None:
        excluded_folders = set()   
    all_pages = []
    start = 0
    # Fetch all pages in a paginated manner
    while True:
        try:
            pages = confluence.get_all_pages_from_space(space=space, start=start, limit=confluence_page_limit)
            if not pages:
                break            
            all_pages.extend(pages)
            start += confluence_page_limit
        except Exception as e:
            print(f"Error fetching pages, retrying in {confluence_retry_delay} seconds... Error: {e}")
            time.sleep(confluence_retry_delay)   
    # Optimized filtering: Filter pages before making extra API calls
    filtered_pages = [] 
    for page in all_pages:
        page_id = page.get("id")        
        try:
            # Fetch ancestors for the page in bulk
            ancestors = confluence.get_page_by_id(page_id, expand="ancestors").get("ancestors", [])
            ancestor_titles = {ancestor.get("title") for ancestor in ancestors}
            # Skip pages belonging to excluded folders
            if not ancestor_titles.intersection(excluded_folders):
                filtered_pages.append(page)
        except Exception as e:
            print(f"Error fetching ancestors for page {page_id}: {e}")
    print(f"Retrieved {len(all_pages)} total pages, {len(filtered_pages)} after filtering.")
    return filtered_pages

def clean_html(raw_html):
    """
    Cleans Confluence wiki pages by removing HTML tags, excessive whitespace,
    and preserving meaningful text structure.
    """
    soup = BeautifulSoup(raw_html, "html.parser")   
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()  
    text = soup.get_text(separator=" ")  # Preserve spacing
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace  
    return text

def split_text(text, max_tokens=400):
    """
    Optimized text splitting for Confluence wiki pages.
    Uses RecursiveCharacterTextSplitter to preserve semantic structure.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunking"]["max_tokens_per_chunk"] * CONFIG["chunking"]["characters_per_token"], 
        chunk_overlap=CONFIG["chunking"]["chunk_overlap"],  # Ensure context retention between chunks
        separators=["\n\n", "\n", " ", ""],  # Prioritize paragraph, line, and word breaks
    )
    return text_splitter.split_text(text)

def get_embedding(text: str) -> list:
    """
    Get text embeddings using Azure OpenAI.
    Adjust parameters based on your deployment and model.
    """
    client = EmbeddingsClient(
        endpoint=CONFIG["azure-ai"]["endpoint"],
        credential=AzureKeyCredential(azure_ai_key),
    )
    response = client.embed(
        model=CONFIG["azure-ai"]["embedding_model"], 
        input=text
    )
    embedding = response.data[0].embedding
    return embedding

def initialize_qdrant_collection():
    """Initialize Qdrant vector database"""
    # Retrieve config
    host = CONFIG["vectordb"]["qdrant"]["host"]
    collection = CONFIG["vectordb"]["qdrant"]["collection"]
    embedding_dimensions = CONFIG["vectordb"]["qdrant"]["embedding_dimensions"]
    vector_distance = CONFIG["vectordb"]["qdrant"]["vector_distance"]

    # Create Qdrant client
    client = QdrantClient(host=host)
    # Check if the collection exists. If not, create it.
    if not client.collection_exists(collection_name=collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=embedding_dimensions,
                distance=vector_distance
            )
        )
        print(f"Collection '{collection}' created.")
    else:
        print(f"Collection '{collection}' already exists.")
    return client

def batch(iterable, size):
    """Helper to create batches from an iterable"""
    iterator = iter(iterable)
    return iter(lambda: list(islice(iterator, size)), [])

def process_chunk(chunk_data: Dict):
    """Process a single chunk and return embedding with metadata."""
    try:
        embedding = get_embedding(chunk_data['text'])
        return PointStruct(
            id=chunk_data['point_id'],
            vector=embedding,
            payload={
                'page_id': chunk_data['page_id'],
                'title': chunk_data['title'],
                'text': chunk_data['text']
            }
        )
    except Exception as e:
        print(f"Error processing chunk {chunk_data['point_id']}: {e}")
        return None

def index_confluence_data(client: QdrantClient):
    """Upsert vector database with chunk embeddings"""
    pages = get_confluence_pages(confluence, CONFIG["confluence"]["space"])
    points = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for page in pages:
            page_id = page.get("id")
            title = page.get("title")            
            page_details = confluence.get_page_by_id(page_id, expand="body.storage")
            raw_content = page_details.get("body", {}).get("storage", {}).get("value", "")
            text = clean_html(raw_content)
            chunks = split_text(text)
            # Process chunks in parallel
            chunk_data_list = [{
                'point_id': str(uuid.uuid4()),  # Ensuring unique ID
                'page_id': page_id,
                'title': title,
                'text': chunk
            } for chunk in chunks]
            for chunk_batch in batch(chunk_data_list, BATCH_SIZE):
                results = list(executor.map(process_chunk, chunk_batch))
                valid_points = [res for res in results if res]
                if valid_points:
                    client.upsert(collection_name=CONFIG["vectordb"]["qdrant"]["collection"], points=valid_points)
                    points.extend(valid_points)
                time.sleep(0.5)  # Adaptive rate limiting
    print(f"Indexed {len(points)} chunks into Qdrant.")
    return len(points)

if __name__ == "__main__":
    client = initialize_qdrant_collection()
    index_confluence_data(client)