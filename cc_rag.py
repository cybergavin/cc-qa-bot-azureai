import os
import logging
import requests
import toml

from types import MappingProxyType
from dotenv import load_dotenv
from wrapt_timeout_decorator import *
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.ai.inference.models import SystemMessage, UserMessage
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from cc_parse_index import get_embedding, initialize_qdrant_collection

# Load TOML config and return an immutable MappingProxyType
with open("config.toml", "r", encoding="utf-8") as file:
    CONFIG = MappingProxyType(toml.load(file))

# Load and retrieve environment variables from .env file
load_dotenv()
azure_ai_key = os.getenv('AZURE_AI_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve config
azure_ai_endpoint = CONFIG["azure-ai"]["endpoint"]
collection = CONFIG["vectordb"]["qdrant"]["collection"]
top_k = CONFIG["azure-ai"]["models"]["top_k"]
top_p = CONFIG["azure-ai"]["models"]["top_p"]
temperature = CONFIG["azure-ai"]["models"]["temperature"]
max_tokens = CONFIG["azure-ai"]["models"]["max_tokens"]

@timeout(30, use_signals=False)
def answer_query(query: str, client: QdrantClient, model: str) -> str:
    try:
        # Get the query embedding
        query_embedding = get_embedding(query)
        
        # Search in Qdrant for the most relevant document chunks
        search_result = client.query_points(
            collection_name=collection,
            query=query_embedding,
            limit=top_k
        ).points

        if not search_result:
            logger.warning("No relevant documents found for query: %s", query)
            return "Sorry, we couldn't find relevant information. Please try again later."
        
        # Combine retrieved texts to form a context for the prompt
        context_texts = [item.payload.get("text", "") for item in search_result]
        context = "\n\n".join(context_texts)
        
        # Construct a prompt for Azure AI
        prompt = (
            "Use only the following context to answer the question. If you cannot find the answer in the context, then say that you don't know. Your answer must be concise and only based on the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        
        # Initialize Azure client
        client = ChatCompletionsClient(
            endpoint=azure_ai_endpoint,
            credential=AzureKeyCredential(azure_ai_key),
        )
        
        # Make the API call with a timeout of 10 seconds
        try:
            response = client.complete(
                model=model,
                messages=[
                    SystemMessage("You are a helpful assistant."),
                    UserMessage(prompt)
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            # Extract and return the answer
            answer = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            return answer, prompt_tokens, completion_tokens
        
        # When handling exceptions, return 0 prompt tokens and 0 completion tokens along with the error message.
        except requests.Timeout:
            logger.error("Azure API request timed out.")
            return "The request timed out at 15 seconds. Please try again later.", 0, 0
        
        except AzureError as e:
            logger.error("Azure API error occurred: %s", e)
            return "Sorry, we're having trouble processing your request. Please try again later.", 0, 0

        except Exception as e:
            logger.error("Unexpected error occurred: %s", e)
            return "An unexpected error occurred. Please try again later.", 0, 0

    except TimeoutError:
        logger.error("Function execution timed out after 15 seconds.")
        return "Processing timed out at 15 seconds. Please try again later.", 0, 0
         
    except Exception as e:
        logger.error("Failed to process the query: %s", e)
        return "We encountered an issue while processing your query. Please try again later.", 0, 0

def get_rag_response(query:str, model:str):
    client = initialize_qdrant_collection()
    answer, prompt_tokens, completion_tokens = answer_query(query, client, model)
    return answer, prompt_tokens, completion_tokens