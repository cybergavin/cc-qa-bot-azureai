
import uvicorn
import toml
import logging

from types import MappingProxyType
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from cc_rag import get_rag_response

# Load TOML config and return an immutable MappingProxyType
with open("config.toml", "r", encoding="utf-8") as file:
    CONFIG = MappingProxyType(toml.load(file))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Set up templates directory
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    query: str
    model: str

# Display Home page
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Renders the HTML template for the user interface with dynamic model options."""
    
    models = [
        {"display_name": model_data["display_name"], "deployment_name": model_data["deployment_name"]}
        for model_data in CONFIG["azure-ai"]["model"].values()
    ]

    return templates.TemplateResponse("index.html", {"request": request, "models": models})


# Process Query
@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.query
    model = request.model

    if not query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")
    
    logger.info(f"Received query: {query} | Model: {model}")
    
    try:
        response, prompt_tokens, completion_tokens = get_rag_response(query, model)

         # Handle case where response is an error message
        if isinstance(response, str) and "error" in response.lower():
            raise HTTPException(status_code=500, detail=response)
        
        # Validate token values
        prompt_tokens = max(prompt_tokens, 1)  # Prevent division by zero
        completion_tokens = max(completion_tokens, 1)
        data_source = CONFIG["confluence"]["wiki_url"]
        model_data = CONFIG["azure-ai"]["model"].get(model, None)
        if model_data:       
            input_cost = model_data["input_token_cost"]
            output_cost = model_data["output_token_cost"]
            token_cost_batch = model_data["token_cost_batch"]
        else:
            print(f"Model '{model}' not found in TOML configuration.")

        input_tokens_cost = f"{(( prompt_tokens / token_cost_batch) * input_cost):.4f}"
        output_tokens_cost = f"{(( completion_tokens / token_cost_batch) * output_cost):.4f}"

        return JSONResponse({
            "answer": response,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "input_tokens_cost": input_tokens_cost,
            "output_tokens_cost": output_tokens_cost,
            "data_source": data_source
        })
    except HTTPException as http_error:
        raise http_error  # Return proper HTTP errors

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again later.")
    
if __name__ == "__main__":
    try:
        uvicorn.run("cc_webapp:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.error(f"Unexpected error occurred. Please try again later.: {e}")