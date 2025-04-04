from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from hx_stomp_simple_qa import HXStompSimpleQA

# Initialize FastAPI app
app = FastAPI(
    title="HX Stomp AI API",
    description="API for HX Stomp pedal information and AI assistance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize HX Stomp QA system
qa_system = HXStompSimpleQA()

# Load pedals data
try:
    with open('data/hx_pedals.json', 'r') as f:
        pedals_data = json.load(f)
except FileNotFoundError:
    pedals_data = []

class Question(BaseModel):
    text: str

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {
        "status": "healthy",
        "message": "API is running"
    }

@app.get("/api/pedals")
async def get_pedals():
    """
    Get the list of all pedals from the HX Stomp.
    Returns a structured response with items array and count.
    Only includes categories with subcategories.
    """
    if not pedals_data:
        raise HTTPException(status_code=404, detail="Pedals data not found")
    
    formatted_items = []
    for category in pedals_data:
        category_name = category.get("name", "Unknown")
        
        # Only process categories with subcategories
        if "subcategories" in category:
            for subcategory in category["subcategories"]:
                if "models" in subcategory:
                    for model in subcategory["models"]:
                        if isinstance(model, dict) and "name" in model:
                            formatted_items.append({
                                "id": str(len(formatted_items)),
                                "category": category_name,
                                "name": model["name"]
                            })

    return {
        "data": {
            "items": formatted_items,
            "count": len(formatted_items)
        }
    }

@app.post("/api/ai")
async def ask_ai(question: Question):
    """
    Ask a question about HX Stomp and get an AI-powered response.
    """
    try:
        response = qa_system.answer_question(question.text)
        return {
            "data": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))