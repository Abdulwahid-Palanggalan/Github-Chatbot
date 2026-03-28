import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from chatbot import load_kb, prepare_kb_embeddings, search_knowledge_base, generate_response

app = FastAPI(title="GitHub Knowledge Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing AI Knowledge Base...")
try:
    kb = load_kb()
    kb_indexed = prepare_kb_embeddings(kb)
    print("Knowledge base ready!")
except Exception as e:
    print(f"Failed to load knowledge base: {e}")
    kb_indexed = []

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        match = search_knowledge_base(request.message, kb_indexed)
        response_text = generate_response(request.message, match)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

os.makedirs("frontend", exist_ok=True)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def read_index():
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
