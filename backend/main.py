import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Allow importing project-level modules when running uvicorn backend.main:app
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from orchestrator import Orchestrator


app = FastAPI(title="FinSight Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator(dataset_dir=str(PROJECT_ROOT / "datasets"))


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "FinSight Orchestrator backend is running.",
        "usage": "POST /ask with {'question': 'your question'}"
    }


@app.post("/ask")
def ask(req: QuestionRequest) -> Dict[str, Any]:
    return orchestrator.handle_question(req.question)


@app.get("/health")
def health():
    return {"status": "ok"}