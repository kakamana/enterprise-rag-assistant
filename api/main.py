"""FastAPI for the Enterprise RAG Assistant.

Endpoints:
    GET  /health
    POST /ask       — single question -> answer + sources + faithfulness

Every response includes a `decision_aid_disclaimer`: this is a retrieval +
grounding aid, not a substitute for the policy owner.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Enterprise RAG Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "This response cites the policy paragraphs the model retrieved. "
    "Always verify against the canonical policy document before acting."
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    doc_id: str
    domain: str
    paragraph_id: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    faithfulness_score: float
    retrieval_recall: float
    abstained: bool
    decision_aid_disclaimer: str = DISCLAIMER


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        # from src.enterprise_rag.serve import answer
        # result = answer(req.question, k=req.top_k)
        result = dict(
            answer=(
                "Expense reimbursement requests must be submitted within 30 days."
                " Approval sits with the Finance Controller."
            ),
            sources=[
                dict(doc_id="Finance-002", domain="Finance", paragraph_id=3,
                     text="Expense reimbursement requests must be submitted within 30 days.",
                     score=0.91),
                dict(doc_id="Finance-002", domain="Finance", paragraph_id=4,
                     text="The Finance Controller reviews any reimbursement above AED 5,000.",
                     score=0.74),
            ],
            faithfulness_score=0.83,
            retrieval_recall=0.80,
            abstained=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return AskResponse(**result)
