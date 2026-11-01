"""FastAPI application entry point for the Support Triage Agent."""

from fastapi import FastAPI

app = FastAPI(
    title="support-triage-agent",
    description="SaaS support-ticket triage and RAG auto-reply agent API",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
