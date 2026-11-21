"""The BankGuard scoring service."""

from fastapi import FastAPI

app = FastAPI(
    title="BankGuard",
    description="Scores a bank transfer for money laundering and shows the past transfers behind the score.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by the host and by Kubernetes."""
    return {"status": "ok"}