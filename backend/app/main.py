from fastapi import FastAPI

app = FastAPI(
    title="AI Factory Operations Agent",
    description=(
        "An AI-assisted factory operations platform "
        "for retrieving procedures and supporting "
        "operational decision-making."
    ),
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Return the application's health status."""
    return {
        "status": "healthy",
        "application": "ai-factory-operations-agent",
    }