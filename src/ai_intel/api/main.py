from fastapi import FastAPI

from ai_intel.config import get_settings

app = FastAPI(title="AI Startup Intelligence Platform")


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "app_env": settings.app_env}
