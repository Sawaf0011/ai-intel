from fastapi import FastAPI

from ai_intel.api.routes.query import router as query_router
from ai_intel.config import get_settings

app = FastAPI(title="AI Startup Intelligence Platform")
app.include_router(query_router)


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "app_env": settings.app_env}
