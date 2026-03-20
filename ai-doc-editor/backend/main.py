from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.ai_routes import router as ai_router
from routes.document_routes import router as document_router
from routes.user_routes import router as user_router

app = FastAPI(title="AI Doc Editor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router, prefix="/ai", tags=["ai"])
app.include_router(document_router, prefix="/documents", tags=["documents"])
app.include_router(user_router, prefix="/users", tags=["users"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
