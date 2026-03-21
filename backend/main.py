from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.ai_routes import router as ai_router
from routes.document_routes import router as document_router
from routes.user_routes import router as user_router

from services.embedding_index_queue import (
    start_embedding_indexer,
    stop_embedding_indexer,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start async embedding indexer so document indexing does not block inference requests.
    start_embedding_indexer(num_workers=1)
    try:
        yield
    finally:
        stop_embedding_indexer()


app = FastAPI(title="AI Doc Editor API", version="1.0.0", lifespan=lifespan)

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
