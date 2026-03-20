from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_style_profiles: dict[str, dict] = {}


class StyleProfilePayload(BaseModel):
    user_id: str
    avg_sentence_length: float
    tone: str
    vocabulary_complexity: float


@router.post("/style-profile")
async def upsert_style_profile(payload: StyleProfilePayload) -> dict:
    _style_profiles[payload.user_id] = payload.model_dump()
    return {"ok": True, "style_profile": _style_profiles[payload.user_id]}


@router.get("/{user_id}/style-profile")
async def get_style_profile(user_id: str) -> dict:
    return {"style_profile": _style_profiles.get(user_id)}
