"""Attach bounded feedback endpoints without loading inference models in tests."""
import hashlib

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

import acct_store
import feedback_store


class FeedbackIn(BaseModel):
    s: str = Field(default="", max_length=2000)
    anonymous: str = Field(default="", max_length=64, pattern=r"^[A-Za-z0-9_-]*$")
    rating: int = Field(ge=-1, le=1)
    question: str = Field(min_length=1, max_length=40000)
    answer: str = Field(min_length=1, max_length=200000)
    model: str = Field(default="", max_length=32)


class FeedbackListIn(BaseModel):
    s: str = Field(min_length=20, max_length=2000)
    rating: int = Field(default=0, ge=-1, le=1)


def register(app, read_session, check_rate_limit, client_ip):
    def claims_for(token):
        try:
            return read_session(token) if token else None
        except (ValueError, TypeError, UnicodeError):
            return None

    @app.post("/feedback")
    def feedback(body: FeedbackIn, request: Request):
        check_rate_limit(client_ip(request))
        claims = claims_for(body.s)
        if body.s and not claims:
            raise HTTPException(401, "다시 로그인한 뒤 평가해 주세요.")
        if not claims and len(body.anonymous) < 16:
            raise HTTPException(400, "평가 정보를 확인할 수 없어요.")
        who = "u:" + claims["u"] if claims else "a:" + body.anonymous
        rater = hashlib.sha256(who.encode()).hexdigest()
        try:
            updated = feedback_store.save(rater, body.rating, body.question, body.answer, body.model)
        except acct_store.StoreError:
            raise HTTPException(503, "평가를 저장하지 못했어요. 잠시 뒤 다시 눌러 주세요.")
        return {"ok": True, "updated": updated}

    @app.post("/feedback/list")
    def feedback_list(body: FeedbackListIn, request: Request):
        check_rate_limit(client_ip(request))
        claims = claims_for(body.s)
        if not claims or not claims.get("o"):
            raise HTTPException(403, "운영자만 볼 수 있어요.")
        try:
            return feedback_store.list_recent(body.rating)
        except acct_store.StoreError:
            raise HTTPException(503, "평가 목록을 불러오지 못했어요.")
