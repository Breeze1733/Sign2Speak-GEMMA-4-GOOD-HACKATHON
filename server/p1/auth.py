"""Shared bearer-token check for the P1/P2 service."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status

TOKEN = os.getenv("S2S_TOKEN", "")


def require_token(authorization: str | None = Header(default=None)) -> None:
    if not TOKEN:
        return
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="bad token"
        )
