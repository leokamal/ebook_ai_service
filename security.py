import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException


def verify_admin_token(x_admin_token: Optional[str] = Header(default=None)):
    """FastAPI dependency: require header X-Admin-Token == $ADMIN_TOKEN."""
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Admin endpoints disabled: ADMIN_TOKEN not configured")
    if not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="Invalid admin token")
