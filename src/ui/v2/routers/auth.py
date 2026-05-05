"""Login + logout endpoints.

The form posts to ``/api/auth/login`` which mirrors the v1 contract,
issues an HMAC-signed session cookie, and redirects back to the
dashboard root. Logout clears the cookie.
"""
from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.ui.v2 import auth as v2_auth

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    from src.ui.v2.app import get_templates
    return get_templates().TemplateResponse(
        request, "login.html", {"title": "Sign in", "error": None}
    )


@router.post("/api/auth/login")
async def login_submit(request: Request, token: str = Form(...)):
    from src.ui.v2.app import get_templates
    if not v2_auth.login_with_token(token.strip()):
        return get_templates().TemplateResponse(
            request,
            "login.html",
            {"title": "Sign in", "error": "Invalid token."},
            status_code=401,
        )
    response = RedirectResponse(url="/", status_code=303)
    secure = request.url.scheme == "https"
    v2_auth.issue_cookie(response, secure=secure)
    return response


@router.post("/api/auth/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/login", status_code=303)
    v2_auth.clear_cookie(response)
    return response
