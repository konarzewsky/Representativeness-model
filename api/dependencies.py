from hmac import compare_digest

from fastapi import HTTPException, Request

from api.env import API_AUTH_TOKEN


async def verify_auth_token(request: Request, token_header: str = "Auth-Token"):
    if not request.headers.get(token_header):
        raise HTTPException(
            status_code=400, detail=f"{token_header} header not provided"
        )
    elif not (
        compare_digest(request.headers.get(token_header), API_AUTH_TOKEN)  # type: ignore
        and request.headers.get(token_header) == API_AUTH_TOKEN
    ):
        raise HTTPException(status_code=401, detail=f"Invalid {token_header}")
