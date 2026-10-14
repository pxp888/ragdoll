from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

# Swagger / OpenAPI security scheme definition
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API Key for multi-user authentication (any string value)",
)
api_key_header_alt = APIKeyHeader(
    name="api-key",
    auto_error=False,
    description="Alternative API Key header",
)
api_key_header_alt2 = APIKeyHeader(
    name="apikey",
    auto_error=False,
    description="Alternative API Key header",
)
authorization_header = APIKeyHeader(
    name="Authorization",
    auto_error=False,
    description="Bearer token or API Key header",
)
api_key_query = APIKeyQuery(
    name="api_key",
    auto_error=False,
    description="API Key query parameter",
)
api_key_query_alt = APIKeyQuery(
    name="apikey",
    auto_error=False,
    description="Alternative API Key query parameter",
)


def get_api_key(
    x_api_key: Optional[str] = Security(api_key_header),
    alt_api_key: Optional[str] = Security(api_key_header_alt),
    alt2_api_key: Optional[str] = Security(api_key_header_alt2),
    auth_header: Optional[str] = Security(authorization_header),
    q_api_key: Optional[str] = Security(api_key_query),
    q_apikey: Optional[str] = Security(api_key_query_alt),
) -> str:
    """
    Validates and extracts the API key from request headers or query parameters.
    The API key can be any non-empty string and represents the user identifier.
    """
    key = x_api_key or alt_api_key or alt2_api_key or q_api_key or q_apikey
    if not key and auth_header:
        if auth_header.lower().startswith("bearer "):
            key = auth_header[7:].strip()
        else:
            key = auth_header.strip()

    if not key or not key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Please provide an API key via 'X-API-Key' header, 'Authorization: Bearer <key>', or 'api_key' query parameter.",
        )
    return key.strip()
