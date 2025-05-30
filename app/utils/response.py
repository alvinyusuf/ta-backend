from fastapi.responses import JSONResponse
from typing import Any, Optional

def success_response(
        message: str,
        data: Optional[Any] = None,
        status_code: int = 200
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "success",
            "message": message,
            "data": data
        }
    )

def error_response(
        message: str,
        detail: Optional[str] = None,
        status_code: int = 400
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": message,
            "detail": detail
        }
    )