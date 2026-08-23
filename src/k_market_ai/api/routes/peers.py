from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from k_market_ai.api.internal_auth import authenticate_internal
from k_market_ai.core.errors import AppError
from k_market_ai.peers.service import GlobalPeerAnalysis, GlobalPeerService

router = APIRouter(prefix="/internal/v1/peers", tags=["global-peers"])


class PeerAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safety_identifier: str = Field(pattern=r"^[0-9a-f]{64}$")


@router.post("/{stock_code}", response_model=GlobalPeerAnalysis)
async def analyze(
    request: Request,
    stock_code: str,
    body: PeerAnalysisRequest,
    _: Annotated[None, Depends(authenticate_internal)],
) -> GlobalPeerAnalysis:
    if len(stock_code) != 6 or not stock_code.isalnum() or stock_code.upper() != stock_code:
        raise AppError(
            code="INVALID_STOCK_CODE",
            message="The stock code is invalid.",
            status_code=422,
        )
    return await _service(request).analyze(stock_code, body.safety_identifier)


def _service(request: Request) -> GlobalPeerService:
    service: object | None = getattr(request.app.state, "global_peer_service", None)
    if service is None:
        raise AppError(
            code="GLOBAL_PEER_SERVICE_NOT_CONFIGURED",
            message="The global peer service is not configured.",
            status_code=503,
        )
    return cast(GlobalPeerService, service)
