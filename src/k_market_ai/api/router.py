from fastapi import APIRouter

from k_market_ai.api.routes.agent import router as agent_router
from k_market_ai.api.routes.disclosure_insight import router as disclosure_insight_router
from k_market_ai.api.routes.disclosure_rag import router as disclosure_rag_router
from k_market_ai.api.routes.health import router as health_router
from k_market_ai.api.routes.news import router as news_router
from k_market_ai.api.routes.peers import router as peers_router
from k_market_ai.api.routes.tax import router as tax_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(agent_router)
api_router.include_router(disclosure_rag_router)
api_router.include_router(disclosure_insight_router)
api_router.include_router(news_router)
api_router.include_router(peers_router)
api_router.include_router(tax_router)
