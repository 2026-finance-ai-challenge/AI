import asyncio
from types import SimpleNamespace

import pytest
from openai import OpenAIError

from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError
from k_market_ai.peers.service import GlobalPeerService


def test_global_peer_service_keeps_ranked_facts_and_disables_provider_storage() -> None:
    responses = FakeResponses()
    service = GlobalPeerService(
        SimpleNamespace(responses=responses),
        Settings(
            environment="test",
            peer_model="test-peer-model",
            peer_prompt_version="peer-test-v2",
        ),
    )

    result = asyncio.run(service.analyze("005930", "a" * 64))

    assert result.primary_peer.ticker == "INTC"
    assert tuple(item.dimension for item in result.comparisons) == (
        "overall_business",
        "semiconductor",
        "memory",
    )
    assert len(result.key_strengths) == 4
    assert result.prompt_version == "peer-test-v2"
    assert responses.arguments["store"] is False
    assert responses.arguments["safety_identifier"] == "a" * 64
    assert "658355740000" in str(responses.arguments["input"])


def test_global_peer_service_refuses_stock_without_validated_catalog_data() -> None:
    service = GlobalPeerService(
        SimpleNamespace(responses=FakeResponses()),
        Settings(environment="test"),
    )

    with pytest.raises(AppError) as error:
        asyncio.run(service.analyze("0126Z0", "a" * 64))

    assert error.value.code == "GLOBAL_PEER_DATA_UNAVAILABLE"


def test_global_peer_service_uses_verified_catalog_when_provider_is_unavailable() -> None:
    service = GlobalPeerService(
        SimpleNamespace(responses=UnavailableResponses()),
        Settings(environment="test", peer_prompt_version="peer-test-v2"),
    )

    result = asyncio.run(service.analyze("005930", "a" * 64))

    assert len(result.comparisons) == 3
    assert all(item.peer.logo_url for item in result.comparisons)
    assert all(len(item.description.split()) <= 36 for item in result.comparisons)
    assert result.source == "KMARKET_GLOBAL_PEER_VERIFIED_CATALOG"


class FakeResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.arguments = arguments
        return SimpleNamespace(
            output_parsed=SimpleNamespace(
                headline="Samsung Electronics and its closest global peers",
                summary=(
                    "Intel is the closest supplied semiconductor reference. "
                    "The comparison is informational and not a one-for-one valuation substitute."
                ),
                comparisons=(
                    SimpleNamespace(
                        dimension="overall_business",
                        description="Intel is the supplied overall business reference.",
                    ),
                    SimpleNamespace(
                        dimension="semiconductor",
                        description="TSMC is the supplied foundry-focused reference.",
                    ),
                    SimpleNamespace(
                        dimension="memory",
                        description="Micron is the supplied memory-focused reference.",
                    ),
                ),
                key_strengths=(
                    SimpleNamespace(
                        title="AI Technology",
                        description="The supplied profile identifies AI infrastructure.",
                        icon_key="ai",
                    ),
                    SimpleNamespace(
                        title="Consumer Devices",
                        description="The supplied profile identifies consumer devices.",
                        icon_key="consumer_electronics",
                    ),
                    SimpleNamespace(
                        title="Foundry Capability",
                        description="The supplied profile identifies foundry capability.",
                        icon_key="foundry",
                    ),
                    SimpleNamespace(
                        title="Memory Technology",
                        description="The supplied profile identifies memory technology.",
                        icon_key="memory",
                    ),
                ),
            )
        )


class UnavailableResponses:
    async def parse(self, **arguments: object) -> SimpleNamespace:
        raise OpenAIError("provider timeout")
