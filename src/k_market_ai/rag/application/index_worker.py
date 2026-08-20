from k_market_ai.rag.application.index_disclosure import IndexDisclosureHandler
from k_market_ai.rag.application.index_metadata import IndexMetadataHandler


class IndexWorkerHandler:
    def __init__(
        self,
        disclosure_handler: IndexDisclosureHandler,
        metadata_handler: IndexMetadataHandler,
    ) -> None:
        self._disclosure_handler = disclosure_handler
        self._metadata_handler = metadata_handler

    async def process_next(self) -> bool:
        disclosure_processed = await self._disclosure_handler.process_next()
        metadata_processed = await self._metadata_handler.process_next()
        return disclosure_processed or metadata_processed
