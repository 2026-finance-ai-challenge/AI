from k_market_ai.rag.infrastructure.local_embedding import (
    EMBEDDING_DIMENSIONS,
    MODEL_NAME,
    MODEL_REVISION,
    LocalEmbeddingAdapter,
)


def test_embedding_contract_is_fixed_to_versioned_local_model() -> None:
    adapter = LocalEmbeddingAdapter()

    assert adapter.dimensions == EMBEDDING_DIMENSIONS == 384
    assert adapter.model == f"{MODEL_NAME}@{MODEL_REVISION}"
