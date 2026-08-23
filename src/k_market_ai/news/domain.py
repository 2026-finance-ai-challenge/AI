from dataclasses import dataclass
from enum import StrEnum


class NewsSentiment(StrEnum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"


class NewsImportance(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MarketImpact(StrEnum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class NewsAnalysis:
    english_title: str
    translated_paragraphs: tuple[str, ...]
    what: str
    why: str
    impact: str
    event_type: str
    sentiment: NewsSentiment
    importance: NewsImportance
    market_impact: MarketImpact
    market_impact_importance: NewsImportance
    market_impact_score: float
    event_confidence: float
    sentiment_confidence: float
    importance_confidence: float
    market_impact_confidence: float
    model: str
    prompt_version: str


@dataclass(frozen=True, slots=True)
class TermEvidence:
    id: str
    title: str
    content: str
    source_url: str | None


@dataclass(frozen=True, slots=True)
class TermExplanation:
    normalized_term: str | None
    definition: str | None
    contextual_meaning: str | None
    evidence_ids: tuple[str, ...]
    confidence: float
    review_required: bool
    sufficient_evidence: bool
    refusal_reason: str | None
    model: str
    prompt_version: str
