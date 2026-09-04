import json
import re
from collections.abc import Sequence
from datetime import date
from uuid import UUID

from k_market_ai.core.answer_language import AnswerLocale, resolve_answer_language
from k_market_ai.core.errors import AppError
from k_market_ai.rag.application.ports import AnswerPort, EmbeddingPort, RagRepository
from k_market_ai.rag.domain.chunker import chunk_sections, normalize_text
from k_market_ai.rag.domain.errors import RagProviderError
from k_market_ai.rag.domain.models import (
    Citation,
    FilingEvidence,
    RagAnswer,
    SearchHit,
    SelectedContext,
    SourceSection,
)

PROMPT_VERSION = "filing-grounded-v8"
MIN_RELEVANCE = 0.28
SEARCH_LIMIT = 6
REFUSAL_MESSAGE = "I could not find sufficient evidence in this filing to answer that question."


class AskDisclosureHandler:
    def __init__(
        self,
        repository: RagRepository,
        embedding: EmbeddingPort,
        answer: AnswerPort,
    ) -> None:
        self._repository = repository
        self._embedding = embedding
        self._answer = answer

    async def retrieve(
        self,
        stock_codes: Sequence[str],
        question: str,
        from_date: date | None,
        to_date: date | None,
        financials: bool,
    ) -> list[FilingEvidence]:
        candidates = await self._repository.evidence_candidates(
            stock_codes,
            from_date,
            to_date,
            financials,
        )
        if not candidates:
            return []
        query = question + (
            " 매출액 영업이익 당기순이익 연결 손익계산서 실적" if financials else ""
        )
        vector = (await self._embedding.embed([query]))[0]
        result: list[FilingEvidence] = []
        latest_overview = (
            financials
            and from_date is None
            and to_date is None
            and re.search(r"\b(recent|latest|current)\b|최근|최신", question, re.IGNORECASE)
            and not re.search(
                r"\b(compar\w*|versus|vs|trend\w*|histor\w*)\b|비교|추이|과거",
                question,
                re.IGNORECASE,
            )
        )
        for filing in candidates:
            # 최신 실적 개요는 종목별 최신 근거를 사용하고 과거 별도·잠정 실적과 혼합하지 않는다.
            if latest_overview and any(
                item.filing.stock_code == filing.stock_code for item in result
            ):
                continue
            hits = await self._repository.search(
                filing.receipt_number,
                vector,
                self._embedding.model,
                None,
                4,
            )
            selected = [h for h in hits if h.score >= MIN_RELEVANCE][:2]
            method = "CURRENT_VECTOR_CHUNKS"
            contents = [h.content for h in selected]
            section_ids = [sid for h in selected for sid in h.section_ids]
            if financials:
                sections = await self._repository.load_current_sections(filing.receipt_number)
                financial_chunks = sorted(
                    chunk_sections(sections),
                    key=lambda c: _financial_score(c.content),
                    reverse=True,
                )
                financial_chunks = [
                    c for c in financial_chunks if _financial_score(c.content) >= 10
                ][:2]
                if financial_chunks:
                    contents = [c.content for c in financial_chunks]
                    section_ids = [sid for c in financial_chunks for sid in c.section_ids]
                    method = "CURRENT_HYBRID_FINANCIAL"
                tables = _financial_tables(sections)
                if tables:
                    contents = [content for _, content in tables]
                    section_ids = [section_id for section_id, _ in tables]
                    method = "CURRENT_STRUCTURED_FINANCIAL"
            # 복구 대기 문서도 현재 원문에서 검색하며 구버전 청크나 생성한 수치를 쓰지 않는다.
            if not contents:
                sections = await self._repository.load_current_sections(filing.receipt_number)
                chunks = chunk_sections(sections)
                terms = re.findall(r"[\w]+", query.lower())
                ranked = sorted(
                    chunks,
                    key=lambda c: sum(
                        c.content.lower().count(term) for term in terms if len(term) > 1
                    ),
                    reverse=True,
                )
                selected_chunks = ranked[:2]
                contents = [c.content for c in selected_chunks]
                section_ids = [sid for c in selected_chunks for sid in c.section_ids]
                method = "CURRENT_SOURCE_LEXICAL"
            if contents:
                content = "\n\n".join(contents)[:7200]
                # 같은 종목의 정정 전후 근거가 동일하면 최신 공시 한 건만 전달한다.
                if any(
                    item.filing.stock_code == filing.stock_code and item.content == content
                    for item in result
                ):
                    continue
                result.append(
                    FilingEvidence(
                        filing,
                        content,
                        tuple(dict.fromkeys(section_ids)),
                        method,
                    )
                )
        return result

    async def ask(
        self,
        receipt_number: str,
        question: str,
        selected: SelectedContext | None,
        answer_locale: AnswerLocale = "auto",
    ) -> RagAnswer:
        answer_locale = resolve_answer_language(question, answer_locale)
        selected_text = await self._validate_selected(receipt_number, selected)
        query = question if selected_text is None else f"{question}\nSelected text: {selected_text}"
        vector = (await self._embedding.embed([query]))[0]
        hits = await self._repository.search(
            receipt_number,
            vector,
            self._embedding.model,
            selected.section_id if selected else None,
            SEARCH_LIMIT,
        )
        relevant = [hit for hit in hits if hit.selected_priority <= 1 or hit.score >= MIN_RELEVANCE]
        if not relevant:
            return _refusal(answer_locale=answer_locale)

        contexts = [(f"C{index}", hit) for index, hit in enumerate(relevant, start=1)]
        try:
            generated = await self._answer.answer(question, contexts, answer_locale)
        except RagProviderError as exception:
            raise AppError(
                code="MODEL_UNAVAILABLE",
                message="The AI service is temporarily unavailable.",
                status_code=503,
            ) from exception

        if not generated.sufficient_evidence:
            return _refusal(generated.refusal_reason, generated.model, answer_locale)

        available = {context_id: hit for context_id, hit in contexts}
        cited_ids = tuple(dict.fromkeys(generated.citation_ids))
        if not cited_ids or any(citation_id not in available for citation_id in cited_ids):
            return _refusal(
                "답변에 유효한 공시 인용이 없습니다."
                if answer_locale == "ko"
                else "The generated answer did not contain valid filing citations.",
                generated.model,
                answer_locale,
            )
        citations = tuple(
            _citation(citation_id, available[citation_id]) for citation_id in cited_ids
        )
        return RagAnswer(
            answer=generated.answer,
            refused=False,
            refusal_reason=None,
            citations=citations,
            model=generated.model,
            prompt_version=PROMPT_VERSION,
        )

    async def _validate_selected(
        self,
        receipt_number: str,
        selected: SelectedContext | None,
    ) -> str | None:
        if selected is None:
            return None
        normalized = normalize_text(selected.text)
        if not normalized or not await self._repository.selected_text_exists(
            receipt_number,
            selected.section_id,
            normalized,
            selected.translation_source_hash,
        ):
            raise AppError(
                code="INVALID_SELECTED_CONTEXT",
                message="The selected text does not belong to the current filing.",
                status_code=400,
            )
        return normalized


def _financial_score(content: str) -> int:
    text = normalize_text(content)
    if not re.search(r"\d", text):
        return 0
    # 투자회사·임원 보수 표 대신 연결 손익과 기간·단위를 함께 포함한 표를 우선한다.
    score = 5 * sum(term in text for term in ("매출액", "영업이익", "순이익", "단위"))
    score += 30 * ("연결 손익계산서" in text or "연결 포괄손익계산서" in text)
    score += 20 * ("매출총이익" in text and "법인세" in text)
    score += 12 * ("3개월" in text and "누적" in text)
    score += 8 * ("지배회사지분" in text or "비지배지분" in text)
    score -= 40 * any(
        term in text for term in ("보수의 종류", "출자목적", "최초취득", "부문 부문 합계")
    )
    return score


def _financial_tables(sections: Sequence[SourceSection]) -> list[tuple[UUID, str]]:
    candidates = []
    for index, section in enumerate(sections):
        if not section.table_rows:
            continue
        context = "\n".join(
            item.text[:700]
            for item in sections[max(0, index - 2) : index]
            if item.document_id == section.document_id
        )
        score = _financial_score(context + "\n" + section.text)
        if score < 50:
            continue
        # 압축 원문의 행·셀 순서를 보존하고 기간·단위를 표와 함께 전달한다.
        columns = _financial_columns(section.table_rows)
        if columns and len(context) + len(columns) + 100 <= 6500:
            content = (
                context + "\nColumns with explicit reporting period (original values):\n" + columns
            )
        else:
            rows = "\n".join(json.dumps(row, ensure_ascii=False) for row in section.table_rows)
            content = context + "\nTable rows (original cell order; multi-row headers):\n" + rows
        if len(content) <= 6500:
            candidates.append((score, section.id, content))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(section_id, content) for _, section_id, content in candidates[:1]]


def _financial_columns(rows: tuple[tuple[str, ...], ...]) -> str | None:
    # 반복되는 3개월·누적 헤더가 명확한 표만 전치해 전년 수치와 당기 수치의 혼동을 줄인다.
    if len(rows) < 3 or not rows[0] or rows[0][0].strip():
        return None
    periods = rows[0][1:]
    if not periods or not all(period.strip() for period in periods):
        return None
    expected = ("3개월", "누적") * len(periods)
    if tuple(cell.strip() for cell in rows[1]) != expected:
        return None
    width = len(expected) + 1
    if any(len(row) != width for row in rows[2:]):
        return None
    columns = []
    for index, interval in enumerate(expected):
        values = [(row[0], row[index + 1]) for row in rows[2:]]
        columns.append(
            json.dumps(
                {"period": periods[index // 2], "interval": interval, "rows": values},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return "\n".join(columns)


def _citation(citation_id: str, hit: SearchHit) -> Citation:
    excerpt = hit.content if len(hit.content) <= 500 else f"{hit.content[:497]}..."
    return Citation(
        id=citation_id,
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        document_version=hit.document_version,
        section_ids=hit.section_ids,
        first_ordinal=hit.first_ordinal,
        last_ordinal=hit.last_ordinal,
        heading=hit.heading,
        excerpt=excerpt,
    )


def _refusal(
    reason: str | None = None, model: str | None = None, answer_locale: AnswerLocale = "en"
) -> RagAnswer:
    return RagAnswer(
        answer="이 공시에서 질문에 답할 충분한 근거를 찾지 못했습니다."
        if answer_locale == "ko"
        else REFUSAL_MESSAGE,
        refused=True,
        refusal_reason=reason
        or (
            "질문과 충분히 관련된 공시 근거가 검색되지 않았습니다."
            if answer_locale == "ko"
            else "No sufficiently relevant filing evidence was retrieved."
        ),
        citations=(),
        model=model,
        prompt_version=PROMPT_VERSION,
    )
