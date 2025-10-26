from __future__ import annotations

import logging
from typing import Any, Dict, List

from graph.state import QAState
from service.rag.augmentation.augmenter import DocumentAugmenter
from service.rag.augmentation.formatters import MarkdownFormatter, PromptFormatter

logger = logging.getLogger(__name__)


def _ensure_content_field(documents: List[Dict[str, Any]]) -> None:
    """DocumentAugmenter가 기대하는 'content' 필드를 채워 넣는다."""
    for doc in documents:
        if not doc.get("content"):
            doc["content"] = doc.get("chunk_text", "")


def _extract_citations(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """증강된 문서 리스트에서 citation 정보를 추출한다."""
    citations: List[Dict[str, str]] = []
    for doc in docs:
        meta = doc.get("metadata") or {}
        citations.append(
            {
                "report_id": doc.get("report_id", meta.get("report_id", "")),
                "date": doc.get("date", meta.get("date", "")),
                "url": doc.get("url", meta.get("url", "")),
                "title": doc.get("title", meta.get("title", "")),
                "chunk_id": doc.get("chunk_id", meta.get("chunk_id", "")),
            }
        )
    return citations


def _fallback_context(documents: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
    """Augmenter 실패 시 사용할 기본 컨텍스트/인용 생성 로직"""
    chunks = [doc.get("chunk_text", "") for doc in documents]
    context = "\n\n---\n\n".join(chunks)
    citations = _extract_citations(documents)
    return {
        "context": context[: max_tokens * 4],
        "citations": citations,
    }


def run(state: QAState) -> QAState:
    """RAG DocumentAugmenter + Formatter로 컨텍스트와 citation을 생성한다."""
    items = state.get("reranked") or state.get("retrieved", [])
    if not items:
        state["context"] = ""
        state["citations"] = []
        return state

    query = state.get("question", "")
    max_tokens = state.get("meta", {}).get("max_ctx_tokens", 2000)
    user_level = state.get("user_level", "intermediate")

    logger.info("Building context from %d documents", len(items))

    _ensure_content_field(items)

    augmenter = DocumentAugmenter(
        max_context_length=max_tokens,
        max_documents=len(items),
        include_metadata=True,
    )

    formatter = MarkdownFormatter() if user_level == "beginner" else PromptFormatter()

    try:
        augmented = augmenter.augment(
            query=query,
            search_results=items,
            formatter=formatter,
        )
        state["context"] = augmented.context_text
        state["citations"] = _extract_citations(augmented.documents)
        logger.info(
            "Context built (%d chars, %d citations)",
            len(augmented.context_text),
            len(state["citations"]),
        )
    except Exception as exc:
        logger.error("Error in context_trim: %s", exc)
        fallback = _fallback_context(items, max_tokens)
        state["context"] = fallback["context"]
        state["citations"] = fallback["citations"]

    return state
