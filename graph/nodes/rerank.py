from __future__ import annotations

from typing import Any, Dict, List

from graph.state import QAState
from service.rag.retrieval.reranker import CombinedReranker, create_default_reranker

# 전역으로 RAG 기본 리랭커 구성 (Keyword/Length/Position 조합)
_DEFAULT_RERANKER: CombinedReranker = create_default_reranker()


def _prepare_candidates(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """LangGraph retrieved 문서를 RAG 리랭커가 기대하는 구조로 정규화한다."""
    # TODO: retriever 결과 구조가 확정되면 chunk_text/content 필드 매핑 정책을 모듈화
    prepared: List[Dict[str, Any]] = []
    for doc in documents:
        content = doc.get("chunk_text") or doc.get("content") or ""
        candidate = dict(doc)  # 얕은 복사
        candidate.setdefault("content", content)
        candidate.setdefault("chunk_text", content)
        candidate.setdefault("metadata", {})
        prepared.append(candidate)
    return prepared


def _format_reranked(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """리랭킹 결과를 LangGraph 파이프라인에서 사용하는 필드명으로 정리한다."""
    formatted: List[Dict[str, Any]] = []
    for item in results:
        meta = item.get("metadata") or {}
        # TODO: report_id/date/url/title 등은 metadata 스키마 통합 후 공통 유틸로 추출
        formatted.append(
            {
                "chunk_text": item.get("chunk_text") or item.get("content", ""),
                "report_id": item.get("report_id", meta.get("report_id", "")),
                "date": item.get("date", meta.get("date", "")),
                "url": item.get("url", meta.get("url", "")),
                "title": item.get("title", meta.get("title", "")),
                "chunk_id": item.get("chunk_id", meta.get("chunk_id", "")),
                "similarity": item.get("similarity", 0.0),
                "rerank_score": item.get("final_rerank_score", item.get("rerank_score", 0.0)),
                "metadata": meta,
            }
        )
    return formatted


def run(state: QAState) -> QAState:
    """검색 후보를 RAG 리랭커 조합으로 재정렬하고 state['reranked']에 저장한다."""
    documents = state.get("retrieved", [])
    if not documents:
        state["reranked"] = []
        return state

    query = state.get("rewritten_query") or state.get("question", "")
    top_k = state.get("meta", {}).get("rerank_n") or len(documents)

    try:
        print(f"[Rerank] start (docs={len(documents)}, top_k={top_k})")
        candidates = _prepare_candidates(documents)
        reranked = _DEFAULT_RERANKER.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )
        state["reranked"] = _format_reranked(reranked)
        print(f"[Rerank] complete (kept={len(state['reranked'])})")
    except Exception as exc:
        print(f"[Rerank] error={exc}")
        state["reranked"] = documents  # 실패 시 원본 유지

    return state
