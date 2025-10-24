#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
문서 증강 모듈
검색된 문서들을 LLM에 전달할 수 있는 형태로 가공
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AugmentedContext:
    """증강된 컨텍스트 정보"""
    query: str
    documents: List[Dict[str, Any]]
    context_text: str
    metadata: Dict[str, Any]
    token_count: int
    processing_time_ms: float


class DocumentAugmenter:
    """문서 증강 클래스"""
    
    def __init__(
        self,
        max_context_length: int = 4000,
        max_documents: int = 5,
        include_metadata: bool = True
    ):
        """
        Args:
            max_context_length: 최대 컨텍스트 길이 (토큰 수)
            max_documents: 최대 문서 수
            include_metadata: 메타데이터 포함 여부
        """
        self.max_context_length = max_context_length
        self.max_documents = max_documents
        self.include_metadata = include_metadata
        
        logger.info(f"DocumentAugmenter initialized: max_context={max_context_length}, max_docs={max_documents}")
    
    def augment(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        formatter: Optional[Any] = None
    ) -> AugmentedContext:
        """
        검색 결과를 증강하여 컨텍스트 생성
        
        Args:
            query: 원본 쿼리
            search_results: 검색 결과 리스트
            formatter: 문서 포맷터 (선택사항)
            
        Returns:
            증강된 컨텍스트
        """
        import time
        start_time = time.time()
        
        # 문서 필터링 및 정렬
        filtered_docs = self._filter_and_sort_documents(search_results)
        
        # 컨텍스트 길이 제한
        selected_docs = self._select_documents_by_length(filtered_docs)
        
        # 문서 포맷팅
        if formatter:
            context_text = formatter.format_documents(query, selected_docs)
        else:
            context_text = self._default_format_documents(query, selected_docs)
        
        # 토큰 수 추정
        token_count = self._estimate_token_count(context_text)
        
        # 메타데이터 생성
        metadata = {
            'total_documents': len(search_results),
            'selected_documents': len(selected_docs),
            'context_length': len(context_text),
            'token_count': token_count,
            'avg_similarity': sum(doc.get('similarity', 0) for doc in selected_docs) / len(selected_docs) if selected_docs else 0
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return AugmentedContext(
            query=query,
            documents=selected_docs,
            context_text=context_text,
            metadata=metadata,
            token_count=token_count,
            processing_time_ms=processing_time
        )
    
    def _filter_and_sort_documents(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서 필터링 및 정렬"""
        # 유사도 기준으로 정렬
        sorted_docs = sorted(
            search_results, 
            key=lambda x: x.get('similarity', 0), 
            reverse=True
        )
        
        # 최대 문서 수 제한
        return sorted_docs[:self.max_documents]
    
    def _select_documents_by_length(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """컨텍스트 길이를 고려하여 문서 선택"""
        selected = []
        current_length = 0
        
        for doc in documents:
            content = doc.get('content', '')
            doc_length = len(content.split())  # 단어 수로 추정
            
            if current_length + doc_length <= self.max_context_length:
                selected.append(doc)
                current_length += doc_length
            else:
                # 부분적으로 포함할 수 있는지 확인
                remaining_length = self.max_context_length - current_length
                if remaining_length > 100:  # 최소 100단어는 남겨야 함
                    # 문서를 잘라서 포함
                    truncated_content = ' '.join(content.split()[:remaining_length])
                    truncated_doc = doc.copy()
                    truncated_doc['content'] = truncated_content + '...'
                    selected.append(truncated_doc)
                break
        
        return selected
    
    def _default_format_documents(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """기본 문서 포맷팅"""
        if not documents:
            return "관련 문서를 찾을 수 없습니다."
        
        context_parts = [f"질문: {query}\n"]
        context_parts.append("관련 문서들:\n")
        
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            similarity = doc.get('similarity', 0)
            
            context_parts.append(f"[문서 {i}] (유사도: {similarity:.3f})")
            context_parts.append(f"{content}\n")
            
            if self.include_metadata and doc.get('metadata'):
                metadata = doc['metadata']
                if metadata.get('source'):
                    context_parts.append(f"출처: {metadata['source']}\n")
        
        return "\n".join(context_parts)
    
    def _estimate_token_count(self, text: str) -> int:
        """토큰 수 추정 (한국어 기준)"""
        # 한국어는 대략 1토큰 = 1.5글자로 추정
        return int(len(text) / 1.5)


class ContextBuilder:
    """컨텍스트 빌더 클래스"""
    
    def __init__(self, augmenter: DocumentAugmenter):
        self.augmenter = augmenter
    
    def build_context(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        context_type: str = "general"
    ) -> str:
        """
        특정 용도에 맞는 컨텍스트 빌드
        
        Args:
            query: 원본 쿼리
            search_results: 검색 결과
            context_type: 컨텍스트 타입 ("general", "qa", "summarization")
            
        Returns:
            빌드된 컨텍스트
        """
        if context_type == "qa":
            return self._build_qa_context(query, search_results)
        elif context_type == "summarization":
            return self._build_summarization_context(query, search_results)
        else:
            return self._build_general_context(query, search_results)
    
    def _build_qa_context(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Q&A용 컨텍스트 빌드"""
        context = f"다음 질문에 답하기 위해 관련 문서들을 검색했습니다:\n\n"
        context += f"질문: {query}\n\n"
        context += "참고 문서들:\n"
        
        for i, doc in enumerate(search_results[:3], 1):  # 상위 3개만
            content = doc.get('content', '')
            similarity = doc.get('similarity', 0)
            context += f"{i}. (유사도: {similarity:.3f}) {content}\n\n"
        
        context += "위 문서들을 참고하여 질문에 정확하고 상세하게 답변해주세요."
        return context
    
    def _build_summarization_context(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """요약용 컨텍스트 빌드"""
        context = f"다음 주제에 대한 문서들을 요약해주세요:\n\n"
        context += f"주제: {query}\n\n"
        context += "문서들:\n"
        
        for i, doc in enumerate(search_results, 1):
            content = doc.get('content', '')
            context += f"[문서 {i}]\n{content}\n\n"
        
        context += "위 문서들을 종합하여 핵심 내용을 요약해주세요."
        return context
    
    def _build_general_context(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """일반용 컨텍스트 빌드"""
        return self.augmenter._default_format_documents(query, search_results)
