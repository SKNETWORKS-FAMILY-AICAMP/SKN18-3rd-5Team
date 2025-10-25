#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 RAG 시스템
Retrieval + Augmentation + Generation을 통합한 완전한 RAG 시스템
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .retrieval.retriever import Retriever
from .retrieval.reranker import BaseReranker, KeywordReranker, SemanticReranker, CombinedReranker
from .augmentation.augmenter import DocumentAugmenter, AugmentedContext
from .augmentation.formatters import BaseFormatter, PromptFormatter, MarkdownFormatter
from .generation.generator import LLMGenerator, OllamaGenerator, GenerationConfig, GeneratedAnswer
from .models.config import EmbeddingModelType

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG 시스템 응답"""
    query: str
    retrieved_documents: List[Dict[str, Any]]
    augmented_context: AugmentedContext
    generated_answer: Optional[GeneratedAnswer] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RAGSystem:
    """통합 RAG 시스템"""
    
    def __init__(
        self,
        model_type: EmbeddingModelType = EmbeddingModelType.MULTILINGUAL_E5_SMALL,
        db_config: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        reranker: Optional[BaseReranker] = None,
        formatter: Optional[BaseFormatter] = None,
        max_context_length: int = 4000,
        max_documents: int = 5,
        llm_generator: Optional[LLMGenerator] = None,
        enable_generation: bool = False
    ):
        """
        Args:
            model_type: 사용할 임베딩 모델
            db_config: 데이터베이스 연결 설정
            device: 디바이스
            reranker: 리랭킹 모듈
            formatter: 문서 포맷터
            max_context_length: 최대 컨텍스트 길이
            max_documents: 최대 문서 수
            llm_generator: LLM 생성기
            enable_generation: 생성 기능 활성화 여부
        """
        # Retrieval 컴포넌트
        self.retriever = Retriever(
            model_type=model_type,
            db_config=db_config,
            device=device,
            reranker=reranker
        )

        # Augmentation 컴포넌트
        self.augmenter = DocumentAugmenter(
            max_context_length=max_context_length,
            max_documents=max_documents
        )

        # Generation 컴포넌트
        self.formatter = formatter or PromptFormatter()
        self.generator = llm_generator
        self.enable_generation = enable_generation and llm_generator is not None

        logger.info(f"RAG System initialized with {self.retriever.encoder.get_display_name()}")
        if self.enable_generation:
            logger.info(f"Generation enabled with {type(self.generator).__name__}")
    
    def retrieve_and_augment(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        use_reranker: bool = True,
        context_type: str = "general"
    ) -> RAGResponse:
        """
        검색 및 증강 수행 (R + A)
        
        Args:
            query: 검색 쿼리
            top_k: 검색할 문서 수
            min_similarity: 최소 유사도
            use_reranker: 리랭킹 사용 여부
            context_type: 컨텍스트 타입
            
        Returns:
            RAG 응답 (검색 + 증강 결과)
        """
        logger.info(f"Starting RAG retrieval and augmentation for query: {query[:50]}...")
        
        # 1. Retrieval: 문서 검색
        search_results = self.retriever.search(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            use_reranker=use_reranker
        )
        
        # 2. Augmentation: 컨텍스트 생성
        augmented_context = self.augmenter.augment(
            query=query,
            search_results=search_results,
            formatter=self.formatter
        )
        
        # 3. 응답 생성
        response = RAGResponse(
            query=query,
            retrieved_documents=search_results,
            augmented_context=augmented_context,
            metadata={
                'model_type': self.retriever.model_type,
                'reranker_used': use_reranker and self.retriever.reranker is not None,
                'formatter_used': self.formatter.name
            }
        )
        
        logger.info(f"RAG retrieval and augmentation completed: {len(search_results)} docs, {augmented_context.token_count} tokens")
        return response
    
    def search_only(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        use_reranker: bool = True
    ) -> List[Dict[str, Any]]:
        """
        검색만 수행 (R만)
        
        Args:
            query: 검색 쿼리
            top_k: 검색할 문서 수
            min_similarity: 최소 유사도
            use_reranker: 리랭킹 사용 여부
            
        Returns:
            검색 결과 리스트
        """
        return self.retriever.search(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            use_reranker=use_reranker
        )
    
    def augment_only(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        context_type: str = "general"
    ) -> AugmentedContext:
        """
        증강만 수행 (A만)
        
        Args:
            query: 원본 쿼리
            search_results: 검색 결과
            context_type: 컨텍스트 타입
            
        Returns:
            증강된 컨텍스트
        """
        return self.augmenter.augment(
            query=query,
            search_results=search_results,
            formatter=self.formatter
        )
    
    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        use_reranker: bool = True,
        context_type: str = "general",
        generation_config: Optional[GenerationConfig] = None
    ) -> RAGResponse:
        """
        전체 RAG 파이프라인 수행 (R + A + G)

        Args:
            query: 검색 쿼리
            top_k: 검색할 문서 수
            min_similarity: 최소 유사도
            use_reranker: 리랭킹 사용 여부
            context_type: 컨텍스트 타입
            generation_config: 생성 설정

        Returns:
            완전한 RAG 응답 (검색 + 증강 + 생성)
        """
        if not self.enable_generation:
            raise ValueError("Generation is not enabled. Initialize RAGSystem with llm_generator and enable_generation=True")

        logger.info(f"Starting full RAG pipeline for query: {query[:50]}...")

        # 1. Retrieval + Augmentation
        response = self.retrieve_and_augment(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            use_reranker=use_reranker,
            context_type=context_type
        )

        # 2. Generation
        generated_answer = self.generator.generate(
            query=query,
            context=response.augmented_context.context_text,
            config=generation_config
        )

        response.generated_answer = generated_answer
        response.metadata['generation_enabled'] = True

        logger.info(f"Full RAG pipeline completed: answer length = {len(generated_answer.answer)} chars")
        return response

    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        use_reranker: bool = True,
        context_type: str = "general"
    ) -> str:
        """
        LLM에 전달할 컨텍스트 생성

        Args:
            query: 검색 쿼리
            top_k: 검색할 문서 수
            min_similarity: 최소 유사도
            use_reranker: 리랭킹 사용 여부
            context_type: 컨텍스트 타입

        Returns:
            LLM용 컨텍스트 문자열
        """
        response = self.retrieve_and_augment(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            use_reranker=use_reranker,
            context_type=context_type
        )

        return response.augmented_context.context_text
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        top_k: int = 5,
        min_similarity: float = 0.0,
        use_reranker: bool = True
    ) -> Dict[str, Any]:
        """
        검색 성능 평가
        
        Args:
            queries: 평가할 쿼리 리스트
            top_k: 검색할 문서 수
            min_similarity: 최소 유사도
            use_reranker: 리랭킹 사용 여부
            
        Returns:
            평가 결과
        """
        import time
        
        results = {
            'total_queries': len(queries),
            'successful_queries': 0,
            'failed_queries': 0,
            'total_time_ms': 0,
            'avg_time_ms': 0,
            'avg_documents_per_query': 0,
            'avg_similarity': 0,
            'queries': []
        }
        
        total_documents = 0
        total_similarity = 0
        
        for i, query in enumerate(queries):
            try:
                start_time = time.time()
                
                search_results = self.search_only(
                    query=query,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    use_reranker=use_reranker
                )
                
                query_time = (time.time() - start_time) * 1000
                
                query_similarity = sum(doc.get('similarity', 0) for doc in search_results) / len(search_results) if search_results else 0
                
                results['queries'].append({
                    'query': query,
                    'documents_found': len(search_results),
                    'avg_similarity': query_similarity,
                    'time_ms': query_time,
                    'success': True
                })
                
                results['successful_queries'] += 1
                results['total_time_ms'] += query_time
                total_documents += len(search_results)
                total_similarity += query_similarity
                
            except Exception as e:
                logger.error(f"Query {i+1} failed: {e}")
                results['queries'].append({
                    'query': query,
                    'error': str(e),
                    'success': False
                })
                results['failed_queries'] += 1
        
        # 통계 계산
        if results['successful_queries'] > 0:
            results['avg_time_ms'] = results['total_time_ms'] / results['successful_queries']
            results['avg_documents_per_query'] = total_documents / results['successful_queries']
            results['avg_similarity'] = total_similarity / results['successful_queries']
        
        return results
