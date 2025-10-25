#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
임베딩 로드 및 검색 테스트 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from service.rag_jsonl.models.config import EmbeddingModelType
from service.rag_jsonl.models.encoder import EmbeddingEncoder
from service.rag_jsonl.vectorstore.pgvector_store import PgVectorStore
from config.vector_database import get_vector_db_config

def test_embedding_load():
    """임베딩 로드 및 검색 테스트"""
    
    print("🧪 임베딩 로드 및 검색 테스트 시작")
    print("=" * 60)
    
    # 데이터베이스 설정
    db_config = get_vector_db_config().get_db_config()
    
    # 벡터 스토어 초기화
    vector_store = PgVectorStore(db_config)
    
    # 통계 조회
    print("📊 현재 임베딩 통계:")
    stats = vector_store.get_stats()
    for model_name, model_stats in stats['models'].items():
        if 'error' not in model_stats:
            print(f"  {model_name}: {model_stats['embedding_count']}개 임베딩")
        else:
            print(f"  {model_name}: 에러 - {model_stats['error']}")
    
    print("\n" + "=" * 60)
    
    # E5 모델로 테스트
    print("🔍 E5 모델 검색 테스트")
    try:
        # E5 인코더 초기화
        e5_encoder = EmbeddingEncoder(EmbeddingModelType.MULTILINGUAL_E5_SMALL)
        
        # 테스트 쿼리
        test_query = "삼성전자 주식"
        print(f"테스트 쿼리: '{test_query}'")
        
        # 쿼리 임베딩 생성
        query_embedding = e5_encoder.encode_query(test_query)
        print(f"쿼리 임베딩 차원: {len(query_embedding)}")
        
        # 벡터 검색
        search_results = vector_store.search_similar(
            query_embedding=query_embedding,
            model_type=EmbeddingModelType.MULTILINGUAL_E5_SMALL,
            top_k=3,
            min_similarity=0.0
        )
        
        print(f"검색 결과: {len(search_results)}개")
        for i, result in enumerate(search_results, 1):
            print(f"  {i}. 유사도: {result.similarity:.4f}")
            print(f"     청크 ID: {result.chunk_id}")
            print(f"     내용: {result.content[:100]}...")
            if result.corp_name:
                print(f"     기업: {result.corp_name}")
            print()
        
    except Exception as e:
        print(f"❌ E5 모델 테스트 실패: {e}")
    
    print("=" * 60)
    
    # KakaoBank 모델로 테스트
    print("🔍 KakaoBank 모델 검색 테스트")
    try:
        # KakaoBank 인코더 초기화
        kakaobank_encoder = EmbeddingEncoder(EmbeddingModelType.KAKAOBANK_DEBERTA)
        
        # 테스트 쿼리
        test_query = "재무제표"
        print(f"테스트 쿼리: '{test_query}'")
        
        # 쿼리 임베딩 생성
        query_embedding = kakaobank_encoder.encode_query(test_query)
        print(f"쿼리 임베딩 차원: {len(query_embedding)}")
        
        # 벡터 검색
        search_results = vector_store.search_similar(
            query_embedding=query_embedding,
            model_type=EmbeddingModelType.KAKAOBANK_DEBERTA,
            top_k=3,
            min_similarity=0.0
        )
        
        print(f"검색 결과: {len(search_results)}개")
        for i, result in enumerate(search_results, 1):
            print(f"  {i}. 유사도: {result.similarity:.4f}")
            print(f"     청크 ID: {result.chunk_id}")
            print(f"     내용: {result.content[:100]}...")
            if result.corp_name:
                print(f"     기업: {result.corp_name}")
            print()
        
    except Exception as e:
        print(f"❌ KakaoBank 모델 테스트 실패: {e}")
    
    # 연결 종료
    vector_store.close()
    print("✅ 테스트 완료")

if __name__ == "__main__":
    test_embedding_load()
