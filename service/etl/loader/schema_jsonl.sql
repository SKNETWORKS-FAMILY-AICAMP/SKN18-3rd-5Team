-- ============================================================================
-- JSONL RAG Vector Store Schema (pgvector) - JSONL 로더용
-- JSONL 파일 로딩에 최적화된 스키마
-- ============================================================================

-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- 1. 청크 테이블 (JSONL 구조에 맞게 단순화)
-- ============================================================================
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(500) UNIQUE NOT NULL,    -- 청크 고유 ID
    doc_id VARCHAR(255),                      -- 문서 ID
    chunk_type VARCHAR(50),                   -- 청크 타입 (text, table_row)
    section_path TEXT,                        -- 섹션 경로
    natural_text TEXT NOT NULL,               -- 자연어 텍스트 (검색 대상)
    structured_data JSONB,                    -- 구조화된 데이터
    metadata JSONB,                           -- 메타데이터 (JSONL의 metadata 필드)
    token_count INTEGER,                      -- 토큰 수
    merged_count INTEGER,                     -- 병합된 청크 수
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 청크 검색 인덱스
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_corp_name ON chunks ((metadata->>'corp_name'));
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks ((metadata->>'doc_type'));
CREATE INDEX IF NOT EXISTS idx_chunks_fiscal_year ON chunks ((metadata->>'fiscal_year'));

-- 전문 검색 (PostgreSQL FTS)
CREATE INDEX IF NOT EXISTS idx_chunks_natural_text_fts ON chunks
USING GIN (to_tsvector('simple', natural_text));

-- ============================================================================
-- 2. 임베딩 테이블 (모델별로 분리)
-- ============================================================================

-- 2.1 Multilingual E5-Small 임베딩 테이블 (384차원)
CREATE TABLE IF NOT EXISTS embeddings_multilingual_e5_small (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(500) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_multilingual_e5_small_chunk ON embeddings_multilingual_e5_small(chunk_id);

-- 2.2 KakaoBank DeBERTa 임베딩 테이블 (768차원)
CREATE TABLE IF NOT EXISTS embeddings_kakaobank (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(500) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_kakaobank_chunk ON embeddings_kakaobank(chunk_id);

-- 2.3 FinE5 임베딩 테이블 (4096차원) - 금융 특화 모델
CREATE TABLE IF NOT EXISTS embeddings_fine5 (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(500) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding vector(4096) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_fine5_chunk ON embeddings_fine5(chunk_id);

-- ============================================================================
-- 3. 벡터 유사도 검색 인덱스 (HNSW)
-- ============================================================================

-- 3.1 Multilingual E5-Small HNSW 인덱스
CREATE INDEX IF NOT EXISTS idx_embeddings_multilingual_e5_small_hnsw
ON embeddings_multilingual_e5_small
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 3.2 KakaoBank HNSW 인덱스
CREATE INDEX IF NOT EXISTS idx_embeddings_kakaobank_hnsw
ON embeddings_kakaobank
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 3.3 FinE5 HNSW 인덱스
CREATE INDEX IF NOT EXISTS idx_embeddings_fine5_hnsw
ON embeddings_fine5
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- 4. 유틸리티 함수들
-- ============================================================================

-- 4.1 코사인 유사도 함수
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
RETURNS NUMERIC AS $$
    SELECT 1 - (a <=> b);  -- pgvector의 코사인 거리를 유사도로 변환
$$ LANGUAGE SQL IMMUTABLE STRICT;

-- 4.2 벡터 검색 함수 (모델별 테이블에서 검색)
CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding vector,
    model_name_param VARCHAR,
    top_k_param INTEGER DEFAULT 5,
    min_similarity NUMERIC DEFAULT 0.0
)
RETURNS TABLE (
    chunk_id VARCHAR(500),
    natural_text TEXT,
    similarity NUMERIC,
    metadata JSONB
) AS $$
BEGIN
    -- 모델별로 적절한 테이블에서 검색
    CASE model_name_param
        WHEN 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' THEN
            RETURN QUERY
            SELECT c.chunk_id, c.natural_text,
                   cosine_similarity(e.embedding, query_embedding) as sim,
                   c.metadata
            FROM embeddings_multilingual_minilm e
            JOIN chunks c ON e.chunk_id = c.chunk_id
            WHERE cosine_similarity(e.embedding, query_embedding) >= min_similarity
            ORDER BY e.embedding <=> query_embedding
            LIMIT top_k_param;

        WHEN 'intfloat/multilingual-e5-small' THEN
            RETURN QUERY
            SELECT c.chunk_id, c.natural_text,
                   cosine_similarity(e.embedding, query_embedding) as sim,
                   c.metadata
            FROM embeddings_e5_small e
            JOIN chunks c ON e.chunk_id = c.chunk_id
            WHERE cosine_similarity(e.embedding, query_embedding) >= min_similarity
            ORDER BY e.embedding <=> query_embedding
            LIMIT top_k_param;

        WHEN 'kakaobank/kf-deberta-base' THEN
            RETURN QUERY
            SELECT c.chunk_id, c.natural_text,
                   cosine_similarity(e.embedding, query_embedding) as sim,
                   c.metadata
            FROM embeddings_kakaobank e
            JOIN chunks c ON e.chunk_id = c.chunk_id
            WHERE cosine_similarity(e.embedding, query_embedding) >= min_similarity
            ORDER BY e.embedding <=> query_embedding
            LIMIT top_k_param;

        ELSE
            RAISE EXCEPTION 'Unknown model: %', model_name_param;
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 5. 검색 로그 테이블 (성능 모니터링용)
-- ============================================================================
CREATE TABLE IF NOT EXISTS search_logs (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,         -- 모델명
    query TEXT NOT NULL,                      -- 검색 쿼리
    query_embedding vector,                   -- 쿼리 임베딩
    top_k INTEGER DEFAULT 5,                  -- 검색 개수
    search_time_ms NUMERIC(10, 2),            -- 검색 시간 (밀리초)
    results JSONB,                            -- 검색 결과
    avg_similarity NUMERIC(5, 4),             -- 평균 유사도
    created_at TIMESTAMP DEFAULT NOW()
);

-- 검색 로그 인덱스
CREATE INDEX IF NOT EXISTS idx_search_logs_model ON search_logs(model_name);
CREATE INDEX IF NOT EXISTS idx_search_logs_created ON search_logs(created_at);

-- ============================================================================
-- 6. 유용한 쿼리 예시 (주석)
-- ============================================================================

/*
-- 6.1 모델별 임베딩 개수 확인
SELECT 
    'multilingual_minilm' as model,
    COUNT(*) as embedding_count
FROM embeddings_multilingual_minilm
UNION ALL
SELECT 
    'e5_small' as model,
    COUNT(*) as embedding_count
FROM embeddings_e5_small
UNION ALL
SELECT 
    'kakaobank' as model,
    COUNT(*) as embedding_count
FROM embeddings_kakaobank;

-- 6.2 특정 모델로 유사도 검색
SELECT * FROM search_similar_chunks(
    '[0.1, 0.2, ..., 0.3]'::vector,  -- 쿼리 임베딩
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    5,     -- top_k
    0.5    -- min_similarity
);

-- 6.3 특정 기업의 모든 청크 조회
SELECT
    chunk_id,
    natural_text,
    chunk_type,
    metadata->>'corp_name' as corp_name,
    metadata->>'doc_type' as doc_type
FROM chunks
WHERE metadata->>'corp_name' = '삼성전자'
ORDER BY doc_id, chunk_id;

-- 6.4 저장 공간 확인
SELECT
    pg_size_pretty(pg_total_relation_size('chunks')) as chunks_size,
    pg_size_pretty(pg_total_relation_size('embeddings_multilingual_minilm')) as minilm_size,
    pg_size_pretty(pg_total_relation_size('embeddings_e5_small')) as e5_size,
    pg_size_pretty(pg_total_relation_size('embeddings_kakaobank')) as kakao_size;

-- 6.5 기업별 청크 수 통계
SELECT 
    metadata->>'corp_name' as corp_name,
    COUNT(*) as chunk_count
FROM chunks
WHERE metadata->>'corp_name' IS NOT NULL
GROUP BY metadata->>'corp_name'
ORDER BY chunk_count DESC
LIMIT 10;
*/
