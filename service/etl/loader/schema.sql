-- ============================================================================
-- RAG Vector Store Schema (pgvector) - Updated for Financial Documents
-- 금융 문서 청크 데이터에 맞게 수정된 스키마
-- ============================================================================

-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- vector_db 스키마 생성
CREATE SCHEMA IF NOT EXISTS vector_db;

-- ============================================================================
-- 1. 임베딩 모델 테이블 (vector_db 스키마)
-- ============================================================================
CREATE TABLE IF NOT EXISTS vector_db.embedding_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) UNIQUE NOT NULL,  -- 예: "intfloat/multilingual-e5-small"
    display_name VARCHAR(255) NOT NULL,       -- 예: "E5-Small (Multilingual)"
    dimension INTEGER NOT NULL,               -- 임베딩 차원
    max_seq_length INTEGER NOT NULL,          -- 최대 시퀀스 길이
    pooling_mode VARCHAR(50),                 -- 'mean', 'cls', 'max'
    normalize_embeddings BOOLEAN DEFAULT TRUE,
    notes TEXT,                               -- 모델 설명
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 모델 이름 인덱스
CREATE INDEX IF NOT EXISTS idx_embedding_models_name ON vector_db.embedding_models(model_name);

-- ============================================================================
-- 2. 문서 소스 테이블 (원본 데이터) - vector_db 스키마
-- UPDATED: 금융 문서 구조에 맞게 수정
-- ============================================================================
CREATE TABLE IF NOT EXISTS vector_db.document_sources (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE NOT NULL,      -- 문서 ID (예: "20241028_00382199")
    corp_name VARCHAR(255),                   -- 기업명
    document_name TEXT,                       -- 문서명
    rcept_dt VARCHAR(8),                      -- 접수일자 (YYYYMMDD)
    doc_type VARCHAR(50),                     -- 문서 유형 (financial_statement, disclosure, other, etc)
    data_category VARCHAR(50),                -- 데이터 카테고리 (stock_info, financial, etc)
    fiscal_year INTEGER,                      -- 회계연도
    metadata JSONB,                           -- 추가 메타데이터
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 소스 인덱스
CREATE INDEX IF NOT EXISTS idx_document_sources_doc_id ON vector_db.document_sources(doc_id);
CREATE INDEX IF NOT EXISTS idx_document_sources_corp_name ON vector_db.document_sources(corp_name);
CREATE INDEX IF NOT EXISTS idx_document_sources_doc_type ON vector_db.document_sources(doc_type);
CREATE INDEX IF NOT EXISTS idx_document_sources_rcept_dt ON vector_db.document_sources(rcept_dt);

-- ============================================================================
-- 3. 문서 청크 테이블 (분할된 텍스트) - vector_db 스키마
-- UPDATED: JSONL 청크 구조에 맞게 수정
-- ============================================================================
CREATE TABLE IF NOT EXISTS vector_db.document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(500) UNIQUE NOT NULL,    -- 청크 고유 ID (예: "20241028_00382199_text_주요사항보고서자기주_000")
    doc_id VARCHAR(255) NOT NULL,             -- 문서 ID (document_sources.doc_id 참조)
    chunk_type VARCHAR(50),                   -- 'text', 'table', 'header' 등
    section_path TEXT,                        -- 섹션 경로
    structured_data JSONB,                    -- 구조화된 데이터
    natural_text TEXT NOT NULL,               -- 자연어 텍스트 (검색 대상 메인 필드)

    -- 메타데이터 (플랫 구조로 저장 - 빠른 필터링을 위해)
    corp_name VARCHAR(255),                   -- 기업명
    document_name TEXT,                       -- 문서명
    rcept_dt VARCHAR(8),                      -- 접수일자
    next_context TEXT,                        -- 다음 컨텍스트 (문맥 힌트)
    doc_type VARCHAR(50),                     -- 문서 유형
    data_category VARCHAR(50),                -- 데이터 카테고리
    fiscal_year INTEGER,                      -- 회계연도
    keywords TEXT[],                          -- 키워드 배열
    token_count INTEGER,                      -- 토큰 수

    metadata JSONB,                           -- 추가 메타데이터 (원본 전체 보존)
    created_at TIMESTAMP DEFAULT NOW()
);

-- 청크 검색 인덱스
CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id ON vector_db.document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_chunk_type ON vector_db.document_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_document_chunks_corp_name ON vector_db.document_chunks(corp_name);
CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_type ON vector_db.document_chunks(doc_type);
CREATE INDEX IF NOT EXISTS idx_document_chunks_keywords ON vector_db.document_chunks USING GIN(keywords);

-- 전문 검색 (PostgreSQL FTS) - 한국어 설정이 없으면 기본 설정 사용
CREATE INDEX IF NOT EXISTS idx_document_chunks_natural_text_fts ON vector_db.document_chunks
USING GIN (to_tsvector('simple', natural_text));

-- ============================================================================
-- 4. 벡터 임베딩 테이블 (모델별로 분리) - vector_db 스키마
-- ============================================================================

-- 4.1 E5-Small 임베딩 테이블 (384차원)
CREATE TABLE IF NOT EXISTS vector_db.embeddings_e5_small (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES vector_db.document_chunks(id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_e5_small_chunk ON vector_db.embeddings_e5_small(chunk_id);

-- 4.2 KakaoBank DeBERTa 임베딩 테이블 (768차원)
CREATE TABLE IF NOT EXISTS vector_db.embeddings_kakaobank (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES vector_db.document_chunks(id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_kakaobank_chunk ON vector_db.embeddings_kakaobank(chunk_id);

-- 4.3 FinE5 임베딩 테이블 (4096차원)
-- FinanceMTEB/FinE5 - 금융 도메인 특화 E5-Mistral-7B 기반
CREATE TABLE IF NOT EXISTS vector_db.embeddings_fine5 (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES vector_db.document_chunks(id) ON DELETE CASCADE,
    embedding vector(4096) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_fine5_chunk ON vector_db.embeddings_fine5(chunk_id);


-- ============================================================================
-- 5. 벡터 유사도 검색 인덱스 (HNSW - 모델별로 생성)
-- ============================================================================
-- HNSW 인덱스: IVFFlat보다 빠르고 정확함
-- lists 파라미터 대신 m, ef_construction 파라미터 사용

-- 5.1 E5-Small HNSW 인덱스
CREATE INDEX IF NOT EXISTS idx_embeddings_e5_small_hnsw
ON vector_db.embeddings_e5_small
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 5.2 KakaoBank HNSW 인덱스
CREATE INDEX IF NOT EXISTS idx_embeddings_kakaobank_hnsw
ON vector_db.embeddings_kakaobank
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 5.3 FinE5 HNSW 인덱스
CREATE INDEX IF NOT EXISTS idx_embeddings_fine5_hnsw
ON vector_db.embeddings_fine5
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 인덱스 없이 사용

-- ============================================================================
-- 6. 검색 로그 테이블 (모델 성능 비교용) - vector_db 스키마
-- ============================================================================
CREATE TABLE IF NOT EXISTS vector_db.search_logs (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES vector_db.embedding_models(id),
    query TEXT NOT NULL,                      -- 검색 쿼리
    query_embedding vector,                   -- 쿼리 임베딩
    top_k INTEGER DEFAULT 5,                  -- 검색 개수
    search_time_ms NUMERIC(10, 2),            -- 검색 시간 (밀리초)
    results JSONB,                            -- 검색 결과 (chunk_ids, similarities)
    avg_similarity NUMERIC(5, 4),             -- 평균 유사도
    created_at TIMESTAMP DEFAULT NOW()
);

-- 검색 로그 인덱스
CREATE INDEX IF NOT EXISTS idx_search_logs_model ON vector_db.search_logs(model_id);
CREATE INDEX IF NOT EXISTS idx_search_logs_created ON vector_db.search_logs(created_at);

-- ============================================================================
-- 7. 모델 성능 메트릭 테이블 - vector_db 스키마
-- ============================================================================
CREATE TABLE IF NOT EXISTS vector_db.model_metrics (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES vector_db.embedding_models(id) ON DELETE CASCADE,
    metric_date DATE DEFAULT CURRENT_DATE,

    -- 성능 지표
    total_searches INTEGER DEFAULT 0,
    avg_search_time_ms NUMERIC(10, 2),
    avg_top1_similarity NUMERIC(5, 4),
    avg_top3_similarity NUMERIC(5, 4),
    avg_top5_similarity NUMERIC(5, 4),

    -- 메타정보
    total_embeddings INTEGER,                 -- 저장된 임베딩 수
    storage_size_mb NUMERIC(10, 2),           -- 저장 공간

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- 날짜별로 유일
    UNIQUE(model_id, metric_date)
);

-- ============================================================================
-- 8. 유틸리티 함수들
-- ============================================================================

-- 8.1 코사인 유사도 함수
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
RETURNS NUMERIC AS $$
    SELECT 1 - (a <=> b);  -- pgvector의 코사인 거리를 유사도로 변환
$$ LANGUAGE SQL IMMUTABLE STRICT;

-- 8.2 벡터 검색 함수 (모델별 테이블에서 검색)
CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding vector,
    model_name_param VARCHAR,
    top_k_param INTEGER DEFAULT 5,
    min_similarity NUMERIC DEFAULT 0.0
)
RETURNS TABLE (
    chunk_id INTEGER,
    content TEXT,
    similarity NUMERIC,
    metadata JSONB
) AS $$
BEGIN
    -- 모델별로 적절한 테이블에서 검색
    CASE model_name_param
        WHEN 'intfloat/multilingual-e5-small' THEN
            RETURN QUERY
            SELECT dc.id, dc.natural_text,
                   cosine_similarity(e.embedding, query_embedding) as sim,
                   dc.metadata
            FROM vector_db.embeddings_e5_small e
            JOIN vector_db.document_chunks dc ON e.chunk_id = dc.id
            WHERE cosine_similarity(e.embedding, query_embedding) >= min_similarity
            ORDER BY e.embedding <=> query_embedding
            LIMIT top_k_param;

        WHEN 'kakaobank/kf-deberta-base' THEN
            RETURN QUERY
            SELECT dc.id, dc.natural_text,
                   cosine_similarity(e.embedding, query_embedding) as sim,
                   dc.metadata
            FROM vector_db.embeddings_kakaobank e
            JOIN vector_db.document_chunks dc ON e.chunk_id = dc.id
            WHERE cosine_similarity(e.embedding, query_embedding) >= min_similarity
            ORDER BY e.embedding <=> query_embedding
            LIMIT top_k_param;

        WHEN 'FinanceMTEB/FinE5' THEN
            RETURN QUERY
            SELECT dc.id, dc.natural_text,
                   cosine_similarity(e.embedding, query_embedding) as sim,
                   dc.metadata
            FROM vector_db.embeddings_fine5 e
            JOIN vector_db.document_chunks dc ON e.chunk_id = dc.id
            WHERE cosine_similarity(e.embedding, query_embedding) >= min_similarity
            ORDER BY e.embedding <=> query_embedding
            LIMIT top_k_param;

        ELSE
            RAISE EXCEPTION 'Unknown model: %', model_name_param;
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- 8.4 타임스탬프 자동 업데이트 트리거 함수
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 트리거 생성 (조건부)
DO $$
BEGIN
    -- embedding_models 트리거
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_embedding_models_updated_at') THEN
        CREATE TRIGGER update_embedding_models_updated_at
            BEFORE UPDATE ON vector_db.embedding_models
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;

    -- document_sources 트리거
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_document_sources_updated_at') THEN
        CREATE TRIGGER update_document_sources_updated_at
            BEFORE UPDATE ON vector_db.document_sources
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;

    -- model_metrics 트리거
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_model_metrics_updated_at') THEN
        CREATE TRIGGER update_model_metrics_updated_at
            BEFORE UPDATE ON vector_db.model_metrics
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- ============================================================================
-- 9. 초기 데이터 삽입 (3가지 모델 - 금융 문서 특화)
-- ============================================================================

INSERT INTO vector_db.embedding_models (model_name, display_name, dimension, max_seq_length, pooling_mode, notes)
VALUES
    ('intfloat/multilingual-e5-small', 'E5-Small (Multilingual)', 384, 512, 'mean', '경량 모델, 빠른 추론 속도, 다국어 지원'),
    ('kakaobank/kf-deberta-base', 'KakaoBank DeBERTa', 768, 512, 'mean', '한국어 금융 데이터 특화, 높은 한국어 이해도'),
    ('FinanceMTEB/FinE5', 'FinE5 (Finance E5-Mistral-7B)', 4096, 32768, 'mean', '금융 도메인 특화 E5-Mistral-7B, 긴 문맥 처리, 고품질 금융 임베딩')
ON CONFLICT (model_name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    dimension = EXCLUDED.dimension,
    max_seq_length = EXCLUDED.max_seq_length,
    pooling_mode = EXCLUDED.pooling_mode,
    notes = EXCLUDED.notes,
    updated_at = NOW();

-- ============================================================================
-- 10. 유용한 쿼리 예시 (주석)
-- ============================================================================

/*
-- 10.1 모델별 임베딩 개수 확인
SELECT
    em.display_name,
    (SELECT COUNT(*) FROM vector_db.embeddings_e5_small WHERE chunk_id IN (SELECT id FROM vector_db.document_chunks)) as e5_count,
    (SELECT COUNT(*) FROM vector_db.embeddings_kakaobank WHERE chunk_id IN (SELECT id FROM vector_db.document_chunks)) as kakao_count,
    (SELECT COUNT(*) FROM vector_db.embeddings_fine5 WHERE chunk_id IN (SELECT id FROM vector_db.document_chunks)) as fine5_count
FROM vector_db.embedding_models em
LIMIT 1;

-- 10.2 특정 모델로 유사도 검색
SELECT * FROM search_similar_chunks(
    '[0.1, 0.2, ..., 0.3]'::vector,  -- 쿼리 임베딩
    'intfloat/multilingual-e5-small',
    5,     -- top_k
    0.5    -- min_similarity
);

-- 10.3 모델별 평균 검색 시간 비교
SELECT
    em.display_name,
    AVG(sl.search_time_ms) as avg_time_ms,
    AVG(sl.avg_similarity) as avg_similarity,
    COUNT(sl.id) as total_searches
FROM vector_db.search_logs sl
JOIN vector_db.embedding_models em ON sl.model_id = em.id
WHERE sl.created_at >= NOW() - INTERVAL '7 days'
GROUP BY em.id, em.display_name
ORDER BY avg_similarity DESC;

-- 10.4 특정 기업의 모든 청크 조회
SELECT
    dc.id as chunk_id,
    dc.chunk_id as chunk_str_id,
    dc.natural_text,
    dc.chunk_type,
    dc.corp_name,
    dc.doc_type
FROM vector_db.document_chunks dc
WHERE dc.corp_name = '삼성전자'
ORDER BY dc.doc_id, dc.chunk_id;

-- 10.5 저장 공간 확인
SELECT
    pg_size_pretty(pg_total_relation_size('vector_db.document_chunks')) as chunks_size,
    pg_size_pretty(pg_total_relation_size('vector_db.embeddings_e5_small')) as e5_size,
    pg_size_pretty(pg_total_relation_size('vector_db.embeddings_kakaobank')) as kakao_size,
    pg_size_pretty(pg_total_relation_size('vector_db.embeddings_fine5')) as fine5_size;
*/
