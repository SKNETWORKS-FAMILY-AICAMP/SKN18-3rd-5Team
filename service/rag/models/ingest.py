#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
데이터 수집 및 임베딩 생성 스크립트
JSON 문서를 로드하고 벡터 임베딩을 생성하여 pgvector에 저장
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .config import EmbeddingModelType
from .encoder import EmbeddingEncoder
from ..vectorstore.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """데이터 수집 및 임베딩 파이프라인"""

    def __init__(
        self,
        db_config: Optional[Dict[str, str]] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            db_config: 데이터베이스 연결 설정
            device: 디바이스 ('cuda', 'cpu', None)
        """
        self.db_config = db_config
        self.device = device
        self.vector_store = PgVectorStore(db_config)

    def load_json_documents(self, json_path: str) -> List[Dict[str, Any]]:
        """JSON 문서 로드"""
        logger.info(f"Loading documents from: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            logger.info(f"Loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise

    def process_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        skip_chunking: bool = False
    ) -> List[Dict[str, Any]]:
        """문서를 청크로 분할하고 전처리"""
        logger.info(f"Processing {len(documents)} documents into chunks")
        
        processed_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            if not content.strip():
                continue
            
            # 청킹 비활성화 옵션
            if skip_chunking:
                chunks = [content]  # 원본 그대로 사용
            else:
                # 간단한 청킹 (문장 단위)
                chunks = self._split_into_chunks(content, chunk_size, chunk_overlap)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                
                chunk_data = {
                    'id': f"{doc.get('id', f'doc_{doc_idx}')}_{chunk_idx}",
                    'source': doc.get('source', 'unknown'),
                    'content': chunk_text.strip(),
                    'chunk_index': chunk_idx,
                    'metadata': {
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'source_doc_id': doc.get('id', f'doc_{doc_idx}'),
                        'source_doc_title': doc.get('source', 'unknown'),
                        'chunk_length': len(chunk_text.strip())
                    }
                }
                
                processed_chunks.append(chunk_data)
        
        logger.info(f"Created {len(processed_chunks)} chunks from {len(documents)} documents")
        return processed_chunks

    def _split_into_chunks(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """텍스트를 청크로 분할"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 문장 경계에서 자르기 시도 (한국어 + 영어)
            last_period = text.rfind('.', start, end)
            last_korean_period = text.rfind('。', start, end)
            last_question = text.rfind('?', start, end)
            last_exclamation = text.rfind('!', start, end)
            
            # 가장 가까운 문장 끝 찾기
            sentence_end = max(last_period, last_korean_period, last_question, last_exclamation)
            
            if sentence_end > start + chunk_size // 2:  # 너무 앞에서 자르지 않도록
                end = sentence_end + 1
            
            chunks.append(text[start:end])
            start = end - chunk_overlap
        
        return chunks

    def create_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        model_type: EmbeddingModelType,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """청크들에 대한 임베딩 생성"""
        logger.info(f"Creating embeddings with {model_type.value}")
        
        encoder = EmbeddingEncoder(model_type, self.device)
        
        # 텍스트만 추출
        texts = [chunk['content'] for chunk in chunks]
        
        # 배치별로 임베딩 생성
        embeddings = encoder.encode_documents(
            texts=texts,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        # 청크에 임베딩 추가
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        logger.info(f"Created {len(embeddings)} embeddings")
        return chunks

    def store_embeddings(
        self,
        chunks_with_embeddings: List[Dict[str, Any]],
        model_type: EmbeddingModelType,
        source_type: str = "finance_support"
    ) -> int:
        """임베딩을 데이터베이스에 저장"""
        logger.info(f"Storing embeddings in database")
        
        try:
            # 데이터베이스 연결
            self.vector_store.connect()
            
            # 스키마 초기화 (필요시)
            self._initialize_schema()
            
            # 문서 저장
            inserted_count = self.vector_store.insert_documents_with_embeddings(
                documents=chunks_with_embeddings,
                model_type=model_type,
                source_type=source_type,
                batch_size=100
            )
            
            # 벡터 인덱스 생성
            logger.info("Creating vector index...")
            self.vector_store.create_vector_index(model_type, lists=100)
            
            logger.info(f"Successfully stored {inserted_count} documents with embeddings")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise
        finally:
            self.vector_store.disconnect()

    def _initialize_schema(self):
        """데이터베이스 스키마 초기화"""
        try:
            # 스키마가 이미 존재하는지 확인
            self.vector_store.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'vector_db'
                    AND table_name = 'embedding_models'
                )
            """)
            schema_exists = self.vector_store.cursor.fetchone()[0]

            if schema_exists:
                logger.info("Database schema already exists, skipping initialization")
                return

            # 스키마 파일 읽기
            schema_path = Path(__file__).parent.parent / "vectorstore" / "schema.sql"
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()

            # 스키마 실행 (별도 커밋)
            self.vector_store.cursor.execute(schema_sql)
            self.vector_store.conn.commit()
            logger.info("Database schema initialized successfully")

        except Exception as e:
            # 스키마 초기화 실패 시 롤백하고 경고만 출력
            self.vector_store.conn.rollback()
            logger.warning(f"Schema initialization failed (may already exist): {e}")
            # 스키마가 이미 존재할 수 있으므로 예외를 발생시키지 않음

    def run_full_pipeline(
        self,
        json_path: str,
        model_type: EmbeddingModelType,
        source_type: str = "finance_support",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 32,
        skip_chunking: bool = False
    ) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        logger.info("Starting full ingestion pipeline")
        
        pipeline_stats = {
            'input_file': json_path,
            'model_type': model_type.value,
            'source_type': source_type,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'batch_size': batch_size,
            'steps': {}
        }
        
        try:
            # 1. 문서 로드
            logger.info("Step 1: Loading documents")
            documents = self.load_json_documents(json_path)
            pipeline_stats['steps']['documents_loaded'] = len(documents)
            
            # 2. 문서 처리 (청킹)
            logger.info("Step 2: Processing documents into chunks")
            chunks = self.process_documents(documents, chunk_size, chunk_overlap, skip_chunking)
            pipeline_stats['steps']['chunks_created'] = len(chunks)
            
            # 3. 임베딩 생성
            logger.info("Step 3: Creating embeddings")
            chunks_with_embeddings = self.create_embeddings(
                chunks, model_type, batch_size, show_progress=True
            )
            pipeline_stats['steps']['embeddings_created'] = len(chunks_with_embeddings)
            
            # 4. 데이터베이스 저장
            logger.info("Step 4: Storing in database")
            inserted_count = self.store_embeddings(
                chunks_with_embeddings, model_type, source_type
            )
            pipeline_stats['steps']['documents_stored'] = inserted_count
            
            pipeline_stats['status'] = 'success'
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            pipeline_stats['status'] = 'failed'
            pipeline_stats['error'] = str(e)
            logger.error(f"Pipeline failed: {e}")
            raise
        
        return pipeline_stats


def main():
    """메인 실행 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 설정
    json_path = "backend/data/vector_db/structured/서울시_주거복지사업_pgvector_ready_clecd ..aned.json"
    model_type = EmbeddingModelType.MULTILINGUAL_E5_SMALL  # 시작은 E5-Small로
    
    # 데이터베이스 설정
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'rey',
        'user': 'postgres',
        'password': 'post1234'
    }
    
    print("🚀 데이터 수집 및 임베딩 생성 시작")
    print(f"입력 파일: {json_path}")
    print(f"모델: {model_type.value}")
    print()
    
    # 파이프라인 실행
    pipeline = DataIngestionPipeline(db_config=db_config)
    
    try:
        stats = pipeline.run_full_pipeline(
            json_path=json_path,
            model_type=model_type,
            source_type="finance_support",
            chunk_size=500,
            chunk_overlap=50,
            batch_size=32
        )
        
        print("\n" + "="*60)
        print("파이프라인 실행 결과")
        print("="*60)
        print(f"상태: {stats['status']}")
        print(f"로드된 문서: {stats['steps']['documents_loaded']}")
        print(f"생성된 청크: {stats['steps']['chunks_created']}")
        print(f"생성된 임베딩: {stats['steps']['embeddings_created']}")
        print(f"저장된 문서: {stats['steps']['documents_stored']}")
        
        if stats['status'] == 'success':
            print("\n✅ 데이터 수집 및 임베딩 생성 완료!")
            print("이제 performance_test.py를 실행하여 성능을 테스트할 수 있습니다.")
        else:
            print(f"\n❌ 파이프라인 실패: {stats.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        logger.exception("Pipeline execution failed")


if __name__ == "__main__":
    main()
