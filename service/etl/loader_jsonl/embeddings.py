#!/usr/bin/env python3
"""
임베딩 생성 스크립트
PostgreSQL의 chunks 테이블을 읽어 임베딩을 생성하고 embeddings_* 테이블에 저장

사용법:
    # ETL loader 디렉토리에서 실행
    cd service/etl/loader_jsonl
    python embeddings.py --model e5 --batch-size 100
    
    # 프로젝트 루트에서 실행
    python service/etl/loader_jsonl/embeddings.py --model kakaobank --limit 1000
    python service/etl/loader_jsonl/embeddings.py --all-models
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from service.rag.models.config import EmbeddingModelType
from service.rag.models.encoder import EmbeddingEncoder
from service.rag.vectorstore.pgvector_store import PgVectorStore

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """임베딩 생성 및 저장"""
    
    def __init__(
        self,
        model_type: EmbeddingModelType,
        db_config: Optional[Dict[str, str]] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model_type: 임베딩 모델 타입
            db_config: 데이터베이스 연결 설정
            device: 디바이스 ('cuda', 'cpu', None)
        """
        self.model_type = model_type
        self.encoder = EmbeddingEncoder(model_type, device)
        self.vector_store = PgVectorStore(db_config)
        
        logger.info(f"임베딩 생성기 초기화: {self.encoder.get_display_name()}")
    
    def generate_for_all_chunks(
        self,
        batch_size: int = 100,
        limit: Optional[int] = None,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """모든 청크에 대해 임베딩 생성"""
        
        logger.info("=" * 80)
        logger.info(f"임베딩 생성 시작: {self.model_type.value}")
        logger.info("=" * 80)
        
        # 통계
        stats = {
            'total_chunks': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'batches': 0,
            'start_time': time.time()
        }
        
        try:
            # 1. 처리할 청크 가져오기
            chunks = self._get_chunks_to_process(limit, skip_existing)
            stats['total_chunks'] = len(chunks)
            
            if not chunks:
                logger.info("처리할 청크가 없습니다")
                return stats
            
            logger.info(f"처리할 청크: {len(chunks):,}개")
            
            # 2. 배치별로 처리
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                logger.info(f"배치 {batch_num}/{total_batches} 처리 중 ({len(batch)}개 청크)")
                
                try:
                    self._process_batch(batch)
                    stats['processed'] += len(batch)
                    stats['batches'] += 1
                    
                except Exception as e:
                    logger.error(f"배치 {batch_num} 처리 실패: {e}")
                    stats['errors'] += len(batch)
                
                # 진행률 표시
                progress = stats['processed'] / stats['total_chunks'] * 100
                logger.info(f"진행률: {stats['processed']:,}/{stats['total_chunks']:,} ({progress:.1f}%)")
            
            # 3. 통계 출력
            elapsed_time = time.time() - stats['start_time']
            stats['elapsed_time'] = elapsed_time
            
            logger.info("\n" + "=" * 80)
            logger.info("임베딩 생성 완료")
            logger.info("=" * 80)
            logger.info(f"총 청크: {stats['total_chunks']:,}")
            logger.info(f"처리됨: {stats['processed']:,}")
            logger.info(f"건너뜀: {stats['skipped']:,}")
            logger.info(f"오류: {stats['errors']:,}")
            logger.info(f"소요 시간: {elapsed_time:.1f}초")
            logger.info(f"처리 속도: {stats['processed'] / elapsed_time:.1f} 청크/초")
            
            return stats
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    def _get_chunks_to_process(
        self, 
        limit: Optional[int], 
        skip_existing: bool
    ) -> List[Dict[str, Any]]:
        """처리할 청크 목록 가져오기"""
        
        # 모델별 임베딩 테이블 매핑
        embedding_table = self.vector_store.MODEL_TABLE_MAP.get(self.model_type)
        
        if not embedding_table:
            raise ValueError(f"지원하지 않는 모델: {self.model_type}")
        
        # SQL 쿼리 작성
        if skip_existing:
            # 아직 임베딩이 없는 청크만
            sql = f"""
                SELECT c.id, c.chunk_id, c.natural_text, c.chunk_type, c.metadata
                FROM chunks c
                LEFT JOIN {embedding_table} e ON c.chunk_id = e.chunk_id
                WHERE e.chunk_id IS NULL
                  AND c.natural_text IS NOT NULL
                  AND c.natural_text != ''
                ORDER BY c.id
            """
        else:
            # 모든 청크
            sql = """
                SELECT id, chunk_id, natural_text, chunk_type, metadata
                FROM chunks
                WHERE natural_text IS NOT NULL
                  AND natural_text != ''
                ORDER BY id
            """
        
        if limit:
            sql += f" LIMIT {limit}"
        
        # 실행
        cursor = self.vector_store.conn.cursor()
        cursor.execute(sql)
        
        chunks = []
        for row in cursor.fetchall():
            chunks.append({
                'id': row[0],
                'chunk_id': row[1],
                'natural_text': row[2],
                'chunk_type': row[3],
                'metadata': row[4]
            })
        
        cursor.close()
        
        return chunks
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """배치 처리"""
        
        # 1. 텍스트 추출
        texts = [chunk['natural_text'] for chunk in batch]
        chunk_ids = [chunk['id'] for chunk in batch]
        
        # 2. 임베딩 생성
        embeddings = self.encoder.encode_batch(
            texts,
            is_query=False,  # document 임베딩
            show_progress=False
        )
        
        # 3. 데이터베이스에 저장
        self.vector_store.insert_embeddings(
            model_type=self.model_type,
            chunk_ids=chunk_ids,
            embeddings=embeddings
        )


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="PostgreSQL chunks → 임베딩 생성 → embeddings_* 테이블 저장",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # E5 모델로 임베딩 생성
  python service/etl/loader/generate_embeddings.py --model e5

  # 특정 개수만 처리
  python service/etl/loader/generate_embeddings.py --model kakaobank --limit 1000

  # 배치 크기 조정
  python service/etl/loader/generate_embeddings.py --model fine5 --batch-size 50

  # 모든 모델에 대해 임베딩 생성
  python service/etl/loader/generate_embeddings.py --all-models

  # 이미 생성된 것도 다시 생성 (강제)
  python service/etl/loader/generate_embeddings.py --model e5 --force
        """
    )
    
    # 모델 선택
    parser.add_argument(
        "--model",
        choices=["e5", "kakaobank", "fine5"],
        help="임베딩 모델"
    )
    
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="모든 모델에 대해 임베딩 생성"
    )
    
    # 처리 옵션
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="배치 크기 (기본값: 100)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="처리할 청크 수 제한"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 임베딩이 있는 청크도 다시 생성"
    )
    
    # 디바이스 설정
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="디바이스 (cuda 또는 cpu)"
    )
    
    # 데이터베이스 설정
    parser.add_argument(
        "--db-host",
        default="localhost",
        help="PostgreSQL 호스트"
    )
    
    parser.add_argument(
        "--db-port",
        type=int,
        default=5432,
        help="PostgreSQL 포트"
    )
    
    parser.add_argument(
        "--db-name",
        default="postgres",
        help="데이터베이스 이름"
    )
    
    parser.add_argument(
        "--db-user",
        default="postgres",
        help="데이터베이스 사용자"
    )
    
    parser.add_argument(
        "--db-password",
        default="postgres",
        help="데이터베이스 비밀번호"
    )
    
    args = parser.parse_args()
    
    # 모델 선택 검증
    if not args.model and not args.all_models:
        parser.error("--model 또는 --all-models 중 하나를 선택해야 합니다")
    
    if args.model and args.all_models:
        parser.error("--model과 --all-models를 동시에 사용할 수 없습니다")
    
    # 모델 타입 매핑
    model_mapping = {
        "e5": EmbeddingModelType.MULTILINGUAL_E5_SMALL,
        "kakaobank": EmbeddingModelType.KAKAOBANK_DEBERTA,
        "fine5": EmbeddingModelType.FINE5_FINANCE
    }
    
    # DB 설정
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    try:
        # 모델별 처리
        if args.all_models:
            models_to_process = list(model_mapping.keys())
        else:
            models_to_process = [args.model]
        
        total_stats = {}
        
        for model_name in models_to_process:
            model_type = model_mapping[model_name]
            
            logger.info(f"\n{'='*80}")
            logger.info(f"모델: {model_name}")
            logger.info(f"{'='*80}\n")
            
            # 임베딩 생성기 생성
            generator = EmbeddingGenerator(
                model_type=model_type,
                db_config=db_config,
                device=args.device
            )
            
            # 임베딩 생성
            stats = generator.generate_for_all_chunks(
                batch_size=args.batch_size,
                limit=args.limit,
                skip_existing=not args.force
            )
            
            total_stats[model_name] = stats
        
        # 전체 통계
        if len(total_stats) > 1:
            logger.info("\n" + "=" * 80)
            logger.info("전체 모델 통계")
            logger.info("=" * 80)
            
            for model_name, stats in total_stats.items():
                logger.info(f"{model_name}: {stats['processed']:,}개 처리, "
                          f"{stats['elapsed_time']:.1f}초 소요")
    
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다")
        sys.exit(1)
    except Exception as e:
        logger.error(f"임베딩 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

