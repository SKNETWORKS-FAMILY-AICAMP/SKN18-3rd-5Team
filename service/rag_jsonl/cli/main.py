"""
RAG 시스템 통합 CLI

임베딩 생성, 검색, 평가 등의 기능을 통합한 명령줄 인터페이스입니다.
"""

import argparse
import sys
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ..core import MultiModelEmbedder, VectorRetriever, RAGEvaluator
from ..models.config import EmbeddingModelType

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """메인 CLI 함수"""
    
    parser = argparse.ArgumentParser(description='RAG 시스템 통합 CLI')
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # 임베딩 생성 명령어
    embed_parser = subparsers.add_parser('embed', help='5개 모델로 데이터 임베딩 생성')
    embed_parser.add_argument('--data-file', required=True, help='임베딩할 JSON 데이터 파일 경로')
    embed_parser.add_argument('--db-config', help='데이터베이스 설정 파일 (선택사항)')
    
    # 검색 명령어
    search_parser = subparsers.add_parser('search', help='문서 검색')
    search_parser.add_argument('--model', required=True, choices=[m.value for m in EmbeddingModelType], 
                              help='사용할 모델')
    search_parser.add_argument('--query', required=True, help='검색 쿼리')
    search_parser.add_argument('--top-k', type=int, default=5, help='반환할 결과 수')
    
    # 평가 명령어
    eval_parser = subparsers.add_parser('evaluate', help='모델 성능 평가')
    eval_parser.add_argument('--model', choices=[m.value for m in EmbeddingModelType], 
                            help='평가할 모델 (지정하지 않으면 모든 모델)')
    eval_parser.add_argument('--queries', help='평가용 쿼리 파일 (선택사항)')
    
    # 통합 실행 명령어
    run_parser = subparsers.add_parser('run', help='전체 파이프라인 실행 (임베딩 + 평가)')
    run_parser.add_argument('--data-file', required=True, help='임베딩할 JSON 데이터 파일 경로')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'embed':
            run_embed_command(args)
        elif args.command == 'search':
            run_search_command(args)
        elif args.command == 'evaluate':
            run_evaluate_command(args)
        elif args.command == 'run':
            run_full_pipeline(args)
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"명령어 실행 실패: {e}")
        sys.exit(1)


def run_embed_command(args):
    """임베딩 생성 명령어 실행"""
    
    logger.info("🚀 5개 모델 임베딩 생성 시작")
    
    embedder = MultiModelEmbedder(args.data_file)
    results = embedder.embed_all_models()
    
    print(embedder.get_summary())
    
    # 결과를 파일로 저장
    import json
    output_file = "embedding_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"결과가 저장되었습니다: {output_file}")


def run_search_command(args):
    """검색 명령어 실행"""
    
    model_type = EmbeddingModelType(args.model)
    logger.info(f"🔍 {model_type.value} 모델로 검색 시작")
    
    retriever = VectorRetriever(model_type)
    
    try:
        results = retriever.search(args.query, args.top_k)
        
        print(f"\n🔍 검색 결과: '{args.query}'")
        print("="*50)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. 유사도: {result.get('similarity', 0):.4f}")
                print(f"   내용: {result.get('content', '')[:200]}...")
                print()
        else:
            print("검색 결과가 없습니다.")
            
    finally:
        retriever.close()


def run_evaluate_command(args):
    """평가 명령어 실행"""
    
    logger.info("📊 모델 성능 평가 시작")
    
    evaluator = RAGEvaluator()
    
    if args.model:
        # 단일 모델 평가
        model_type = EmbeddingModelType(args.model)
        result = evaluator.evaluate_model(model_type)
        
        print(f"\n📊 {model_type.value} 모델 평가 결과")
        print("="*50)
        print(f"총 쿼리: {result.get('total_queries', 0)}개")
        print(f"성공한 쿼리: {result.get('successful_queries', 0)}개")
        
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"평균 검색 시간: {metrics.get('avg_search_time_ms', 0):.2f}ms")
            print(f"평균 유사도: {metrics.get('avg_similarity', 0):.4f}")
            print(f"한국어 이해도: {metrics.get('korean_understanding_score', 0):.4f}")
    else:
        # 모든 모델 평가
        models = list(EmbeddingModelType)
        results = evaluator.evaluate_all_models(models)
        
        print(results['summary'])
        
        # 결과를 파일로 저장
        import json
        output_file = "evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"결과가 저장되었습니다: {output_file}")


def run_full_pipeline(args):
    """전체 파이프라인 실행"""
    
    logger.info("🚀 전체 RAG 파이프라인 실행 시작")
    
    # 1. 임베딩 생성
    logger.info("1단계: 임베딩 생성")
    embedder = MultiModelEmbedder(args.data_file)
    embed_results = embedder.embed_all_models()
    print(embedder.get_summary())
    
    # 2. 성능 평가
    logger.info("2단계: 성능 평가")
    evaluator = RAGEvaluator()
    models = list(EmbeddingModelType)
    eval_results = evaluator.evaluate_all_models(models)
    print(eval_results['summary'])
    
    # 3. 최종 결과 저장
    import json
    final_results = {
        'embedding_results': embed_results,
        'evaluation_results': eval_results
    }
    
    output_file = "rag_pipeline_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"전체 파이프라인 완료! 결과: {output_file}")


if __name__ == "__main__":
    main()
