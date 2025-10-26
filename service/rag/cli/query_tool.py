#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 쿼리 도구
사용자 쿼리를 받아서 검색하고 답변을 생성하는 CLI 도구 (대화형 모드)
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # service/rag/cli/ -> project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from service.rag.generation.generator import OllamaGenerator, OpenAIGenerator, GenerationConfig
from service.rag.retrieval.retriever import Retriever
from service.rag.models.config import EmbeddingModelType
from service.rag.cli.rag_cli import get_db_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGQueryTool:
    """RAG 쿼리 도구"""

    def __init__(self, model: str = "gemma"):
        """
        초기화

        Args:
            model: 사용할 모델 ("gemma" 또는 "openai")
        """
        self.model = model

        # 고급 RAG 시스템 초기화 (벡터 + 키워드 + 하이브리드 검색)
        self.rag_system = Retriever(
            model_type=EmbeddingModelType.MULTILINGUAL_E5_SMALL,
            db_config=get_db_config(),  # 데이터베이스 설정 추가
            enable_temporal_filter=True,
            enable_hybrid=True
        )

        # LLM Generator 초기화
        if model == "openai":
            logger.info("Using OpenAI Generator (gpt-5-nano)")
            self.generator = OpenAIGenerator(default_model="gpt-5-nano")
            model_name = "gpt-5-nano"
            timeout = 60
        else:  # gemma
            logger.info("Using Ollama Generator (gemma3:4b)")
            self.generator = OllamaGenerator(
                base_url="http://localhost:11434",
                default_model="gemma3:4b"
            )
            model_name = "gemma3:4b"
            timeout = 120

        # Generation Config
        self.generation_config = GenerationConfig(
            model=model_name,
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            timeout=timeout
        )

        logger.info("RAG Query Tool 초기화 완료")

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        corp_filter: Optional[str] = None,
        verbose: bool = False
    ) -> dict:
        """
        질문에 대한 답변 생성

        Args:
            question: 사용자 질문
            top_k: 검색할 문서 수
            min_similarity: 최소 유사도
            corp_filter: 기업 필터
            verbose: 상세 정보 출력 여부

        Returns:
            결과 딕셔너리 (question, answer, contexts, search_results)
        """
        logger.info(f"질문: {question}")

        # 1. 고급 검색 수행 (하이브리드 검색)
        logger.info(f"문서 검색 중... (top_k={top_k}, 하이브리드 검색)")
        search_results = self.rag_system.search(
            query=question,
            top_k=top_k,
            min_similarity=min_similarity,
            search_method="hybrid",  # 벡터 + 키워드 하이브리드
            use_reranker=True,  # 리랭킹 사용
            include_context=True
        )

        if not search_results:
            logger.warning("검색 결과가 없습니다.")
            return {
                "question": question,
                "answer": "죄송합니다. 관련 정보를 찾을 수 없습니다.",
                "contexts": [],
                "search_results": []
            }

        logger.info(f"검색 완료: {len(search_results)}개 문서 발견")

        # 2. 컨텍스트 추출 (natural_text 또는 chunk_text 필드 사용)
        contexts = [
            result.get("natural_text", result.get("chunk_text", ""))
            for result in search_results
        ]
        # 빈 컨텍스트 필터링
        contexts = [ctx for ctx in contexts if ctx.strip()]
        context_text = "\n\n".join(contexts)

        if verbose:
            logger.info(f"컨텍스트 길이: {len(context_text)} chars")
            for i, result in enumerate(search_results, 1):
                logger.info(f"  [{i}] {result.get('title', 'N/A')} (유사도: {result.get('similarity', 0):.3f})")

        # 3. 답변 생성
        logger.info("답변 생성 중...")
        try:
            generated_result = self.generator.generate(
                query=question,
                context=context_text,
                config=self.generation_config
            )
            answer = generated_result.answer

            logger.info(f"답변 생성 완료: {len(answer)} chars")
            if generated_result.generation_time_ms:
                logger.info(f"생성 시간: {generated_result.generation_time_ms:.0f}ms")

        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "search_results": search_results
        }

    def interactive_mode(
        self,
        top_k: int = 5,
        min_similarity: float = 0.0,
        corp_filter: Optional[str] = None
    ):
        """
        대화형 모드

        Args:
            top_k: 검색할 문서 수
            min_similarity: 최소 유사도
            corp_filter: 기업 필터
        """
        print("=" * 80)
        print("RAG 쿼리 도구 - 대화형 모드")
        print(f"모델: {self.model}")
        print(f"Top-K: {top_k}, Min Similarity: {min_similarity}")
        if corp_filter:
            print(f"기업 필터: {corp_filter}")
        print("=" * 80)
        print("질문을 입력하세요 (종료: 'exit', 'quit', 'q')")
        print()

        while True:
            try:
                # 질문 입력
                question = input("질문> ").strip()

                if not question:
                    continue

                # 종료 명령
                if question.lower() in ['exit', 'quit', 'q']:
                    print("종료합니다.")
                    break

                print()

                # 쿼리 실행
                result = self.query(
                    question=question,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    corp_filter=corp_filter,
                    verbose=False
                )

                # 결과 출력
                print("=" * 80)
                print("답변:")
                print("-" * 80)
                print(result["answer"])
                print("=" * 80)
                print(f"참고 문서: {len(result['search_results'])}개")
                for i, doc in enumerate(result['search_results'], 1):
                    print(f"  [{i}] {doc.get('title', 'N/A')} (유사도: {doc.get('similarity', 0):.3f})")
                print()

            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                break
            except Exception as e:
                logger.error(f"오류 발생: {e}")
                print(f"오류: {e}")
                print()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="RAG 쿼리 도구 - 질문에 대한 답변 생성 (대화형 모드)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 대화형 모드 (Gemma - 기본값)
  python query_tool.py
  python query_tool.py --model gemma

  # 대화형 모드 (OpenAI)
  python query_tool.py --model openai
        """
    )

    # 모델 옵션
    parser.add_argument("--model", type=str, default="gemma", choices=["gemma", "openai"],
                       help="사용할 모델 (기본값: gemma)")

    args = parser.parse_args()

    # 쿼리 도구 초기화
    tool = RAGQueryTool(model=args.model)

    # 대화형 모드 실행
    tool.interactive_mode()


if __name__ == "__main__":
    main()
