#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 파이프라인 평가 도구
전체 LangGraph 파이프라인 (Router → QueryRewrite → Retrieve → Rerank → ContextTrim → Generate → GroundingCheck → Guardrail → Answer)을 평가합니다.
"""

import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # service/rag/evaluation/ -> project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graph.app_graph import build_app
from graph.state import QAState
from graph.utils.level import defaults as get_level_config

# unified_evaluator에서 RAGASEvaluator import
sys.path.insert(0, str(current_file.parent))
from unified_evaluator import RAGASEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangGraphEvaluator:
    """LangGraph 파이프라인 평가기"""
    
    def __init__(self, model: str = "gemma", verify_metadata: bool = True):
        """
        초기화
        
        Args:
            model: 사용할 LLM 모델 ("gemma" 또는 "openai")
            verify_metadata: 메타데이터 검증 실행 여부
        """
        self.model = model
        self.graph = build_app()
        self.pipeline_name = "LangGraph Full Pipeline"
        
        # RAGAs 평가기 초기화
        self.ragas_evaluator = RAGASEvaluator(use_ragas_library=False)
        
        # 메타데이터 검증 실행 (기본값: True)
        if verify_metadata:
            self._verify_metadata_once()
        
        logger.info(f"LangGraph Evaluator 초기화 완료")
        logger.info(f"  - Pipeline: {self.pipeline_name}")
        logger.info(f"  - Model: {model}")
    
    def _verify_metadata_once(self):
        """메타데이터 추출 검증 (한 번만 실행)"""
        try:
            from service.rag.vectorstore.pgvector_store import PgVectorStore
            from service.rag.models.encoder import EmbeddingEncoder
            from service.rag.models.config import EmbeddingModelType
            
            logger.info("🔍 메타데이터 추출 검증 중...")
            
            # 벡터 스토어 초기화
            store = PgVectorStore()
            encoder = EmbeddingEncoder(EmbeddingModelType.MULTILINGUAL_E5_SMALL)
            
            # 간단한 테스트 쿼리
            test_query = "테스트"
            query_embedding = encoder.encode_query(test_query)
            
            # 검색 실행
            results = store.search_similar(
                query_embedding=query_embedding,
                model_type=EmbeddingModelType.MULTILINGUAL_E5_SMALL,
                top_k=1,
                min_similarity=0.0
            )
            
            # 메타데이터 확인
            if results and results[0].metadata:
                has_corp_name = bool(results[0].corp_name or results[0].metadata.get('corp_name'))
                if has_corp_name:
                    logger.info("✓ 메타데이터 추출 정상 작동")
                else:
                    logger.warning("⚠️ 메타데이터에 corp_name이 없습니다")
            else:
                logger.warning("⚠️ 검색 결과가 없거나 메타데이터가 없습니다")
                
        except Exception as e:
            logger.warning(f"⚠️ 메타데이터 검증 실패: {e}")
    
    def _load_evaluation_queries(self) -> List[Dict[str, Any]]:
        """평가 쿼리 로드"""
        queries_path = project_root / "service" / "rag" / "evaluation" / "evaluation_queries.json"
        try:
            with open(queries_path, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            logger.info(f"평가 쿼리 로드 완료: {len(queries)}개")
            return queries
        except Exception as e:
            logger.error(f"평가 쿼리 로드 실패: {e}")
            return []
    
    def _create_initial_state(self, query_data: Dict[str, Any], user_level: str = "intermediate") -> QAState:
        """
        초기 State 생성
        
        Args:
            query_data: 쿼리 데이터
            user_level: 사용자 수준 ("beginner", "intermediate", "advanced")
        
        Returns:
            초기 QAState
        """
        # 레벨별 설정 가져오기
        meta = get_level_config(user_level)
        
        initial_state = QAState(
            question=query_data.get("query", ""),
            user_level=user_level,
            rewritten_query="",  # QueryRewrite 노드에서 채워짐
            retrieved=[],  # Retrieve 노드에서 채워짐
            reranked=[],  # Rerank 노드에서 채워짐
            context="",  # ContextTrim 노드에서 채워짐
            draft_answer="",  # Generate 노드에서 채워짐
            citations=[],  # ContextTrim 노드에서 채워짐
            grounded=False,  # GroundingCheck 노드에서 채워짐
            policy_flag=None,  # Guardrail 노드에서 채워짐
            meta=meta
        )
        
        return initial_state
    
    def run_evaluation(
        self,
        queries: Optional[List[Dict[str, Any]]] = None,
        user_level: str = "intermediate"
    ) -> List[Dict[str, Any]]:
        """
        LangGraph 파이프라인 평가 실행
        
        Args:
            queries: 평가 쿼리 리스트 (기본값: evaluation_queries.json)
            user_level: 사용자 수준
        
        Returns:
            평가 결과 리스트
        """
        if queries is None:
            queries = self._load_evaluation_queries()
        
        logger.info("🚀 LangGraph 파이프라인 평가 시작")
        logger.info(f"📊 Pipeline: {self.pipeline_name}")
        logger.info(f"👤 User Level: {user_level}")
        logger.info(f"🔍 Total Queries: {len(queries)}")
        logger.info("=" * 80)
        
        results = []
        
        for i, query_data in enumerate(queries, 1):
            query_id = query_data.get("query_id", f"Q{i:03d}")
            query_text = query_data.get("query", "")
            
            logger.info(f"[{i}/{len(queries)}] {query_id}: {query_text[:50]}...")
            
            try:
                start_time = time.time()
                
                # 초기 State 생성
                initial_state = self._create_initial_state(query_data, user_level)
                
                # LangGraph 실행
                final_state = self.graph.invoke(initial_state)
                
                execution_time = (time.time() - start_time) * 1000  # ms
                
                # RAGAs 평가 수행
                answer = final_state.get("draft_answer", "")
                context_text = final_state.get("context", "")
                contexts = [ctx for ctx in context_text.split("\n\n---\n\n") if ctx.strip()]
                ground_truth = query_data.get("ground_truth_answer")
                
                # 검색된 문서 정보 추출 (verify_metadata 스타일)
                retrieved_docs = []
                for doc in final_state.get("retrieved", []):
                    retrieved_docs.append({
                        "chunk_id": doc.get("chunk_id"),
                        "content_preview": doc.get("chunk_text", "")[:200] if isinstance(doc.get("chunk_text"), str) else "",
                        "corp_name": doc.get("metadata", {}).get("corp_name") if isinstance(doc.get("metadata"), dict) else None,
                        "document_name": doc.get("metadata", {}).get("document_name") if isinstance(doc.get("metadata"), dict) else None,
                        "doc_type": doc.get("metadata", {}).get("doc_type") if isinstance(doc.get("metadata"), dict) else None,
                        "rcept_dt": doc.get("metadata", {}).get("rcept_dt") if isinstance(doc.get("metadata"), dict) else None,
                        "fiscal_year": doc.get("metadata", {}).get("fiscal_year") if isinstance(doc.get("metadata"), dict) else None
                    })
                
                ragas_evaluation = self.ragas_evaluator.evaluate(
                    query=query_text,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth
                )
                
                # 결과 구성
                result = {
                    "query_id": query_id,
                    "query": query_text,
                    "query_type": query_data.get("query_type", "unknown"),
                    "difficulty": query_data.get("difficulty", "medium"),
                    "user_level": user_level,
                    
                    # LangGraph 파이프라인 결과
                    "retrieved_count": len(final_state.get("retrieved", [])),
                    "reranked_count": len(final_state.get("reranked", [])),
                    "citations_count": len(final_state.get("citations", [])),
                    "context_length": len(context_text),
                    "answer": answer,
                    "context": context_text,  # 전체 context 추가
                    "grounded": final_state.get("grounded", False),
                    "policy_flag": final_state.get("policy_flag"),
                    
                    # 검색된 문서 정보 (verify_metadata 스타일)
                    "retrieved_docs": retrieved_docs,
                    
                    # Citations 정보 (메타데이터 포함)
                    "citations": self._extract_citations_info(final_state.get("citations", [])),
                    
                    # RAGAs 평가 점수
                    "ragas_scores": {
                        "faithfulness": ragas_evaluation.faithfulness,
                        "answer_relevancy": ragas_evaluation.answer_relevancy,
                        "context_precision": ragas_evaluation.context_precision,
                        "context_recall": ragas_evaluation.context_recall,
                        "average_score": ragas_evaluation.average_score
                    },
                    
                    # Execution metrics
                    "execution_time_ms": execution_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                
                # 파이프라인 검증
                pipeline_validation = self._validate_pipeline_output(result, final_state)
                result["pipeline_validation"] = pipeline_validation
                
                logger.info(f"✅ {query_id} 완료")
                logger.info(f"   - Retrieved: {result['retrieved_count']} docs")
                logger.info(f"   - Reranked: {result['reranked_count']} docs")
                logger.info(f"   - Citations: {result['citations_count']}")
                logger.info(f"   - Grounded: {result['grounded']}")
                logger.info(f"   - RAGAs Score: {ragas_evaluation.average_score:.3f}")
                logger.info(f"   - Pipeline Valid: {pipeline_validation['is_valid']}")
                logger.info(f"   - Time: {execution_time:.0f}ms")
                
            except Exception as e:
                logger.error(f"❌ {query_id} 오류: {e}")
                results.append({
                    "query_id": query_id,
                    "query": query_text,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        logger.info("🎉 LangGraph 파이프라인 평가 완료!")
        logger.info(f"📈 총 {len(results)}개 쿼리 처리 완료")
        
        return results
    
    def _validate_pipeline_output(self, result: Dict[str, Any], final_state: QAState) -> Dict[str, Any]:
        """
        LangGraph 파이프라인 출력 검증
        
        Args:
            result: 평가 결과 딕셔너리
            final_state: LangGraph 최종 상태
        
        Returns:
            검증 결과 딕셔너리
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        # 1. 검색 단계 검증
        if result['retrieved_count'] == 0:
            validation["issues"].append("No documents retrieved")
            validation["is_valid"] = False
        
        # 2. 메타데이터 검증
        retrieved_docs = result.get('retrieved_docs', [])
        metadata_complete = 0
        for doc in retrieved_docs:
            if doc.get('corp_name') and doc.get('document_name'):
                metadata_complete += 1
        
        if retrieved_docs and metadata_complete == 0:
            validation["issues"].append("No metadata found in retrieved documents")
            validation["is_valid"] = False
        elif retrieved_docs and metadata_complete < len(retrieved_docs):
            validation["warnings"].append(f"Incomplete metadata: {metadata_complete}/{len(retrieved_docs)} docs")
        
        # 3. Context 검증
        if not result.get('context') or len(result['context']) < 50:
            validation["issues"].append("Context too short or empty")
            validation["is_valid"] = False
        
        # 4. 답변 검증
        answer = result.get('answer', '')
        if not answer or len(answer) < 10:
            validation["issues"].append("Answer too short or empty")
            validation["is_valid"] = False
        elif "오류가 발생했습니다" in answer:
            validation["issues"].append("Answer generation failed")
            validation["is_valid"] = False
        
        # 5. Grounding 검증
        if not result.get('grounded'):
            validation["warnings"].append("Answer not grounded (no [ref:] citations)")
        
        # 6. Citations 검증
        citations = result.get('citations', [])
        if not citations:
            validation["warnings"].append("No citations extracted")
        else:
            citation_with_metadata = sum(1 for c in citations if c.get('corp_name'))
            if citation_with_metadata == 0:
                validation["issues"].append("Citations missing metadata")
                validation["is_valid"] = False
        
        # 7. RAGAs 점수 검증
        ragas_scores = result.get('ragas_scores', {})
        avg_score = ragas_scores.get('average_score', 0)
        if avg_score < 0.3:
            validation["warnings"].append(f"Low RAGAs score: {avg_score:.3f}")
        
        return validation
    
    def _extract_citations_info(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Citations 정보 추출 (메타데이터 포함)
        
        Args:
            citations: citations 리스트
        
        Returns:
            추출된 citations 정보
        """
        extracted = []
        for citation in citations:
            citation_info = {
                "corp_name": citation.get("corp_name"),
                "document_name": citation.get("document_name"),
                "doc_type": citation.get("doc_type"),
                "chunk_id": citation.get("chunk_id")
            }
            extracted.append(citation_info)
        return extracted
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str = "results") -> Dict[str, str]:
        """
        결과를 파일로 저장
        
        Args:
            results: 평가 결과 리스트
            output_dir: 출력 디렉토리
        
        Returns:
            저장된 파일 경로 딕셔너리
        """
        current_file = Path(__file__).resolve()
        rag_dir = current_file.parents[1]
        
        if Path(output_dir).is_absolute():
            output_path = Path(output_dir)
        else:
            if output_dir == "results":
                output_path = rag_dir / "results"
            else:
                output_path = rag_dir / output_dir
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 상세 결과 저장 (JSON)
        detailed_file = output_path / f"langgraph_evaluation_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 요약 보고서 생성
        summary_file = output_path / f"langgraph_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"LangGraph Pipeline Evaluation Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Pipeline: {self.pipeline_name}\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Total Samples: {len(results)}\n\n")
            
            # 통계 정보
            successful = [r for r in results if "error" not in r and "ragas_scores" in r]
            valid_pipelines = [r for r in successful if r.get('pipeline_validation', {}).get('is_valid', False)]
            
            f.write(f"=== Summary Statistics ===\n")
            f.write(f"Successful: {len(successful)}/{len(results)}\n")
            f.write(f"Valid Pipelines: {len(valid_pipelines)}/{len(successful)}\n")
            f.write(f"Grounded Answers: {sum(1 for r in successful if r.get('grounded'))}\n")
            f.write(f"Avg Retrieved: {sum(r.get('retrieved_count', 0) for r in successful) / len(successful) if successful else 0:.2f}\n")
            f.write(f"Avg Citations: {sum(r.get('citations_count', 0) for r in successful) / len(successful) if successful else 0:.2f}\n")
            f.write(f"Avg Execution Time: {sum(r.get('execution_time_ms', 0) for r in successful) / len(successful) if successful else 0:.2f}ms\n")
            if successful:
                f.write(f"\n=== RAGAs Metrics ===\n")
                ragas_scores = [r.get('ragas_scores', {}) for r in successful if r.get('ragas_scores')]
                if ragas_scores:
                    avg_faithfulness = sum(s.get('faithfulness', 0) or 0 for s in ragas_scores) / len(ragas_scores)
                    avg_relevancy = sum(s.get('answer_relevancy', 0) or 0 for s in ragas_scores) / len(ragas_scores)
                    avg_precision = sum(s.get('context_precision', 0) or 0 for s in ragas_scores) / len(ragas_scores)
                    
                    # Context Recall (ground_truth가 있을 때만)
                    recall_scores = [s.get('context_recall') for s in ragas_scores if s.get('context_recall') is not None]
                    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else None
                    
                    avg_score = sum(s.get('average_score', 0) or 0 for s in ragas_scores) / len(ragas_scores)
                    
                    f.write(f"Avg Faithfulness: {avg_faithfulness:.3f}\n")
                    f.write(f"Avg Answer Relevancy: {avg_relevancy:.3f}\n")
                    f.write(f"Avg Context Precision: {avg_precision:.3f}\n")
                    if avg_recall is not None:
                        f.write(f"Avg Context Recall: {avg_recall:.3f}\n")
                    f.write(f"Avg RAGAs Score: {avg_score:.3f}\n")
            f.write("\n")
            
            # 개별 결과
            f.write("=== Individual Results ===\n")
            for result in results:
                f.write(f"\n{result['query_id']}: {result['query'][:60]}...\n")
                if "error" in result:
                    f.write(f"  Error: {result['error']}\n")
                else:
                    f.write(f"  Retrieved: {result.get('retrieved_count', 0)}\n")
                    f.write(f"  Reranked: {result.get('reranked_count', 0)}\n")
                    f.write(f"  Citations: {result.get('citations_count', 0)}\n")
                    f.write(f"  Grounded: {result.get('grounded', False)}\n")
                    
                    # Pipeline validation
                    validation = result.get('pipeline_validation', {})
                    f.write(f"  Pipeline Valid: {validation.get('is_valid', False)}\n")
                    if validation.get('issues'):
                        f.write(f"    Issues: {', '.join(validation['issues'])}\n")
                    if validation.get('warnings'):
                        f.write(f"    Warnings: {', '.join(validation['warnings'])}\n")
                    
                    # RAGAs scores
                    if 'ragas_scores' in result and result['ragas_scores']:
                        f.write(f"  RAGAs Score: {result['ragas_scores'].get('average_score', 0):.3f}\n")
                        f.write(f"    - Faithfulness: {result['ragas_scores'].get('faithfulness', 0):.3f}\n")
                        f.write(f"    - Answer Relevancy: {result['ragas_scores'].get('answer_relevancy', 0):.3f}\n")
                        f.write(f"    - Context Precision: {result['ragas_scores'].get('context_precision', 0):.3f}\n")
                        recall = result['ragas_scores'].get('context_recall')
                        if recall is not None:
                            f.write(f"    - Context Recall: {recall:.3f}\n")
                    f.write(f"  Time: {result.get('execution_time_ms', 0):.0f}ms\n")
        
        logger.info(f"💾 상세 결과 저장: {detailed_file}")
        logger.info(f"📄 요약 보고서 저장: {summary_file}")
        
        return {
            "detailed_results": str(detailed_file),
            "summary_report": str(summary_file)
        }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="LangGraph 파이프라인 평가 도구 - 전체 파이프라인 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 평가 (intermediate level)
  python langgraph_evaluator.py
  
  # Beginner level 평가
  python langgraph_evaluator.py --level beginner
  
  # Advanced level 평가
  python langgraph_evaluator.py --level advanced
  
  # 메타데이터 검증 없이 실행
  python langgraph_evaluator.py --skip-verify
        """
    )
    
    parser.add_argument("--level", type=str, default="intermediate", 
                       choices=["beginner", "intermediate", "advanced"],
                       help="사용자 수준 (기본값: intermediate)")
    parser.add_argument("--skip-verify", action="store_true",
                       help="메타데이터 검증 건너뛰기")
    
    args = parser.parse_args()
    
    # 평가 도구 초기화
    evaluator = LangGraphEvaluator(model="gemma", verify_metadata=not args.skip_verify)
    
    # 평가 실행
    results = evaluator.run_evaluation(user_level=args.level)
    
    # 결과 저장
    saved_files = evaluator.save_results(results, "results")
    
    # 결과 출력
    successful = [r for r in results if "error" not in r]
    print(f"\n📊 평가 결과 요약:")
    print(f"   - 총 쿼리 수: {len(results)}")
    print(f"   - 성공: {len(successful)}")
    print(f"   - 실패: {len(results) - len(successful)}")
    print(f"   - Grounded: {sum(1 for r in successful if r.get('grounded'))}")
    
    print(f"\n💾 결과 저장 위치:")
    print(f"   - 상세 결과: {saved_files['detailed_results']}")
    print(f"   - 요약 보고서: {saved_files['summary_report']}")


if __name__ == "__main__":
    main()

