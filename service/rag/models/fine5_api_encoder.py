#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FinE5 API를 통한 임베딩 인코더
AbaciNLP API를 사용하여 FinE5 모델로 임베딩 생성
"""

import logging
import os
from typing import List, Optional, Dict, Any
import numpy as np
import requests
from dotenv import load_dotenv

try:
    from .config import EmbeddingModelType
except ImportError:
    # 테스트용 절대 import
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from service.rag.models.config import EmbeddingModelType

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)


class FinE5APIEncoder:
    """FinE5 API를 통한 임베딩 인코더"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "abacinlp-text-v1"):
        """
        Args:
            api_key: AbaciNLP API 키
            model_name: 사용할 모델 이름 (기본값: abacinlp-text-v1)
        """
        self.api_key = api_key or os.getenv('FIN_E5_API_KEY')
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("FinE5 API 키가 필요합니다. FIN_E5_API_KEY 환경변수를 설정하거나 api_key를 전달해주세요.")
        
        # AbaciNLP API 설정
        self.base_url = "https://abacinlp.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"FinE5 API 인코더 초기화 완료: {model_name}")
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """FinE5 모델을 위한 instruction 포맷팅"""
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def encode_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        문서들을 임베딩으로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기 (API 제한 고려)
            show_progress: 진행률 표시 여부
            
        Returns:
            임베딩 벡터 리스트
        """
        if not texts:
            return []
        
        # 배치 크기 설정 (API 제한 고려)
        if batch_size is None:
            batch_size = min(10, len(texts))  # API 제한을 고려한 작은 배치 크기
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if show_progress:
                logger.info(f"FinE5 API 배치 처리 중: {i+1}/{len(texts)}")
            
            batch_embeddings = self._encode_batch(batch_texts)
            embeddings.extend(batch_embeddings)
        
        logger.info(f"FinE5 API 임베딩 완료: {len(embeddings)}개 문서")
        return embeddings
    
    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 단위로 임베딩 생성"""
        try:
            # FinE5 모델을 위한 instruction 포맷팅
            formatted_texts = []
            for text in texts:
                formatted_text = self.get_detailed_instruct(
                    "Given a financial document, retrieve the most relevant information.",
                    text
                )
                formatted_texts.append(formatted_text)
            
            # AbaciNLP API 호출
            payload = {
                "model": self.model_name,
                "input": formatted_texts,
                "encoding_format": "float"
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API 호출 실패: {response.status_code} - {response.text}")
            
            response_data = response.json()
            
            # 응답에서 임베딩 추출
            embeddings = []
            for item in response_data["data"]:
                embeddings.append(item["embedding"])
            
            return embeddings
            
        except Exception as e:
            logger.error(f"FinE5 API 배치 처리 실패: {e}")
            raise
    
    def encode_query(self, query: str) -> List[float]:
        """
        쿼리를 임베딩으로 변환
        
        Args:
            query: 검색 쿼리
            
        Returns:
            쿼리 임베딩 벡터
        """
        try:
            # 쿼리용 instruction 포맷팅
            formatted_query = self.get_detailed_instruct(
                "Given a financial question, retrieve user replies that best answer the question.",
                query
            )
            
            # AbaciNLP API 호출
            payload = {
                "model": self.model_name,
                "input": [formatted_query],
                "encoding_format": "float"
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API 호출 실패: {response.status_code} - {response.text}")
            
            response_data = response.json()
            return response_data["data"][0]["embedding"]
            
        except Exception as e:
            logger.error(f"FinE5 API 쿼리 처리 실패: {e}")
            raise
    
    def get_dimension(self) -> int:
        """임베딩 차원 반환 (FinE5는 4096 차원)"""
        return 4096
    
    def get_model_name(self) -> str:
        """모델 이름 반환"""
        return self.model_name


def test_fine5_api():
    """FinE5 API 테스트"""
    try:
        # API 키 설정
        api_key = "sk-Syqe5xW9s13vVAoKWLvycW9MJtcoaTwFjMjGY7GJVrS6bI3r"
        
        # 인코더 생성
        encoder = FinE5APIEncoder(api_key=api_key)
        
        # 테스트 텍스트
        test_texts = [
            "삼성전자 주식 가격이 상승했습니다.",
            "금융 시장에서 변동성이 증가하고 있습니다."
        ]
        
        # 문서 임베딩 테스트
        print("📄 문서 임베딩 테스트...")
        doc_embeddings = encoder.encode_documents(test_texts, show_progress=True)
        print(f"✅ 문서 임베딩 완료: {len(doc_embeddings)}개, 차원: {len(doc_embeddings[0])}")
        
        # 쿼리 임베딩 테스트
        print("\n🔍 쿼리 임베딩 테스트...")
        query_embedding = encoder.encode_query("주식 시장 동향은?")
        print(f"✅ 쿼리 임베딩 완료: 차원 {len(query_embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ FinE5 API 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    # 테스트 실행
    print("🧪 FinE5 API 테스트 시작")
    print("=" * 50)
    
    success = test_fine5_api()
    
    if success:
        print("\n🎉 FinE5 API 테스트 성공!")
    else:
        print("\n💥 FinE5 API 테스트 실패!")
