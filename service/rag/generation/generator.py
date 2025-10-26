#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Generator
Ollama를 통한 답변 생성
"""

import logging
import requests
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """생성 설정"""
    model: str = "gemma3:4b"  # gemma3:4b 모델 사용
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    timeout: int = 30


@dataclass
class GeneratedAnswer:
    """생성된 답변"""
    answer: str
    model: str
    tokens_used: Optional[int] = None
    generation_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMGenerator(ABC):
    """LLM 생성기 기본 클래스"""

    @abstractmethod
    def generate(
        self,
        query: str,
        context: str,
        config: Optional[GenerationConfig] = None
    ) -> GeneratedAnswer:
        """답변 생성"""
        pass


class OllamaGenerator(LLMGenerator):
    """Ollama 기반 생성기"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "gemma2:2b"
    ):
        """
        Args:
            base_url: Ollama API URL
            default_model: 기본 모델명
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.api_url = f"{self.base_url}/api/generate"

        logger.info(f"OllamaGenerator initialized with model: {default_model}")

    def _create_prompt(self, query: str, context: str) -> str:
        """프롬프트 생성"""
        prompt = f"""당신은 청년 주거 정책 전문 상담사입니다. 주어진 문서를 바탕으로 사용자의 질문에 정확하고 친절하게 답변해주세요.

## 참고 문서
{context}

## 사용자 질문
{query}

## 답변 작성 규칙
1. **정확성**: 참고 문서의 내용만을 기반으로 답변하고, 문서에 없는 내용은 추측하지 마세요
2. **구조화**: 명확한 섹션으로 나누어 체계적으로 작성하세요 (예: 신청 자격, 금리, 한도, 신청 방법 등)
3. **완전성**: 표나 리스트를 작성할 때 중간에 끊기지 않도록 완전하게 작성하세요
4. **구체성**: 조건, 금액, 금리, 절차 등 구체적인 수치와 정보를 명시하세요
5. **명확성**: 마크다운 형식을 사용하여 읽기 쉽게 작성하세요 (**, -, 1. 등)
6. **관련성**: 질문과 직접 관련된 정보만 제공하고, 불필요한 내용은 제외하세요
7. **친절성**: 한국어로 자연스럽고 친절하게 답변하세요

답변:"""
        return prompt

    def generate(
        self,
        query: str,
        context: str,
        config: Optional[GenerationConfig] = None
    ) -> GeneratedAnswer:
        """
        답변 생성

        Args:
            query: 사용자 질문
            context: RAG로 검색된 컨텍스트
            config: 생성 설정

        Returns:
            생성된 답변
        """
        import time

        if config is None:
            config = GenerationConfig(model=self.default_model)

        # 프롬프트 생성
        prompt = self._create_prompt(query, context)

        # API 요청 페이로드
        payload = {
            "model": config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "top_p": config.top_p
            }
        }

        logger.info(f"Generating answer with {config.model}...")
        start_time = time.time()

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=config.timeout
            )
            response.raise_for_status()

            result = response.json()
            generation_time = (time.time() - start_time) * 1000

            # 응답 파싱
            answer_text = result.get("response", "").strip()

            generated_answer = GeneratedAnswer(
                answer=answer_text,
                model=config.model,
                tokens_used=result.get("eval_count"),
                generation_time_ms=generation_time,
                metadata={
                    "prompt_tokens": result.get("prompt_eval_count"),
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration")
                }
            )

            logger.info(f"Answer generated successfully in {generation_time:.2f}ms")
            return generated_answer

        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            raise Exception("LLM 생성 시간 초과")
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama server")
            raise Exception("Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise Exception(f"답변 생성 중 오류 발생: {str(e)}")

    def check_health(self) -> bool:
        """Ollama 서버 상태 확인"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()

            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
