#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Generator
Ollama 및 OpenAI를 통한 답변 생성
"""

import logging
import requests
import os
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).resolve().parents[2] / '.env'
if env_path.exists():
    load_dotenv(env_path)
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded .env from {env_path}")
else:
    logger = logging.getLogger(__name__)
    logger.warning(f".env file not found at {env_path}")

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
        default_model: str = "gemma3:4b"
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
        prompt = f"""당신은 금융 및 기업 분석 전문가입니다. 주어진 문서를 바탕으로 사용자의 질문에 정확하고 전문적으로 답변해주세요.

## 참고 문서
{context}

## 사용자 질문
{query}

## 답변 작성 규칙
1. **정확성**: 참고 문서의 내용만을 기반으로 답변하고, 문서에 없는 내용은 추측하지 마세요
2. **구체성**: 매출액, 영업이익, 성장률 등 구체적인 수치와 정보를 명시하세요
3. **구조화**: 명확한 섹션으로 나누어 체계적으로 작성하세요
4. **완전성**: 표나 리스트를 작성할 때 중간에 끊기지 않도록 완전하게 작성하세요
5. **명확성**: 마크다운 형식을 사용하여 읽기 쉽게 작성하세요
6. **관련성**: 질문과 직접 관련된 정보만 제공하고, 불필요한 내용은 제외하세요
7. **전문성**: 금융 및 기업 분석 용어를 정확하게 사용하세요

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


class OpenAIGenerator(LLMGenerator):
    """OpenAI 기반 생성기"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-5-nano",
        reasoning_effort: str = "high"
    ):
        """
        Args:
            api_key: OpenAI API 키 (None이면 환경 변수에서 로드)
            default_model: 기본 모델명
            reasoning_effort: 논리성 강화 수준 (low, medium, high)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.default_model = default_model
        self.reasoning_effort = reasoning_effort
        self.api_url = "https://api.openai.com/v1/chat/completions"

        logger.info(f"OpenAIGenerator initialized with model: {default_model}, reasoning_effort: {reasoning_effort}")

    def _create_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """메시지 생성"""
        system_message = """당신은 금융 및 기업 분석 전문가입니다. 주어진 문서를 바탕으로 사용자의 질문에 정확하고 전문적으로 답변해주세요.

답변 작성 규칙:
1. **정확성**: 참고 문서의 내용만을 기반으로 답변하고, 문서에 없는 내용은 추측하지 마세요
2. **구체성**: 매출액, 영업이익, 성장률 등 구체적인 수치와 정보를 명시하세요
3. **구조화**: 명확한 섹션으로 나누어 체계적으로 작성하세요
4. **완전성**: 표나 리스트를 작성할 때 중간에 끊기지 않도록 완전하게 작성하세요
5. **명확성**: 마크다운 형식을 사용하여 읽기 쉽게 작성하세요
6. **관련성**: 질문과 직접 관련된 정보만 제공하고, 불필요한 내용은 제외하세요
7. **전문성**: 금융 및 기업 분석 용어를 정확하게 사용하세요"""

        user_message = f"""## 참고 문서
{context}

## 사용자 질문
{query}

위 참고 문서를 바탕으로 질문에 답변해주세요."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

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

        # 메시지 생성
        messages = self._create_messages(query, context)

        # API 요청 페이로드
        payload = {
            "model": config.model,
            "messages": messages
        }

        # GPT-5 모델의 경우 특별한 파라미터 사용
        if "gpt-5" in config.model.lower():
            # GPT-5는 temperature 기본값(1)만 지원하므로 제외
            payload["max_completion_tokens"] = config.max_tokens
            payload["reasoning_effort"] = self.reasoning_effort
            # top_p도 제외 (GPT-5가 지원하지 않을 수 있음)
        else:
            # 다른 모델은 기존 파라미터 사용
            payload["temperature"] = config.temperature
            payload["max_tokens"] = config.max_tokens
            payload["top_p"] = config.top_p

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"Generating answer with {config.model}...")
        start_time = time.time()

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=config.timeout
            )

            # 에러 발생 시 상세 정보 로깅
            if response.status_code != 200:
                logger.error(f"OpenAI API Error: {response.status_code}")
                logger.error(f"Request URL: {self.api_url}")
                logger.error(f"Request Headers: {headers}")
                logger.error(f"Request Payload: {payload}")
                logger.error(f"Response Headers: {dict(response.headers)}")
                logger.error(f"Response Body: {response.text}")
                
                # 400 에러의 경우 더 자세한 분석
                if response.status_code == 400:
                    try:
                        error_detail = response.json()
                        logger.error(f"Error Detail: {error_detail}")
                        if "error" in error_detail:
                            logger.error(f"Error Type: {error_detail['error'].get('type', 'Unknown')}")
                            logger.error(f"Error Message: {error_detail['error'].get('message', 'Unknown')}")
                    except:
                        logger.error("Could not parse error response as JSON")

            response.raise_for_status()

            result = response.json()
            generation_time = (time.time() - start_time) * 1000

            # 응답 파싱
            answer_text = result["choices"][0]["message"]["content"].strip()
            usage = result.get("usage", {})

            generated_answer = GeneratedAnswer(
                answer=answer_text,
                model=config.model,
                tokens_used=usage.get("completion_tokens"),
                generation_time_ms=generation_time,
                metadata={
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "finish_reason": result["choices"][0].get("finish_reason")
                }
            )

            logger.info(f"Answer generated successfully in {generation_time:.2f}ms")
            logger.info(f"Tokens used: {usage.get('total_tokens')} (prompt: {usage.get('prompt_tokens')}, completion: {usage.get('completion_tokens')})")
            return generated_answer

        except requests.exceptions.Timeout:
            logger.error("OpenAI request timed out")
            raise Exception("LLM 생성 시간 초과")
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to OpenAI API")
            raise Exception("OpenAI API에 연결할 수 없습니다. 인터넷 연결을 확인하세요.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error during generation: {e}")
            try:
                error_detail = response.json()
                logger.error(f"Error details: {error_detail}")
                error_msg = error_detail.get('error', {}).get('message', str(e))
                raise Exception(f"답변 생성 중 오류 발생: {error_msg}")
            except:
                raise Exception(f"답변 생성 중 오류 발생: {str(e)}")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise Exception(f"답변 생성 중 오류 발생: {str(e)}")

    def check_health(self) -> bool:
        """OpenAI API 상태 확인"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=5
            )
            response.raise_for_status()

            models = response.json().get("data", [])
            return [model["id"] for model in models if "gpt" in model["id"]]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
