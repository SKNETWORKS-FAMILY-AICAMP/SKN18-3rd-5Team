#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
임베딩 모델 래퍼 클래스
다양한 HuggingFace 모델을 통합 관리
"""

import logging
import os
from typing import Optional, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv

from .config import EmbeddingModelType, ModelConfig, get_model_config

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """임베딩 모델 래퍼 클래스"""

    def __init__(
        self,
        model_type: EmbeddingModelType,
        device: Optional[str] = None
    ):
        """
        Args:
            model_type: 사용할 모델 타입
            device: 'cuda', 'cpu', 또는 None (자동 선택)
        """
        self.model_type = model_type
        self.config = get_model_config(model_type)

        # Device 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.config.device = self.device

        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        logger.info(f"Initialized {self.config.display_name} on {self.device}")

    def load(self):
        """모델 로드"""
        if self._is_loaded:
            logger.info(f"{self.config.display_name} already loaded")
            return

        try:
            logger.info(f"Loading {self.config.display_name}...")

            # HuggingFace 토큰 가져오기
            hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_API_TOKEN') or os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
            if hf_token:
                logger.info("Using HuggingFace token from environment")

            # FinE5 모델의 경우 LoRA 가중치 로드 시도
            if self.model_type == EmbeddingModelType.FINE5_FINANCE:
                try:
                    logger.info("FinE5 LoRA 모델 로드 시도...")
                    self._load_fine5_lora(hf_token)
                    return
                except Exception as e:
                    logger.warning(f"FinE5 LoRA 로드 실패: {e}")
                    logger.info("일반 FinE5 모델 로드 시도...")

            # Sentence Transformers로 로드 시도
            try:
                model_kwargs = {
                    "device": self.device,
                    "trust_remote_code": self.config.trust_remote_code,
                    "token": hf_token
                }
                
                self.model = SentenceTransformer(
                    self.config.model_name,
                    **model_kwargs
                )
                logger.info(f"Loaded via SentenceTransformer: {self.config.model_name}")
                self._is_loaded = True
                return

            except Exception as e:
                logger.warning(f"SentenceTransformer load failed: {e}")
                logger.info("Trying HuggingFace Transformers...")

                # HuggingFace Transformers로 로드
                import torch
                
                # 모델 설정
                model_kwargs = {
                    "trust_remote_code": self.config.trust_remote_code,
                    "token": hf_token
                }
                
                # dtype 설정 (torch_dtype 대신 dtype 사용)
                if self.config.extra_params.get("torch_dtype"):
                    dtype_str = self.config.extra_params["torch_dtype"]
                    if dtype_str == "float16":
                        model_kwargs["dtype"] = torch.float16
                    elif dtype_str == "bfloat16":
                        model_kwargs["dtype"] = torch.bfloat16
                    elif dtype_str == "float32":
                        model_kwargs["dtype"] = torch.float32
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=self.config.trust_remote_code,
                    token=hf_token
                )
                self.model = AutoModel.from_pretrained(
                    self.config.model_name,
                    **model_kwargs
                ).to(self.device)
                self.model.eval()

                logger.info(f"Loaded via Transformers: {self.config.model_name}")
                self._is_loaded = True

        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise

    def _load_fine5_lora(self, hf_token: Optional[str] = None):
        """FinE5 LoRA 모델 로드"""
        try:
            # PEFT 라이브러리 import
            from peft import PeftModel, PeftConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # LoRA 모델 경로 설정 (로컬 경로 또는 HuggingFace Hub)
            lora_model_path = "FinanceMTEB/FinE5"  # 또는 로컬 경로
            
            logger.info(f"FinE5 LoRA 모델 로드 중: {lora_model_path}")
            
            # LoRA 설정 로드
            peft_config = PeftConfig.from_pretrained(lora_model_path, token=hf_token)
            
            # 베이스 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                trust_remote_code=True,
                token=hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # LoRA 가중치 로드
            self.model = PeftModel.from_pretrained(base_model, lora_model_path, token=hf_token)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                peft_config.base_model_name_or_path,
                trust_remote_code=True,
                token=hf_token
            )
            
            logger.info(f"FinE5 LoRA 모델 로드 완료: {lora_model_path}")
            self._is_loaded = True
            
        except ImportError:
            logger.error("PEFT 라이브러리가 설치되지 않았습니다. 'pip install peft'로 설치해주세요.")
            raise
        except Exception as e:
            logger.error(f"FinE5 LoRA 모델 로드 실패: {e}")
            raise

    def unload(self):
        """모델 언로드 (메모리 절약)"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False
        logger.info(f"Unloaded {self.config.display_name}")

    def is_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self._is_loaded

    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.config.dimension

    def get_model_name(self) -> str:
        """모델 이름 반환"""
        return self.config.model_name

    def get_display_name(self) -> str:
        """표시 이름 반환"""
        return self.config.display_name

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"EmbeddingModel({self.config.display_name}, {status}, {self.device})"


class ModelFactory:
    """임베딩 모델 팩토리"""

    _instances: Dict[EmbeddingModelType, EmbeddingModel] = {}

    @classmethod
    def create_model(
        cls,
        model_type: EmbeddingModelType,
        device: Optional[str] = None,
        auto_load: bool = True
    ) -> EmbeddingModel:
        """
        모델 생성 (싱글톤 패턴)

        Args:
            model_type: 모델 타입
            device: 디바이스
            auto_load: 자동 로드 여부

        Returns:
            EmbeddingModel 인스턴스
        """
        # 기존 인스턴스가 있으면 재사용
        if model_type in cls._instances:
            logger.info(f"Reusing existing instance: {model_type.value}")
            return cls._instances[model_type]

        # 새 인스턴스 생성
        model = EmbeddingModel(model_type, device)

        if auto_load:
            model.load()

        cls._instances[model_type] = model
        return model

    @classmethod
    def get_model(cls, model_type: EmbeddingModelType) -> Optional[EmbeddingModel]:
        """기존 모델 인스턴스 반환"""
        return cls._instances.get(model_type)

    @classmethod
    def unload_all(cls):
        """모든 모델 언로드"""
        for model in cls._instances.values():
            model.unload()
        cls._instances.clear()
        logger.info("All models unloaded")

    @classmethod
    def get_loaded_models(cls) -> list[str]:
        """로드된 모델 목록 반환"""
        return [
            model.get_display_name()
            for model in cls._instances.values()
            if model.is_loaded()
        ]


# ============================================================================
#  유틸리티 함수
# ============================================================================

def create_all_models(device: Optional[str] = None, auto_load: bool = False) -> Dict[EmbeddingModelType, EmbeddingModel]:
    """모든 모델 생성 (배치용)"""
    models = {}
    for model_type in EmbeddingModelType:
        try:
            model = ModelFactory.create_model(model_type, device, auto_load=auto_load)
            models[model_type] = model
            logger.info(f"Created: {model.get_display_name()}")
        except Exception as e:
            logger.error(f"Failed to create {model_type.value}: {e}")

    return models


def get_model_info(model_type: EmbeddingModelType) -> Dict[str, Any]:
    """모델 정보 반환"""
    config = get_model_config(model_type)
    model = ModelFactory.get_model(model_type)

    return {
        "model_name": config.model_name,
        "display_name": config.display_name,
        "dimension": config.dimension,
        "max_seq_length": config.max_seq_length,
        "batch_size": config.batch_size,
        "is_loaded": model.is_loaded() if model else False,
        "device": model.device if model else None
    }


if __name__ == "__main__":
    # 테스트: 모든 모델 생성 (메모리 절약)
    print("=== Creating all embedding models ===\n")

    models = create_all_models(auto_load=False)

    print(f"\nCreated {len(models)} models:")
    for model_type, model in models.items():
        print(f"  - {model}")

    # 테스트: 개별 모델 로드
    print("\n\n=== Testing individual model loading ===\n")

    test_model_type = EmbeddingModelType.MULTILINGUAL_E5_SMALL
    test_model = ModelFactory.create_model(test_model_type, auto_load=True)

    print(f"Loaded: {test_model}")
    print(f"Dimension: {test_model.get_dimension()}")
    print(f"Is loaded: {test_model.is_loaded()}")

    # 언로드
    test_model.unload()
    print(f"After unload: {test_model}")