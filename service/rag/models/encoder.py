#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
임베딩 인코더
텍스트를 벡터로 변환 (데이터 & 쿼리 모두 지원)
모든 모델의 특수 처리 포함
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from .config import EmbeddingModelType, ModelConfig, get_model_config
from .loader import ModelFactory, EmbeddingModel

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """임베딩 인코더 (데이터 & 쿼리 임베딩)"""

    def __init__(
        self,
        model_type: EmbeddingModelType = EmbeddingModelType.MULTILINGUAL_E5_SMALL,
        device: Optional[str] = None
    ):
        """
        Args:
            model_type: 사용할 모델 타입
            device: 디바이스 ('cuda', 'cpu', None)
        """
        self.model_type = model_type
        self.config: ModelConfig = get_model_config(model_type)

        # 모델 래퍼 생성 (지연 로딩)
        self.model_wrapper: EmbeddingModel = ModelFactory.create_model(
            model_type, device, auto_load=True
        )

        self.device = self.model_wrapper.device
        self.model = self.model_wrapper.model
        self.tokenizer = self.model_wrapper.tokenizer

        logger.info(
            f"EmbeddingEncoder initialized: {self.config.display_name} "
            f"(dim={self.config.dimension}, device={self.device})"
        )

    def _prepare_text_for_model(self, text: str, is_query: bool = False) -> str:
        """
        모델별 텍스트 전처리 (prefix, instruction 등)

        Args:
            text: 원본 텍스트
            is_query: 쿼리 여부

        Returns:
            전처리된 텍스트
        """
        extra_params = self.config.extra_params

        # 1. E5 모델 - prefix 필수
        if self.model_type == EmbeddingModelType.MULTILINGUAL_E5_SMALL:
            if is_query:
                return extra_params.get("query_prefix", "") + text
            else:
                return extra_params.get("passage_prefix", "") + text

        # 2. FinE5 모델 - E5 계열이므로 prefix 필수
        elif self.model_type == EmbeddingModelType.FINE5_FINANCE:
            if is_query:
                return extra_params.get("query_prefix", "") + text
            else:
                return extra_params.get("passage_prefix", "") + text

        # 3. 기타 모델 (KakaoBank DeBERTa 등)
        else:
            return text

    def encode_query(self, query: str) -> List[float]:
        """
        쿼리 인코딩 (단일)

        Args:
            query: 검색 쿼리

        Returns:
            쿼리 벡터
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return [0.0] * self.config.dimension

        try:
            # SentenceTransformer 사용
            if isinstance(self.model, SentenceTransformer):
                return self._encode_with_sentence_transformer(
                    [query], is_query=True
                )[0]
            # Transformers 사용
            else:
                return self._encode_with_transformers(
                    [query], is_query=True
                )[0]

        except Exception as e:
            logger.error(f"Query encoding failed: {e}", exc_info=True)
            return [0.0] * self.config.dimension

    def encode_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        문서 인코딩 (배치)

        Args:
            texts: 문서 텍스트 리스트
            batch_size: 배치 크기 (None이면 config 사용)
            show_progress: 진행률 표시 여부

        Returns:
            벡터 리스트
        """
        if not texts:
            logger.warning("Empty text list provided")
            return []

        # Batch size 설정
        if batch_size is None:
            batch_size = self.config.batch_size

        try:
            # SentenceTransformer 사용
            if isinstance(self.model, SentenceTransformer):
                return self._encode_with_sentence_transformer(
                    texts, 
                    is_query=False,
                    batch_size=batch_size,
                    show_progress=show_progress
                )
            # Transformers 사용
            else:
                return self._encode_with_transformers(
                    texts,
                    is_query=False,
                    batch_size=batch_size,
                    show_progress=show_progress
                )

        except Exception as e:
            logger.error(f"Document encoding failed: {e}", exc_info=True)
            return [[0.0] * self.config.dimension] * len(texts)

    def _encode_with_sentence_transformer(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        SentenceTransformer로 인코딩

        Args:
            texts: 텍스트 리스트
            is_query: 쿼리 여부
            batch_size: 배치 크기
            show_progress: 진행률 표시

        Returns:
            벡터 리스트
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        # 텍스트 전처리
        processed_texts = [
            self._prepare_text_for_model(text, is_query) 
            for text in texts
        ]

        # 모델별 특수 처리
        encode_kwargs = {
            "convert_to_tensor": False,
            "normalize_embeddings": self.config.normalize_embeddings,
            "batch_size": batch_size,
            "show_progress_bar": show_progress
        }

        # FinE5는 E5 계열이므로 특별한 처리 없음 (prefix만 사용)

        embeddings = self.model.encode(processed_texts, **encode_kwargs)

        # numpy array를 list로 변환
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return embeddings

    def _encode_with_transformers(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        HuggingFace Transformers로 인코딩 수행

        Args:
            texts: 텍스트 리스트
            is_query: 쿼리 여부
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부

        Returns:
            벡터 리스트
        """
        # 텍스트 전처리
        processed_texts = [
            self._prepare_text_for_model(text, is_query) 
            for text in texts
        ]

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(processed_texts), batch_size):
                batch_texts = processed_texts[i:i + batch_size]

                # Tokenizer 설정 (모델별)
                tokenize_kwargs = {
                    "padding": True,
                    "truncation": True,
                    "max_length": self.config.max_seq_length,
                    "return_tensors": "pt"
                }

                # FinE5는 일반적인 padding 사용

                # Tokenize
                inputs = self.tokenizer(batch_texts, **tokenize_kwargs).to(self.device)

                # Model dtype 설정
                dtype = self._get_model_dtype()
                if dtype and self.device == "cuda":
                    self.model = self.model.to(dtype)

                # Forward pass
                outputs = self.model(**inputs)

                # Pooling
                embeddings = self._pool_embeddings(
                    outputs,
                    inputs["attention_mask"]
                )

                # Normalize
                if self.config.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

                if show_progress:
                    print(f"Processed {min(i + batch_size, len(processed_texts))}/{len(processed_texts)}")

        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings.tolist()

    def _pool_embeddings(
        self,
        model_output,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pooling 수행 (모델별 처리)

        Args:
            model_output: 모델 출력
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        # last_hidden_state 추출
        if hasattr(model_output, "last_hidden_state"):
            hidden_states = model_output.last_hidden_state
        else:
            hidden_states = model_output[0]

        pooling_mode = self.config.pooling_mode

        # 1. Mean pooling
        if pooling_mode == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                hidden_states.size()
            ).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        # 2. Last token pooling
        elif pooling_mode == "last_token":
            # 마지막 유효 토큰 사용
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            return hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
            ]

        # 3. CLS token (첫 번째 토큰)
        elif pooling_mode == "cls":
            return hidden_states[:, 0, :]

        # 4. Max pooling
        elif pooling_mode == "max":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                hidden_states.size()
            ).float()
            hidden_states[input_mask_expanded == 0] = -1e9
            return torch.max(hidden_states, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")

    def _get_model_dtype(self) -> Optional[torch.dtype]:
        """모델별 적절한 dtype 반환"""
        extra_params = self.config.extra_params
        dtype_str = extra_params.get("torch_dtype")

        if dtype_str == "float16":
            return torch.float16
        elif dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "float32":
            return torch.float32
        else:
            return None

    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.config.dimension

    def get_model_name(self) -> str:
        """모델 이름 반환"""
        return self.config.model_name

    def get_display_name(self) -> str:
        """표시 이름 반환"""
        return self.config.display_name

    def get_model_type(self) -> EmbeddingModelType:
        """모델 타입 반환"""
        return self.model_type



# ============================================================================
# 다중 모델 인코더 (배치 처리)
# ============================================================================

class MultiModelEncoder:
    """여러 모델로 동시에 인코딩 수행 (배치용)"""

    def __init__(
        self,
        model_types: List[EmbeddingModelType],
        device: Optional[str] = None
    ):
        """
        Args:
            model_types: 사용할 모델 타입 리스트
            device: 디바이스
        """
        self.encoders = {}

        for model_type in model_types:
            try:
                encoder = EmbeddingEncoder(model_type, device)
                self.encoders[model_type] = encoder
                logger.info(f"Loaded encoder: {encoder.get_display_name()}")
            except Exception as e:
                logger.error(f"Failed to load {model_type.value}: {e}")

    def encode_query_all(self, query: str) -> dict:
        """모든 모델로 쿼리 인코딩"""
        results = {}
        for model_type, encoder in self.encoders.items():
            try:
                embedding = encoder.encode_query(query)
                results[model_type.value] = {
                    "embedding": embedding,
                    "dimension": len(embedding),
                    "model_name": encoder.get_display_name()
                }
            except Exception as e:
                logger.error(f"Query encoding failed for {model_type.value}: {e}")

        return results

    def encode_documents_all(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> dict:
        """모든 모델로 문서 인코딩"""
        results = {}
        for model_type, encoder in self.encoders.items():
            try:
                logger.info(f"Encoding with {encoder.get_display_name()}...")
                embeddings = encoder.encode_documents(texts, show_progress=show_progress)
                results[model_type.value] = {
                    "embeddings": embeddings,
                    "dimension": encoder.get_dimension(),
                    "model_name": encoder.get_display_name(),
                    "count": len(embeddings)
                }
            except Exception as e:
                logger.error(f"Document encoding failed for {model_type.value}: {e}")

        return results

    def get_loaded_models(self) -> List[str]:
        """로드된 모델 목록"""
        return [encoder.get_display_name() for encoder in self.encoders.values()]


if __name__ == "__main__":
    # 테스트는 comparator에서 수행
    print("EmbeddingEncoder 모듈 - 테스트는 comparator에서 수행하세요")
