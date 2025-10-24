"""
Generation module for RAG system
LLM 기반 답변 생성 모듈
"""

from .generator import LLMGenerator, OllamaGenerator

__all__ = ["LLMGenerator", "OllamaGenerator"]
