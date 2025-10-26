#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
문서 포맷터 모듈
다양한 형태로 문서를 포맷팅
"""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseFormatter(ABC):
    """기본 포맷터 클래스"""
    
    def __init__(self, name: str = "BaseFormatter"):
        self.name = name
        logger.info(f"Initialized formatter: {self.name}")
    
    @abstractmethod
    def format_documents(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """문서들을 포맷팅"""
        pass


class PromptFormatter(BaseFormatter):
    """프롬프트 포맷터"""
    
    def __init__(self, system_prompt: Optional[str] = None):
        super().__init__("PromptFormatter")
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def format_documents(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """프롬프트 형태로 포맷팅"""
        if not documents:
            return f"{self.system_prompt}\n\n사용자 질문: {query}\n\n관련 문서를 찾을 수 없습니다."
        
        context = f"{self.system_prompt}\n\n"
        context += f"사용자 질문: {query}\n\n"
        context += "참고 문서들:\n"
        
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            similarity = doc.get('similarity', 0)
            context += f"[문서 {i}] (관련도: {similarity:.3f})\n{content}\n\n"
        
        context += "위 문서들을 참고하여 질문에 답변해주세요."
        return context
    
    def _default_system_prompt(self) -> str:
        """기본 시스템 프롬프트"""
        return """당신은 도움이 되는 AI 어시스턴트입니다. 주어진 문서들을 참고하여 사용자의 질문에 정확하고 상세하게 답변해주세요. 답변할 때는 다음 사항을 고려해주세요:

1. 문서의 내용을 정확히 인용하고 출처를 명시하세요
2. 직접적인 수치나 명시가 없을 때는 그 사실을 밝히되, 문서에서 파생되는 합리적인 시사점·전망·리스크를 간단히 정리하세요
3. 여러 문서에서 상충되는 정보가 있다면 이를 명시하세요
4. 답변은 한국어로 작성하세요"""


class MarkdownFormatter(BaseFormatter):
    """마크다운 포맷터"""
    
    def __init__(self):
        super().__init__("MarkdownFormatter")
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        import re
        
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 마침표 앞 공백 제거
        text = re.sub(r'\s+\.', '.', text)
        
        # 쉼표 앞 공백 제거
        text = re.sub(r'\s+,', ',', text)
        
        # 문장 끝 정리
        text = re.sub(r'\s+([.!?])', r'\1', text)
        
        # 단어 중간 공백 제거 (예: "가 능합니다" -> "가능합니다")
        text = re.sub(r'(\w)\s+(\w)', r'\1\2', text)
        
        return text.strip()
    
    def format_documents(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """마크다운 형태로 포맷팅"""
        if not documents:
            return f"# 질문\n{query}\n\n## 관련 문서\n관련 문서를 찾을 수 없습니다."
        
        context = f"# 질문\n{query}\n\n## 관련 문서\n"
        
        for i, doc in enumerate(documents, 1):
            content = self._clean_text(doc.get('content', ''))
            similarity = doc.get('similarity', 0)
            metadata = doc.get('metadata', {})
            
            context += f"### 문서 {i} (관련도: {similarity:.3f})\n"
            
            if metadata.get('source'):
                context += f"**출처:** {metadata['source']}\n\n"
            
            # 문장 단위로 정리
            sentences = content.split('.')
            cleaned_sentences = []
            for sentence in sentences:
                cleaned = self._clean_text(sentence)
                if cleaned:
                    cleaned_sentences.append(cleaned + '.')
            
            context += ' '.join(cleaned_sentences) + "\n\n"
        
        return context


class JSONFormatter(BaseFormatter):
    """JSON 포맷터"""
    
    def __init__(self):
        super().__init__("JSONFormatter")
    
    def format_documents(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """JSON 형태로 포맷팅"""
        import json
        
        formatted_data = {
            "query": query,
            "documents": []
        }
        
        for i, doc in enumerate(documents, 1):
            formatted_doc = {
                "id": i,
                "content": doc.get('content', ''),
                "similarity": doc.get('similarity', 0),
                "metadata": doc.get('metadata', {})
            }
            formatted_data["documents"].append(formatted_doc)
        
        return json.dumps(formatted_data, ensure_ascii=False, indent=2)


class ConversationalFormatter(BaseFormatter):
    """대화형 포맷터"""
    
    def __init__(self):
        super().__init__("ConversationalFormatter")
    
    def format_documents(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """대화형 형태로 포맷팅"""
        if not documents:
            return f"질문: {query}\n\n죄송하지만 관련 문서를 찾을 수 없습니다."
        
        context = f"질문: {query}\n\n"
        context += "관련 정보를 찾았습니다:\n\n"
        
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            similarity = doc.get('similarity', 0)
            
            context += f"정보 {i} (신뢰도: {similarity:.1%}):\n"
            context += f"{content}\n\n"
        
        context += "이 정보들을 바탕으로 답변드리겠습니다."
        return context


class StructuredFormatter(BaseFormatter):
    """구조화된 포맷터"""

    def __init__(self, include_scores: bool = True, include_metadata: bool = True):
        super().__init__("StructuredFormatter")
        self.include_scores = include_scores
        self.include_metadata = include_metadata

    def format_documents(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """구조화된 형태로 포맷팅"""
        if not documents:
            return f"QUERY: {query}\n\nDOCUMENTS: None found"

        context = f"QUERY: {query}\n\n"
        context += "DOCUMENTS:\n"
        context += "=" * 50 + "\n"

        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            similarity = doc.get('similarity', 0)
            metadata = doc.get('metadata', {})

            context += f"DOCUMENT {i}:\n"
            context += f"Content: {content}\n"

            if self.include_scores:
                context += f"Relevance Score: {similarity:.4f}\n"

            if self.include_metadata and metadata:
                context += f"Metadata: {metadata}\n"

            context += "-" * 30 + "\n"

        return context


class PolicyFormatter(BaseFormatter):
    """청년 주거 정책 전용 포맷터 - 문서를 정책 관련 정보로 구조화"""

    def __init__(self):
        super().__init__("PolicyFormatter")

    def _clean_text(self, text: str) -> str:
        """텍스트 정리 - 불필요한 공백 제거"""
        import re
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        # 특수문자 앞 공백 제거
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        return text.strip()

    def _extract_key_info(self, content: str) -> Dict[str, List[str]]:
        """정책 관련 핵심 정보 추출"""
        import re

        key_info = {
            "조건": [],
            "금리": [],
            "한도": [],
            "신청방법": [],
            "기타": []
        }

        # 금리 관련 정보
        interest_patterns = [
            r'연\s*\d+\.?\d*\s*%',
            r'\d+\.?\d*\s*%',
            r'금리.*?\d+\.?\d*'
        ]
        for pattern in interest_patterns:
            matches = re.findall(pattern, content)
            key_info["금리"].extend(matches)

        # 금액/한도 관련 정보
        amount_patterns = [
            r'\d+억\s*원?',
            r'\d+천만\s*원?',
            r'\d+백만\s*원?',
            r'\d+만\s*원?',
            r'\d+%\s*이내'
        ]
        for pattern in amount_patterns:
            matches = re.findall(pattern, content)
            key_info["한도"].extend(matches)

        # 조건 관련 키워드
        condition_keywords = ['자격', '조건', '대상', '요건', '기준']
        for keyword in condition_keywords:
            if keyword in content:
                # 키워드 주변 문장 추출
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        key_info["조건"].append(self._clean_text(sentence))

        # 신청 방법 관련
        application_keywords = ['신청', '방법', '절차', '접수']
        for keyword in application_keywords:
            if keyword in content:
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        key_info["신청방법"].append(self._clean_text(sentence))

        return key_info

    def format_documents(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """정책 정보에 최적화된 포맷팅"""
        if not documents:
            return f"질문: {query}\n\n관련 정책 정보를 찾을 수 없습니다."

        # 헤더
        context = f"# 질문\n{query}\n\n"
        context += "# 관련 정책 정보\n\n"

        # 문서별 정보
        all_key_info = {
            "조건": [],
            "금리": [],
            "한도": [],
            "신청방법": [],
        }

        for i, doc in enumerate(documents, 1):
            content = self._clean_text(doc.get('content', ''))
            similarity = doc.get('similarity', 0)
            metadata = doc.get('metadata', {})
            source = metadata.get('source', '출처 미상')

            context += f"## 참고 문서 {i} (관련도: {similarity:.1%})\n"
            context += f"**출처:** {source}\n\n"

            # 핵심 정보 추출
            key_info = self._extract_key_info(content)

            # 추출된 정보를 전체 정보에 추가
            for category, items in key_info.items():
                all_key_info[category].extend(items)

            # 원문 내용 (간결하게)
            context += f"**내용:**\n{content}\n\n"
            context += "---\n\n"

        # 핵심 정보 요약 (선택사항)
        if any(all_key_info.values()):
            context += "# 핵심 정보 요약\n\n"

            if all_key_info["금리"]:
                unique_rates = list(set(all_key_info["금리"]))
                context += f"**금리:** {', '.join(unique_rates[:5])}\n\n"

            if all_key_info["한도"]:
                unique_limits = list(set(all_key_info["한도"]))
                context += f"**대출한도:** {', '.join(unique_limits[:5])}\n\n"

        return context


class EnhancedPromptFormatter(BaseFormatter):
    """개선된 프롬프트 포맷터 - 더 나은 답변을 위한 구조화"""

    def __init__(self):
        super().__init__("EnhancedPromptFormatter")

    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        return text.strip()

    def format_documents(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """개선된 형태로 포맷팅"""
        if not documents:
            return f"질문: {query}\n\n관련 문서를 찾을 수 없습니다."

        context = f"질문: {query}\n\n"
        context += "다음 문서들을 참고하여 질문에 답변해주세요:\n\n"

        for i, doc in enumerate(documents, 1):
            content = self._clean_text(doc.get('content', ''))
            similarity = doc.get('similarity', 0)
            metadata = doc.get('metadata', {})
            source = metadata.get('source', '출처 미상')

            # 관련도가 높은 문서 강조
            if similarity > 0.85:
                context += f"[🔥 매우 관련성 높음] "
            elif similarity > 0.75:
                context += f"[✓ 관련성 높음] "

            context += f"문서 {i} (출처: {source})\n"
            context += f"{content}\n\n"

        context += "\n답변 시 주의사항:\n"
        context += "- 문서의 정보만을 사용하여 정확하게 답변하세요\n"
        context += "- 조건, 금리, 한도 등 구체적인 수치를 명확히 제시하세요\n"
        context += "- 체계적으로 구조화하여 답변하세요\n"
        context += "- 불완전한 정보는 '문서에 명시되지 않음'이라고 표시하세요\n"

        return context
