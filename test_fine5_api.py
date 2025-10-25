#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FinE5 API 테스트 스크립트
"""

import os
import requests
import json
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_detailed_instruct(task_description: str, query: str) -> str:
    """FinE5 모델을 위한 instruction 포맷팅"""
    return f'Instruct: {task_description}\nQuery: {query}'

def test_fine5_api():
    """FinE5 API 테스트"""
    try:
        # API 키 가져오기
        api_key = os.getenv('FIN_E5_API_KEY')
        if not api_key:
            print("❌ FIN_E5_API_KEY 환경변수가 설정되지 않았습니다.")
            return False
        
        print(f"🔑 API 키 확인: {api_key[:10]}...")
        
        # API 설정
        base_url = "https://abacinlp.com/v1"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 테스트 쿼리
        query = "삼성전자 주식 가격 동향"
        formatted_query = get_detailed_instruct(
            "Given a financial question, retrieve user replies that best answer the question.",
            query
        )
        
        print(f"📝 테스트 쿼리: {query}")
        print(f"📝 포맷된 쿼리: {formatted_query}")
        
        # API 호출
        payload = {
            "model": "FinE5",
            "input": [formatted_query],
            "encoding_format": "float"
        }
        
        print("🚀 API 호출 중...")
        response = requests.post(
            f"{base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 응답 상태 코드: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ API 호출 실패: {response.text}")
            return False
        
        response_data = response.json()
        print(f"✅ API 호출 성공!")
        print(f"📊 응답 데이터: {json.dumps(response_data, indent=2)}")
        
        # 임베딩 정보
        if "data" in response_data and len(response_data["data"]) > 0:
            embedding = response_data["data"][0]["embedding"]
            print(f"🎯 임베딩 차원: {len(embedding)}")
            print(f"🎯 임베딩 샘플: {embedding[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("🧪 FinE5 API 테스트 시작")
    print("=" * 50)
    
    success = test_fine5_api()
    
    if success:
        print("\n🎉 FinE5 API 테스트 성공!")
    else:
        print("\n💥 FinE5 API 테스트 실패!")
