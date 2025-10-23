from pgvector.psycopg2 import register_vector

# ==========================================
# 1. Parquet -> 임베딩
# ==========================================


# ==========================================
# 1. 데이터베이스 연결 및 pgvector 확장 활성화
# ==========================================
def setup_pgvector_extension(connection_string):
    """PostgreSQL에 pgvector 확장을 설치하고 활성화"""
    try:
        # 데이터베이스 연결
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # pgvector 확장 설치 (이미 설치되어 있다면 무시됨)
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # 벡터 타입 등록
        register_vector(conn)
        
        print("pgvector 확장이 성공적으로 설치되었습니다.")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"pgvector 설정 중 오류 발생: {e}")
        print("PostgreSQL이 실행 중이고 연결 정보가 올바른지 확인하세요.")

# ==========================================
# 1. 임베딩 모델 설정
# ==========================================

# E5

# kakaobank

# Fin-E5


# ==========================================
# 1. 임베딩 모델 설정
# ==========================================

# E5

# kakaobank

# Fin-E5

