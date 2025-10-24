#!/usr/bin/env python3
"""
RAG 데이터 변환 파이프라인

전체 파이프라인을 순차적으로 실행:
1. Parser: 마크다운 → 구조화된 청크
2. Normalizer: 텍스트 정규화 및 단위 변환
3. Chunker: 스마트 청킹 및 메타데이터 추가

사용법:
    python pipeline.py           # 테스트 모드 (20개 파일)
    python pipeline.py --all     # 전체 파일 처리
    python pipeline.py --help    # 도움말
"""

import sys
import time
import shutil
from pathlib import Path
from typing import Optional

# 공통 유틸리티 import
from utils import get_project_paths, get_transform_paths

# 현재 스크립트의 디렉토리
SCRIPT_DIR = Path(__file__).parent  # service/etl/transform

def cleanup_parser_files():
    """Parser 파일 정리 (Normalizer 완료 후)"""
    print("\n" + "=" * 80)
    print("🧹 Parser 파일 정리")
    print("=" * 80)
    
    paths = get_transform_paths(__file__)
    parser_dir = paths['parser_dir']
    
    if parser_dir.exists():
        # 파일 개수 확인
        parser_files = list(parser_dir.glob("*.jsonl"))
        file_count = len(parser_files)
        
        if file_count > 0:
            # 디렉토리 전체 삭제
            shutil.rmtree(parser_dir)
            print(f"✅ Parser 파일 {file_count}개 삭제 완료")
            print(f"   삭제된 디렉토리: {parser_dir}")
        else:
            print("ℹ️  삭제할 Parser 파일 없음")
    else:
        print("ℹ️  Parser 디렉토리 없음")

def cleanup_normalized_files():
    """Normalized 파일 정리 (Chunker 완료 후)"""
    print("\n" + "=" * 80)
    print("🧹 Normalized 파일 정리")
    print("=" * 80)
    
    paths = get_transform_paths(__file__)
    normalized_dir = paths['normalized_dir']
    
    if normalized_dir.exists():
        # 파일 개수 확인
        normalized_files = list(normalized_dir.glob("*.jsonl"))
        file_count = len(normalized_files)
        
        if file_count > 0:
            # 디렉토리 전체 삭제
            shutil.rmtree(normalized_dir)
            print(f"✅ Normalized 파일 {file_count}개 삭제 완료")
            print(f"   삭제된 디렉토리: {normalized_dir}")
        else:
            print("ℹ️  삭제할 Normalized 파일 없음")
    else:
        print("ℹ️  Normalized 디렉토리 없음")

def run_parser(process_all: bool = False) -> bool:
    """Step 1: Parser 실행"""
    print("=" * 80)
    print("🔧 Step 1: Parser 실행")
    print("=" * 80)
    
    try:
        # parser.py 모듈 import 및 실행
        import importlib.util
        parser_path = SCRIPT_DIR / "parser.py"
        spec = importlib.util.spec_from_file_location("parser", parser_path)
        parser_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser_module)
        parser_main = parser_module.main
        
        parser_main(process_all=process_all)
        print("✅ Parser 완료")
        return True
        
    except Exception as e:
        print(f"❌ Parser 실패: {e}")
        return False

def run_normalizer() -> bool:
    """Step 2: Normalizer 실행"""
    print("\n" + "=" * 80)
    print("🔧 Step 2: Normalizer 실행")
    print("=" * 80)
    
    try:
        # normalizer.py 모듈 import 및 실행
        import importlib.util
        normalizer_path = SCRIPT_DIR / "normalizer.py"
        spec = importlib.util.spec_from_file_location("normalizer", normalizer_path)
        normalizer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(normalizer_module)
        normalizer_main = normalizer_module.main
        
        # sys.argv를 임시로 수정하여 디렉토리 모드로 실행
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]  # 스크립트 이름만 남김
        
        try:
            normalizer_main()
            print("✅ Normalizer 완료")
        finally:
            # sys.argv 복원
            sys.argv = original_argv
        
        # Parser 파일 정리
        cleanup_parser_files()
        
        return True
        
    except Exception as e:
        print(f"❌ Normalizer 실패: {e}")
        return False

def run_chunker() -> bool:
    """Step 3: Chunker 실행"""
    print("\n" + "=" * 80)
    print("🔧 Step 3: Chunker 실행")
    print("=" * 80)
    
    try:
        # chunker.py 모듈 import 및 실행
        import importlib.util
        chunker_path = SCRIPT_DIR / "chunker.py"
        spec = importlib.util.spec_from_file_location("chunker", chunker_path)
        chunker_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chunker_module)
        chunker_main = chunker_module.main
        
        # sys.argv를 임시로 수정하여 디렉토리 모드로 실행
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]  # 스크립트 이름만 남김
        
        try:
            chunker_main()
            print("✅ Chunker 완료")
        finally:
            # sys.argv 복원
            sys.argv = original_argv
        
        # Normalized 파일 정리
        cleanup_normalized_files()
        
        return True
        
    except Exception as e:
        print(f"❌ Chunker 실패: {e}")
        return False

def check_input_files(process_all: bool = False) -> bool:
    """입력 파일 확인"""
    print("=" * 80)
    print("📁 입력 파일 확인")
    print("=" * 80)
    
    # 경로 설정
    paths = get_transform_paths(__file__)
    markdown_dir = paths['markdown_dir']
    
    if not markdown_dir.exists():
        print(f"❌ 마크다운 디렉토리 없음: {markdown_dir}")
        return False
    
    # 마크다운 파일 목록
    md_files = list(markdown_dir.glob("*.md"))
    
    if not md_files:
        print(f"❌ 마크다운 파일 없음: {markdown_dir}")
        return False
    
    print(f"✅ 마크다운 파일: {len(md_files)}개")
    
    if process_all:
        print(f"   → 전체 파일 처리 모드")
    else:
        print(f"   → 테스트 모드 (처음 20개만)")
        md_files = md_files[:20]
    
    print(f"   → 처리할 파일: {len(md_files)}개")
    
    # 샘플 파일명 출력
    if md_files:
        print(f"   → 샘플: {md_files[0].name}")
        if len(md_files) > 1:
            print(f"   → 샘플: {md_files[1].name}")
    
    return True

def check_output_directories():
    """출력 디렉토리 확인 및 생성"""
    print("\n" + "=" * 80)
    print("📁 출력 디렉토리 확인")
    print("=" * 80)
    
    paths = get_transform_paths(__file__)
    output_dirs = [
        paths['parser_dir'],
        paths['normalized_dir'], 
        paths['final_dir']
    ]
    
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ {output_dir.name}: {output_dir}")
    
    return True

def get_pipeline_stats():
    """파이프라인 결과 통계"""
    print("\n" + "=" * 80)
    print("📊 파이프라인 결과 통계")
    print("=" * 80)
    
    paths = get_transform_paths(__file__)
    
    # 각 단계별 파일 수 확인
    stages = {
        "Parser": paths['parser_dir'],
        "Normalized": paths['normalized_dir'],
        "Final": paths['final_dir']
    }
    
    for stage_name, stage_dir in stages.items():
        if stage_dir.exists():
            files = list(stage_dir.glob("*.jsonl"))
            print(f"  {stage_name:12s}: {len(files):4,}개 파일")
        else:
            if stage_name in ["Parser", "Normalized"]:
                print(f"  {stage_name:12s}: 정리됨 (삭제)")
            else:
                print(f"  {stage_name:12s}: 디렉토리 없음")
    
    # Final 파일의 총 청크 수
    final_dir = paths['final_dir']
    if final_dir.exists():
        total_chunks = 0
        for jsonl_file in final_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                total_chunks += sum(1 for _ in f)
        
        print(f"\n  총 청크 수: {total_chunks:,}개")
        
        # 디스크 사용량 계산
        total_size = sum(f.stat().st_size for f in final_dir.glob("*.jsonl"))
        size_mb = total_size / (1024 * 1024)
        print(f"  총 파일 크기: {size_mb:.1f} MB")

def main():
    """메인 파이프라인 실행"""
    
    # 명령행 인자 처리
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("RAG 데이터 변환 파이프라인")
        print()
        print("사용법:")
        print(f"  python {Path(__file__).name}        # 테스트 모드 (20개 파일)")
        print(f"  python {Path(__file__).name} --all  # 전체 파일 처리")
        print(f"  python {Path(__file__).name} --help # 도움말")
        print()
        print("파이프라인 단계:")
        print("  1. Parser: 마크다운 → 구조화된 청크")
        print("  2. Normalizer: 텍스트 정규화 및 단위 변환")
        print("     → Parser 파일 자동 정리")
        print("  3. Chunker: 스마트 청킹 및 메타데이터 추가")
        print("     → Normalized 파일 자동 정리")
        print()
        print("자동 정리:")
        print("  • Normalizer 완료 후 Parser 파일 삭제")
        print("  • Chunker 완료 후 Normalized 파일 삭제")
        print("  • 최종 결과만 Final 폴더에 보존")
        sys.exit(0)
    
    # --all 옵션 확인
    process_all = len(sys.argv) > 1 and sys.argv[1] == "--all"
    
    # 시작 시간 기록
    start_time = time.time()
    
    print("🚀 RAG 데이터 변환 파이프라인 시작")
    print(f"   모드: {'전체 파일 처리' if process_all else '테스트 모드 (20개 파일)'}")
    print(f"   시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 입력 파일 확인
    if not check_input_files(process_all):
        print("\n❌ 입력 파일 확인 실패")
        sys.exit(1)
    
    # 2. 출력 디렉토리 확인
    if not check_output_directories():
        print("\n❌ 출력 디렉토리 확인 실패")
        sys.exit(1)
    
    # 3. 파이프라인 실행
    success_count = 0
    total_steps = 3
    
    # Step 1: Parser
    if run_parser(process_all):
        success_count += 1
    else:
        print("\n❌ 파이프라인 중단: Parser 실패")
        sys.exit(1)
    
    # Step 2: Normalizer
    if run_normalizer():
        success_count += 1
    else:
        print("\n❌ 파이프라인 중단: Normalizer 실패")
        sys.exit(1)
    
    # Step 3: Chunker
    if run_chunker():
        success_count += 1
    else:
        print("\n❌ 파이프라인 중단: Chunker 실패")
        sys.exit(1)
    
    # 4. 결과 통계
    get_pipeline_stats()
    
    # 5. 완료 메시지
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("🎉 파이프라인 완료!")
    print("=" * 80)
    print(f"   성공 단계: {success_count}/{total_steps}")
    print(f"   소요 시간: {duration:.1f}초")
    print(f"   완료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_steps:
        print("   상태: ✅ 모든 단계 성공")
    else:
        print("   상태: ⚠️  일부 단계 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()
