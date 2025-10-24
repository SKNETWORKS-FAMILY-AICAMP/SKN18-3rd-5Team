#!/usr/bin/env python3
"""
macOS 시스템 정리 스크립트

디스크 공간 부족 시 시스템 데이터 정리
- 캐시 파일 삭제
- 로그 파일 정리
- 임시 파일 삭제
- 휴지통 비우기
- 개발 관련 캐시 정리
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """명령어 실행 및 결과 출력"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} 완료")
            if result.stdout.strip():
                print(f"   출력: {result.stdout.strip()}")
        else:
            print(f"⚠️  {description} 실패: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ {description} 오류: {e}")
        return False

def get_directory_size(path):
    """디렉토리 크기 계산 (MB)"""
    try:
        if not os.path.exists(path):
            return 0
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
        return total_size / (1024 * 1024)  # MB로 변환
    except Exception:
        return 0

def cleanup_caches():
    """캐시 파일 정리"""
    print("\n" + "=" * 80)
    print("🧹 캐시 파일 정리")
    print("=" * 80)
    
    cache_dirs = [
        "~/Library/Caches",
        "~/Library/Application Support/CrashReporter",
        "~/Library/Logs",
        "~/.cache",
        "/tmp",
        "/var/tmp"
    ]
    
    total_freed = 0
    
    for cache_dir in cache_dirs:
        expanded_dir = os.path.expanduser(cache_dir)
        if os.path.exists(expanded_dir):
            size_before = get_directory_size(expanded_dir)
            
            # 안전한 캐시 정리
            if "Caches" in cache_dir:
                # Caches 폴더는 내용만 삭제, 폴더는 유지
                for item in os.listdir(expanded_dir):
                    item_path = os.path.join(expanded_dir, item)
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                    except (OSError, PermissionError):
                        pass
            else:
                # tmp 폴더는 내용만 삭제
                for item in os.listdir(expanded_dir):
                    item_path = os.path.join(expanded_dir, item)
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                    except (OSError, PermissionError):
                        pass
            
            size_after = get_directory_size(expanded_dir)
            freed = size_before - size_after
            total_freed += freed
            
            if freed > 0:
                print(f"✅ {cache_dir}: {freed:.1f} MB 정리")
            else:
                print(f"ℹ️  {cache_dir}: 정리할 파일 없음")
    
    print(f"\n📊 캐시 정리 결과: {total_freed:.1f} MB 확보")
    return total_freed

def cleanup_logs():
    """로그 파일 정리"""
    print("\n" + "=" * 80)
    print("📝 로그 파일 정리")
    print("=" * 80)
    
    log_dirs = [
        "~/Library/Logs",
        "/var/log",
        "~/Library/Application Support/DiagnosticReports"
    ]
    
    total_freed = 0
    
    for log_dir in log_dirs:
        expanded_dir = os.path.expanduser(log_dir)
        if os.path.exists(expanded_dir):
            size_before = get_directory_size(expanded_dir)
            
            # 7일 이상 된 로그 파일 삭제
            current_time = time.time()
            for root, dirs, files in os.walk(expanded_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if os.path.getmtime(file_path) < current_time - (7 * 24 * 3600):  # 7일
                            os.remove(file_path)
                    except (OSError, PermissionError):
                        pass
            
            size_after = get_directory_size(expanded_dir)
            freed = size_before - size_after
            total_freed += freed
            
            if freed > 0:
                print(f"✅ {log_dir}: {freed:.1f} MB 정리")
            else:
                print(f"ℹ️  {log_dir}: 정리할 로그 없음")
    
    print(f"\n📊 로그 정리 결과: {total_freed:.1f} MB 확보")
    return total_freed

def cleanup_development_caches():
    """개발 관련 캐시 정리"""
    print("\n" + "=" * 80)
    print("💻 개발 캐시 정리")
    print("=" * 80)
    
    dev_caches = [
        "~/.npm",
        "~/.yarn",
        "~/.pip/cache",
        "~/.cache/pip",
        "~/Library/Caches/pip",
        "~/.gradle/caches",
        "~/.m2/repository",
        "~/Library/Caches/Homebrew",
        "~/.cache/go-build",
        "~/.cargo/registry/cache"
    ]
    
    total_freed = 0
    
    for cache_path in dev_caches:
        expanded_path = os.path.expanduser(cache_path)
        if os.path.exists(expanded_path):
            size_before = get_directory_size(expanded_path)
            
            try:
                shutil.rmtree(expanded_path)
                size_after = 0
                freed = size_before
                total_freed += freed
                print(f"✅ {cache_path}: {freed:.1f} MB 정리")
            except (OSError, PermissionError):
                print(f"⚠️  {cache_path}: 권한 없음")
        else:
            print(f"ℹ️  {cache_path}: 없음")
    
    print(f"\n📊 개발 캐시 정리 결과: {total_freed:.1f} MB 확보")
    return total_freed

def cleanup_trash():
    """휴지통 비우기"""
    print("\n" + "=" * 80)
    print("🗑️  휴지통 비우기")
    print("=" * 80)
    
    trash_dirs = [
        "~/.Trash",
        "~/Library/Mobile Documents/com~apple~CloudDocs/.Trash"
    ]
    
    total_freed = 0
    
    for trash_dir in trash_dirs:
        expanded_dir = os.path.expanduser(trash_dir)
        if os.path.exists(expanded_dir):
            size_before = get_directory_size(expanded_dir)
            
            try:
                for item in os.listdir(expanded_dir):
                    item_path = os.path.join(expanded_dir, item)
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                    except (OSError, PermissionError):
                        pass
                
                size_after = get_directory_size(expanded_dir)
                freed = size_before - size_after
                total_freed += freed
                
                if freed > 0:
                    print(f"✅ {trash_dir}: {freed:.1f} MB 정리")
                else:
                    print(f"ℹ️  {trash_dir}: 비어있음")
            except (OSError, PermissionError):
                print(f"⚠️  {trash_dir}: 권한 없음")
        else:
            print(f"ℹ️  {trash_dir}: 없음")
    
    print(f"\n📊 휴지통 정리 결과: {total_freed:.1f} MB 확보")
    return total_freed

def cleanup_xcode_caches():
    """Xcode 캐시 정리"""
    print("\n" + "=" * 80)
    print("🔨 Xcode 캐시 정리")
    print("=" * 80)
    
    xcode_caches = [
        "~/Library/Developer/Xcode/DerivedData",
        "~/Library/Developer/Xcode/Archives",
        "~/Library/Caches/com.apple.dt.Xcode",
        "~/Library/Developer/Xcode/iOS DeviceSupport"
    ]
    
    total_freed = 0
    
    for cache_path in xcode_caches:
        expanded_path = os.path.expanduser(cache_path)
        if os.path.exists(expanded_path):
            size_before = get_directory_size(expanded_path)
            
            try:
                shutil.rmtree(expanded_path)
                size_after = 0
                freed = size_before
                total_freed += freed
                print(f"✅ {cache_path}: {freed:.1f} MB 정리")
            except (OSError, PermissionError):
                print(f"⚠️  {cache_path}: 권한 없음")
        else:
            print(f"ℹ️  {cache_path}: 없음")
    
    print(f"\n📊 Xcode 캐시 정리 결과: {total_freed:.1f} MB 확보")
    return total_freed

def cleanup_python_caches():
    """Python 캐시 정리"""
    print("\n" + "=" * 80)
    print("🐍 Python 캐시 정리")
    print("=" * 80)
    
    # 현재 프로젝트의 __pycache__ 정리
    project_root = Path(__file__).parent
    total_freed = 0
    
    for pycache_dir in project_root.rglob("__pycache__"):
        if pycache_dir.is_dir():
            size_before = get_directory_size(str(pycache_dir))
            try:
                shutil.rmtree(pycache_dir)
                total_freed += size_before
                print(f"✅ {pycache_dir.relative_to(project_root)}: {size_before:.1f} MB 정리")
            except (OSError, PermissionError):
                print(f"⚠️  {pycache_dir}: 권한 없음")
    
    # .pyc 파일 정리
    for pyc_file in project_root.rglob("*.pyc"):
        try:
            size = os.path.getsize(pyc_file) / (1024 * 1024)
            os.remove(pyc_file)
            total_freed += size
            print(f"✅ {pyc_file.relative_to(project_root)}: {size:.1f} MB 정리")
        except (OSError, PermissionError):
            pass
    
    print(f"\n📊 Python 캐시 정리 결과: {total_freed:.1f} MB 확보")
    return total_freed

def get_disk_usage():
    """디스크 사용량 확인"""
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 5:
                    total = parts[1]
                    used = parts[2]
                    available = parts[3]
                    percent = parts[4]
                    return total, used, available, percent
    except Exception:
        pass
    return None, None, None, None

def main():
    """메인 정리 함수"""
    print("🚀 macOS 시스템 정리 시작")
    print("=" * 80)
    
    # 시작 전 디스크 사용량
    total, used, available, percent = get_disk_usage()
    if total:
        print(f"📊 시작 전 디스크 사용량:")
        print(f"   전체: {total}, 사용: {used}, 여유: {available} ({percent})")
    
    total_freed = 0
    
    # 1. Python 캐시 정리
    total_freed += cleanup_python_caches()
    
    # 2. 개발 캐시 정리
    total_freed += cleanup_development_caches()
    
    # 3. Xcode 캐시 정리
    total_freed += cleanup_xcode_caches()
    
    # 4. 일반 캐시 정리
    total_freed += cleanup_caches()
    
    # 5. 로그 파일 정리
    total_freed += cleanup_logs()
    
    # 6. 휴지통 비우기
    total_freed += cleanup_trash()
    
    # 정리 후 디스크 사용량
    print("\n" + "=" * 80)
    print("🎉 정리 완료!")
    print("=" * 80)
    
    total_after, used_after, available_after, percent_after = get_disk_usage()
    if total_after:
        print(f"📊 정리 후 디스크 사용량:")
        print(f"   전체: {total_after}, 사용: {used_after}, 여유: {available_after} ({percent_after})")
    
    print(f"💾 총 확보된 공간: {total_freed:.1f} MB")
    
    if total_freed > 1000:
        print(f"✅ {total_freed/1024:.1f} GB 공간 확보 성공!")
    elif total_freed > 100:
        print(f"✅ {total_freed:.1f} MB 공간 확보 성공!")
    else:
        print(f"ℹ️  {total_freed:.1f} MB 공간 확보 (추가 정리 필요할 수 있음)")

if __name__ == "__main__":
    main()
