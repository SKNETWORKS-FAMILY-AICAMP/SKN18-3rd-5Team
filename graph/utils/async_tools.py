from __future__ import annotations

import asyncio
from typing import Any


def run_sync(awaitable_or_value: Any) -> Any:
    """
    동기 컨텍스트에서 코루틴/미래 객체를 안전하게 실행.

    - 현재 이벤트 루프가 존재하고 실행 중이면 run_coroutine_threadsafe 사용
    - 존재하지만 실행 중이 아니면 run_until_complete 사용
    - 아예 없으면 asyncio.run으로 새 이벤트 루프 생성
    - 코루틴/미래가 아닌 일반 값은 그대로 반환
    """
    if not asyncio.isfuture(awaitable_or_value) and not asyncio.iscoroutine(awaitable_or_value):
        return awaitable_or_value

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(awaitable_or_value, loop).result()
    if loop:
        return loop.run_until_complete(awaitable_or_value)
    return asyncio.run(awaitable_or_value)
