def normalize(citations):
    """
    중복 레퍼런스(동일 report_id와 chunk_id를 가진 경우)를 제거하고,
    최초 등장 순서로 정렬하여 반환한다.

    Args:
        citations (list[dict]): citation 객체 리스트
            각 citation은 최소한 "report_id"와 "chunk_id"를 포함해야 한다.

    Returns:
        list[dict]: 중복이 제거된 citation 리스트 (입력 순서 유지)
    """
    seen = set()
    out = []
    for c in citations:
        key = (c.get("report_id"), c.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
