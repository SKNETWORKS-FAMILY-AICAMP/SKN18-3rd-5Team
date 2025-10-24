from typing import List, Dict

async def rerank(question: str, candidates: List[Dict], top_n: int = 4) -> List[Dict]:
    return candidates[:top_n]
