from typing import List
import numpy as np

async def embed_query(text: str) -> List[float]:
    return np.random.RandomState(abs(hash(text)) % (2**32)).rand(384).tolist()
