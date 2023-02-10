from typing import List
import numpy as np

class KRanker:

    def __init__(self, k_ranks) -> None:
        self.k_ranks = k_ranks

    def form_k_ranks(self, logits: List[float]) -> List[int]:
        logits = np.array(logits)
        indices = np.argsort(logits)[::-1].tolist()
        return indices[:self.k_ranks], logits[indices].tolist()[:self.k_ranks]
