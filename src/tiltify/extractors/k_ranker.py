from typing import List


class KRanker:

    def __init__(self, k_ranks) -> None:
        self.k_ranks = k_ranks

    def form_k_ranks(self, logits: List[float]) -> List[int]:
        indices = sorted(logits, key=lambda k: logits[k])
        return indices[self.k_ranks], logits[indices][self.k_ranks]
