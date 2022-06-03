

import re


class MatchMetricCalculator:

    def __init__(self) -> None:
        pass

    def get_match_accuracy(self, relevant_indices, retrieved_indices):
        document_match = []
        for idx, relevant_per_doc_indices in enumerate(relevant_indices):
            retrieved_per_doc_indices = retrieved_indices[idx]
            match_per_doc = [
                True if relevant_index in retrieved_per_doc_indices else False
                for relevant_index in relevant_per_doc_indices]
            document_match.append(all(match_per_doc))
        accuracy = sum(document_match)/len(document_match)
        return accuracy

    def get_support(self, relevant_indices):
        return len(relevant_indices)

    def calculate_retrieval_metrics(self, relevant_indices, retrieved_indices):
        accuracy = self.get_match_accuracy(relevant_indices, retrieved_indices)
        support = self.get_support(relevant_indices)

        metrics = {
            "accuracy": accuracy,
            "support": support
        }
        return metrics
