from typing import List
from collections import Counter
import numpy as np


class MatchMetricCalculator:

    def __init__(self) -> None:
        pass

    def get_match_accuracy(self, labels: List[int], predicted_annotations):
        relevant_indices, retrieved_indices = self._get_relevant_indices(
            labels, predicted_annotations)
        document_match = []
        document_match_counts = []
        document_match_rate = []
        for idx, relevant_per_doc_indices in enumerate(relevant_indices):
            retrieved_per_doc_indices = retrieved_indices[idx]
            match_per_doc = [
                True if relevant_index in retrieved_per_doc_indices else False
                for relevant_index in relevant_per_doc_indices]
            document_match.append(all(match_per_doc))
            document_match_counts.append(sum(match_per_doc))
            document_match_rate.append(sum(match_per_doc)/len(match_per_doc))
        accuracy = sum(document_match)/len(document_match)
        avg_match_per_doc = np.round(np.mean(document_match_counts), 3).astype(float)
        avg_document_match_rate = np.round(np.mean(document_match_rate), 3).astype(float)
        accuracy_dict = {
            "exact_match_accuracy": accuracy,
            "avg_match_per_doc": avg_match_per_doc,
            "avg_match_rate_per_doc": avg_document_match_rate
        }
        return accuracy_dict

    def get_support(self, relevant_indices):
        return len(relevant_indices)

    def calculate_retrieval_metrics(self, labels: List[int], predicted_annotations):
        accuracy = self.get_match_accuracy(labels, predicted_annotations)
        support = self.get_support(labels)
        label_support = self._get_label_support(labels)

        metrics = {
            "accuracy": accuracy,
            "support": support,
            "labels": label_support
        }
        return metrics

    def _get_relevant_indices(self, labels: List[int], predicted_annotations):
        """_summary_

        Args:
            labels (_type_): _description_
            predicted_annotations (_type_): _description_

        Returns:
            _type_: _description_
        """
        relevant_indices = []
        retrieved_indices = []
        for idx, document_labels in enumerate(labels):
            relevant_indices.append([idx for idx, label in enumerate(document_labels) if label > 0])
            retrieved_indices.append([
                predicted_annotation.blob_idx for predicted_annotation in predicted_annotations[idx]])
        return relevant_indices, retrieved_indices

    def _get_label_support(self, labels: List[int]):
        uniques_counts = Counter(sum(labels, []))
        return dict(uniques_counts)


if __name__ == "__main__":
    from tiltify.data_structures.document import PredictedAnnotation

    predictions = [
        [PredictedAnnotation(blob_idx=1, blob_text="", label="consumer_right")],
        [PredictedAnnotation(blob_idx=4, blob_text="", label="consumer_right")]]
    labels = [[0, 1, 1, 3], [4, 2, 3, 3, 4]]
    calculator = MatchMetricCalculator()
    print(calculator._get_label_support(labels))
    print(calculator.get_match_accuracy(labels, predictions))
    print(calculator.calculate_retrieval_metrics(labels, predictions))


# Negative Sampling?
