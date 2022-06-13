

class MatchMetricCalculator:

    def __init__(self) -> None:
        pass

    def get_match_accuracy(self, labels, predicted_annotations):
        relevant_indices, retrieved_indices = self._get_relevant_indices(
            labels, predicted_annotations)
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

    def calculate_retrieval_metrics(self, labels, predicted_annotations):
        relevant_indices, retrieved_indices = self._get_relevant_indices(
            labels, predicted_annotations)
        accuracy = self.get_match_accuracy(relevant_indices, retrieved_indices)
        support = self.get_support(relevant_indices)

        metrics = {
            "accuracy": accuracy,
            "support": support
        }
        return metrics

    def _get_relevant_indices(self, labels, predicted_annotations):
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
            relevant_indices.append([idx for idx, label in enumerate(document_labels) if label == 1])
            retrieved_indices.append([
                predicted_annotation.blob_idx for predicted_annotation in predicted_annotations[idx]])
        return relevant_indices, retrieved_indices
