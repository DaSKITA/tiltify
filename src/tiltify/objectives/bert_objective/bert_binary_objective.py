from typing import Dict
from rapidflow.objective import Objective

from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.objectives.bert_objective.binary_bert_model import BinaryBERTModel
from tiltify.preprocessing.label_retriever import LabelRetriever
from tiltify.utils.match_metric_calculator import MatchMetricCalculator


class BERTBinaryObjective(Objective):

    def __init__(
            self, train_collection: DocumentCollection, val_collection: DocumentCollection,
            test_collection: DocumentCollection):
        """https://huggingface.co/docs/transformers/training

        Args:
            train_dataset (TiltDataset): _description_
            val_dataset (TiltDataset): _description_
            test_dataset (TiltDataset): _description_
        """
        super().__init__()
        self.train_collection, self.val_collection, self.test_collection = \
            train_collection, val_collection, test_collection

    def train(self, trial=None) -> Dict:
        hyperparameters = dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            num_train_epochs=5,
            weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-5, log=True),
            batch_size=35
        )
        metrics_handler = MatchMetricCalculator()
        # model setup
        model = BinaryBERTModel(learning_rate=hyperparameters["learning_rate"],
                                weight_decay=hyperparameters["weight_decay"],
                                num_train_epochs=hyperparameters["num_train_epochs"],
                                batch_size=hyperparameters["batch_size"])
        # add  custom save strategy to rapidflow
        self.track_model(model, hyperparameters)
        model.train(document_collection=self.train_collection)

        predicted_annotations = []
        labels = []
        label_retriever = LabelRetriever()
        for document in self.val_collection:
            document_labels = label_retriever.retrieve_labels(document.blobs)
            document_labels = self.model.preprocessor.prepare_labels(document_labels)
            predicted_annotations.append(self.model.predict(document))
            labels.append(document_labels)
        accuracy = metrics_handler.get_match_accuracy(labels, predicted_annotations)
        # metrics = metrics_handler.calculate_classification_metrics(val_labels, val_preds)
        return accuracy

    def test(self):
        metrics_handler = MatchMetricCalculator()
        predicted_annotations = []
        predicted_annotations_10 = []
        labels = []
        label_retriever = LabelRetriever()
        for document in self.test_collection:
            document_labels = label_retriever.retrieve_labels(document.blobs)
            document_labels = self.model.preprocessor.prepare_labels(document_labels)
            labels.append(document_labels)
            predicted_annotations.append(self.model.predict(document))
            self.model.set_k_ranks(10)
            predicted_annotations_10.append(self.model.predict(document))
        accuracy = metrics_handler.calculate_retrieval_metrics(labels, predicted_annotations)
        accuracy_10 = metrics_handler.calculate_retrieval_metrics(labels, predicted_annotations_10)
        metrics = {"accuracy": accuracy, "accuracy_10": accuracy_10}
        return metrics
