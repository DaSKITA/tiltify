from typing import Dict
from rapidflow.metrics_handler import MetricsHandler
from rapidflow.objective import Objective

from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.objectives.bert_objective.binary_bert_model import BinaryBERTModel
from tiltify.preprocessing.label_retriever import LabelRetriever


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
            num_train_epochs=1,
            weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-5, log=True),
            batch_size=2
        )
        metrics_handler = MetricsHandler()
        # model setup
        model = BinaryBERTModel(learning_rate=hyperparameters["learning_rate"],
                                weight_decay=hyperparameters["weight_decay"],
                                num_train_epochs=hyperparameters["num_train_epochs"],
                                batch_size=hyperparameters["batch_size"])
        # add  custom save strategy to rapidflow
        self.track_model(model, hyperparameters)
        model.train(document_collection=self.train_collection)

        val_labels = []
        val_preds = []
        label_retriever = LabelRetriever()
        for document in self.val_collection:
            predicted_annotations = self.model.predict(document)
            pred_idx = [predicted_annotation.blob_idx for predicted_annotation in predicted_annotations]
            document_labels = label_retriever.retrieve_labels(document.blobs)
            document_labels = self.model.preprocessor.prepare_labels(document_labels)
            # adjust evaluation
            found = [document_labels[pred_id] for pred_id in pred_idx]
        metrics_handler = MetricsHandler()
        metrics = metrics_handler.calculate_classification_metrics(val_labels, val_preds)
        return metrics['macro avg f1-score']

    def test(self):
        test_labels = []
        test_preds = []
        label_retriever = LabelRetriever()
        for document in self.test_collection:
            predicted_annotations = self.model.predict(document)
            blob_idx = [predicted_annotation.blob_idx for predicted_annotation in predicted_annotations]
            document_labels = label_retriever.retrieve_labels(document.blobs)
            document_labels = self.model.preprocessor.prepare_labels(document_labels)
            test_labels += document_labels
            test_preds += [1 if idx in blob_idx else 0 for idx, _ in enumerate(document_labels)]
        metrics_handler = MetricsHandler()
        metrics = metrics_handler.calculate_classification_metrics(test_labels, test_preds)
        return metrics
