from abc import ABC, abstractmethod
import os
from rapidflow.experiments.experiment import Experiment

from tiltify.data_structures.document import Document
from tiltify.config import Path
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.preprocessing.document_collection_splitter import DocumentCollectionSplitter

from tiltify.objectives.bert_objective.bert_binary_objective import BERTBinaryObjective
from tiltify.objectives.bert_objective.binary_bert_model import BinaryBERTModel
import torch


class Extractor(ABC):

    extraction_model_cls = None
    extraction_model = None

    @abstractmethod
    def train(self, config):
        pass

    @abstractmethod
    def predict(self, document: Document):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def train_online(self, document: DocumentCollection):
        pass


class BinaryBERTExtractor(Extractor):
    extraction_model_cls = BinaryBERTModel

    def __init__(self) -> None:
        self.extraction_model = None
        self.model_path = os.path.join(Path.root_path, f"src/tiltify/model_files/{self.__class__.__name__}")
        # TODO: adjust objective?

    def train(
            self, k: int, trials: int, val: bool, split_ratio: float, num_processes: int = None):
        """
        Train is performed on sentence level.
        Validate and Test is performed on Document Level

        Args:
            k (_type_): _description_
            trials (_type_): _description_
            num_processes (_type_): _description_
            val (_type_): _description_
            split_ratio (_type_): _description_
            batch_size (_type_): _description_
        """

        document_collection = DocumentCollection.from_json_files()
        # bert_splitter = DocumentCollectionSplitter(val=val, split_ratio=split_ratio)
        # train, val, test = bert_splitter.split(document_collection)
        self.extraction_model = self.extraction_model_cls(
            learning_rate=1e-3, weight_decay=1e-5, num_train_epochs=5, batch_size=3, k_ranks=5)
        self.extraction_model.train(document_collection=document_collection)
        # experiment = Experiment(
        #     experiment_path=self.model_path, folder_name=self.__class__.__name__)
        # experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
        # experiment.run(k=k, trials=trials, num_processes=num_processes)

    def predict(self, document: Document):
        predicted_annotations = self.extraction_model.predict(document)
        return predicted_annotations

    def train_online(self, documents: DocumentCollection):
        pass

    def load_best_model(self):
        pass

    def load(self):
        self.extraction_model = self.extraction_model_cls()
        self.extraction_model.model.from_pretrained(self.model_path)

    def save(self):
        self.extraction_model.model.save_pretrained(self.model_path)


if __name__ == "__main__":
    extractor = BinaryBERTExtractor()
    config = {
        "k": 1,
        "trials": 1,
        "val": True,
        "split_ratio": 0.33,
    }
    extractor.train(**config)
    extractor.save()
    extractor.load()
