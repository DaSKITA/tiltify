from abc import ABC, abstractmethod
import os
from numpy import extract
from rapidflow.experiments.experiment import Experiment

from tiltify.data_structures.document import Document
from tiltify.config import Path
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.preprocessing.document_collection_splitter import DocumentCollectionSplitter

from tiltify.objectives.bert_objective.bert_binary_objective import BERTBinaryObjective
from tiltify.objectives.bert_objective.binary_bert_model import BinaryBERTModel


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


class BinaryBERTExtractor(Extractor):

    def __init__(self) -> None:
        self.exp_dir = os.path.join(Path.experiment_path)
        self.extraction_model_cls = BinaryBERTModel

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
        bert_splitter = DocumentCollectionSplitter(val=val, split_ratio=split_ratio)
        train, val, test = bert_splitter.split(document_collection)
        experiment = Experiment(
            experiment_path=self.exp_dir, title=self.__class__.__name__,
            model_name=self.extraction_model.__class__.__name__)
        experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
        experiment.run(k=k, trials=trials, num_processes=num_processes)

        self.extraction_model = self.load_best_model()

    def predict(self, document: Document):
        if self.extraction_model:
            predicted_annotations = self.extraction_model.predict(document)
        else:
            Warning("Modle not loaded!")
        # predicted annotations
        pass

    def load_best_model(self):
        pass

    def load(self):
        pass


if __name__ == "__main__":
    extractor = BinaryBERTExtractor()
    config = {
        "k": 1,
        "trials": 1,
        "val": True,
        "split_ratio": 0.33,
    }
    extractor.train(**config)
