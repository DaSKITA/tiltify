from abc import ABC, abstractmethod
import os
from rapidflow.experiments.experiment import Experiment

from tiltify.data_structures.document import Document
from tiltify.config import Path
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.preprocessing.document_collection_splitter import DocumentCollectionSplitter
from tiltify.objectives.bert_objective.bert_preprocessor import BERTPreprocessor
from tiltify.objectives.bert_objective.bert_binary_objective import BERTBinaryObjective
from tiltify.config import BASE_BERT_MODEL


class Extractor(ABC):

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
        self.extraction_model = None

    def train(
            self, k: int, trials: int, num_processes: int, val: bool, split_ratio: float, batch_size: float):
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
        preprocessor = BERTPreprocessor(bert_model=BASE_BERT_MODEL, binary=True, batch_size=batch_size)
        train, val, test = bert_splitter.split(document_collection)
        val = [preprocessor.preprocess_document(document) for document in val]
        test = [preprocessor.preprocess_document(document) for document in test]
        experiment = Experiment(
            experiment_path=self.exp_dir, title=self.__class__.__name__,
            model_name=self.extraction_model.__class__.__name__)
        experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
        experiment.run(k=k, trials=trials, num_processes=num_processes)

    def predict(self, document: Document):
        document_predictions = self.extraction_model.predict(document)
        # predicted annotations
        pass
