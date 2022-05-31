from abc import ABC, abstractmethod
import os

from tiltify.data_structures.document import Document
from tiltify.config import Path


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
        self.exp_dir = os.path.join(Path.experiment_path, self.__class__.__name__)

    def train(self, config):

        document_collection = DocumentCollection.from_json_files()
        # pandas_path = os.path.join(Path.data_path, "de_sentence_data.csv")
        preprocessor = BERTPreprocessor(
            bert_model=BASE_BERT_MODEL, binary=binary)
        preprocessed_dataaset = preprocessor.preprocess(document_collection)
        bert_splitter = BERTSplitter(val=True, split_ratio=split_ratio, batch_size=batch_size)
        train, val, test = bert_splitter.split(preprocessed_dataaset)
        experiment = Experiment(experiment_path=exp_dir, title="Binary-Classification", model_name="Bert-FF")

        if binary:
            experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
        else:
            experiment.add_objective(BERTRightToObjective, args=[train, val, test])

        experiment.run(k=k, trials=trials, num_processes=num_processes)

    def predict(self, document: Document):
        return super().predict(document)
