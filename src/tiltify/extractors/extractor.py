from abc import ABC, abstractmethod
import os
import pathlib
from typing import Union


from tiltify.config import Path, TILTIFY_ADD, TILTIFY_PORT
from tiltify.data_structures.annotation import PredictedAnnotation
from tiltify.data_structures.document import Document
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.models.binary_bert_model import BinaryBERTModel
from tiltify.models.gaussian_nb_model import GaussianNBModel
from tiltify.models.sentence_bert import SentenceBert
from tiltify.models.test_model import TestModel
from tiltify.extractors.k_ranker import KRanker
from tiltify.data_structures.annotation import PredictedAnnotation
from tiltify.extractors.learning_manager import LearningManager


class ExtractorInterface(ABC):

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

class ModelRegistry:
    model_registry = {
            "BinaryBert": BinaryBERTModel,
            "Test": TestModel,
            "GaussianNB": GaussianNBModel,
            "SentenceBert": SentenceBert
    }

    def __init__(self) -> None:
        self.default_entry = (None, None)

    def get(self, extractor_type: str = None):
        extractor_entry = self.model_registry.get(extractor_type)

        if extractor_entry:
            return extractor_entry
        else:
            print(Warning("Model not found, using default model."))
            return self.default_entry


class ExtractorRegistry:

    def __init__(self) -> None:
        self.extractors = []
        self.extractor_labels = []
        self.max = len(self.extractors)

    def __getitem__(self, idx: Union[str, list]):
        if idx in self.extractor_labels:
            label_idx = [index for index, label in enumerate(self.extractor_labels) if label in idx]
            return self.extractors[label_idx[0]]
        else:
            return None

    def __setitem__(self, idx: Union[str, list], value):
        if idx in self.extractor_labels:
            index = self.extractor_labels.index(idx)
            self.extractors[index] = value
        else:
            self.extractor_labels.append(idx)
            self.extractors.append(value)

    def append(self, labels, extractor):
        self.extractor_labels.append(labels)
        self.extractors.append(extractor)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.extractors):
            extractor = self.extractors[self.n]
            extractor_label = self.extractor_labels[self.n]
            self.n += 1
            return extractor, extractor_label
        else:
            raise StopIteration


class ExtractorManager:

    def __init__(self, extractor_config: list) -> None:
        self._model_registry = ModelRegistry()
        self._extractor_registry = ExtractorRegistry()
        self._learning_manager = LearningManager(extractor_registry=self._extractor_registry, storage_size=1)
        self._init_extractors(extractor_config=extractor_config)

        """

        Args:
            extractor_config (dict): _description_
            mode (str, optional): _description_. Defaults to "group".
        """

    def _init_extractors(self, extractor_config: dict) -> None:
        for model_type, labels in extractor_config:
            extraction_model_cls = self._model_registry.get(model_type)
            # TODO: what to do with multiple labels?
            extractor = Extractor(
                extraction_model_cls=extraction_model_cls, extractor_label=labels)
            self._extractor_registry.append(labels, extractor)

    def get_learning_manager_storage(self):
        return self._learning_manager.get_storage()

    def get_registry_data(self):
        return [(extractor, label) for extractor, label in self._extractor_registry]

    def predict(self, labels: str, document: Document, bare_document: str):
        predictions = []
        for label in labels:
            extractor = self._extractor_registry[label]
            if extractor:
                extractor_predictions = extractor.predict(label, document, bare_document)
                predictions.append(extractor_predictions)
            else:
                predictions.append([PredictedAnnotation()])
        predictions = sum(predictions, [])
        return predictions

    def load(self, extractor_label):
        extractor = self._extractor_registry[extractor_label]
        extractor.load()
        self._extractor_registry[extractor_label] = extractor

    def load_all(self):
        for extractor, extractor_label in self._extractor_registry:
            print(f"Loading Model for {extractor_label}...")
            extractor.load()
            self._extractor_registry[extractor_label] = extractor
        print("All Models loaded!")

    def train(self, labels):
        self._learning_manager.train(labels)

    def train_all(self):
        self._learning_manager.train_all()

    def train_online(self, document_collection, label):
        self._learning_manager.train_online(document_collection, label)


class Extractor(ExtractorInterface):

    def __init__(self, extraction_model_cls, extractor_label, model_path=None, k_ranks=5) -> None:

        self.extractor_label = extractor_label
        if not model_path:
            model_path = os.path.join(
                Path.root_path, "src/tiltify/model_files/")
        self.model_path = os.path.join(model_path + f"{self.__class__.__name__}", f"{self.extractor_label}")
        self.extraction_model_cls = extraction_model_cls
        self.extraction_model = self.extraction_model_cls(label=self.extractor_label)
        self.k_ranker = KRanker(k_ranks)

    def train(self):
        document_collection = DocumentCollection.from_json_files(language="de")
        # bert_splitter = DocumentCollectionSplitter(val=val, split_ratio=split_ratio)
        # train, val, test = bert_splitter.split(document_collection)
        self.extraction_model.train(document_collection=document_collection)
        # experiment = Experiment(
        #     experiment_path=self.model_path, folder_name=self.__class__.__name__)
        # experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
        # experiment.run(k=k, trials=trials, num_processes=num_processes)

    def predict(self, labels: str, document: Document, bare_document: str):
        logits = self.extraction_model.predict(document)
        indices, _ = self.k_ranker.form_k_ranks(logits)
        predictions = [
            PredictedAnnotation.from_model_prediction(
                idx, document, bare_document, self.extractor_label) for idx in indices]
        return predictions

    def load(self):
        self.extraction_model = self.extraction_model_cls.load(self.model_path, self.extractor_label)

    def save(self):
        pathlib.Path(self.model_path).mkdir(mode=0o755, parents=True, exist_ok=True)
        self.extraction_model.save(self.model_path)

    def train_online(self, document_collection: DocumentCollection):
        if self.extraction_model:
            self.extraction_model.train(document_collection)
        else:
            print(Warning("No Model loaded, online training not possible."))


if __name__ == "__main__":
    from tiltify.config import EXTRACTOR_CONFIG
    extractor_manager = ExtractorManager(EXTRACTOR_CONFIG)
    extractor_manager.train(labels=["Right to Information"])
    # extractor_manager.train_all()
