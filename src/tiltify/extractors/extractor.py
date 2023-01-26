from abc import ABC, abstractmethod
import os
import pathlib

from tiltify.data_structures.document import Document
from tiltify.config import Path
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.preprocessing.document_collection_splitter import DocumentCollectionSplitter

from tiltify.objectives.bert_objective.bert_binary_objective import BERTBinaryObjective
from tiltify.objectives.bert_objective.binary_bert_model import BinaryBERTModel
from tiltify.extractors.extraction_model import ExtractionModel
from tiltify.models.test_model import TestModel
from tiltify.annotation_shaper import AnnotationShaper


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


class ExtractionModelRegistry:
    model_registry = {
            "BinaryBert": (
                BinaryBERTModel,
                os.path.join(Path.root_path, f"src/tiltify/model_files/{BinaryBERTModel.__class__.__name__}")
            ),
            "Test": (
                TestModel,
                os.path.join(Path.root_path, f"src/tiltify/model_files/{TestModel.__class__.__name__}")
            )
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


class Extractor(ExtractorInterface):

    def __init__(self, extractor_type, extractor_label) -> None:

        self.extraction_model_registry = ExtractionModelRegistry()
        self.extraction_model_cls, self.model_path = self.extraction_model_registry.get(extractor_type)
        self.extractor_label = extractor_label
        self.annotation_shaper = AnnotationShaper(label=extractor_label)

    def train(self):
        document_collection = DocumentCollection.from_json_files()
        # bert_splitter = DocumentCollectionSplitter(val=val, split_ratio=split_ratio)
        # train, val, test = bert_splitter.split(document_collection)
        self.extraction_model = self.extraction_model_cls(label=self.extractor_label)
        self.extraction_model.train(document_collection=document_collection)
        # experiment = Experiment(
        #     experiment_path=self.model_path, folder_name=self.__class__.__name__)
        # experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
        # experiment.run(k=k, trials=trials, num_processes=num_processes)

    def predict(self, labels: str, document: Document, bare_document: str):
        if self.extractor_label in labels:
            predictions = self.extraction_model.predict(document)
        else:
            predictions = []
        predictions = self.annotation_shaper.form_predict_annotations(predictions, document, bare_document)
        return predictions

    def load(self):
        self.extraction_model = self.extraction_model_cls.load(self.model_path)

    def save(self):
        pathlib.Path(self.model_path).mkdir(parents=True, exist_ok=True)
        self.extraction_model.save(self.model_path)

    def train_online(self, document_collection: DocumentCollection):
        if self.extraction_model:
            self.extraction_model.train(document_collection=document_collection)
            self.save()
        else:
            print(Warning("No Model loaded, online training not possible."))


if __name__ == "__main__":
    extractor = Extractor("Test")
    extractor.train()
    extractor.save()
    extractor.load()
    extractor.predict()
