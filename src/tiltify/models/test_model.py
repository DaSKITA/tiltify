from time import sleep
import random
from tqdm import tqdm
from typing import List
import os
from tiltify.extractors.extraction_model import ExtractionModel
from tiltify.data_structures.document import Document, PredictedAnnotation
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.preprocessing.preprocessor import Preprocessor
from tiltify.preprocessing.label_retriever import LabelRetriever


class TestModel(ExtractionModel):

    def __init__(self, num_train_epochs=2, label=None, k_ranks=5) -> None:
        self.num_train_epochs = num_train_epochs
        self.label = label
        self.model = False
        self.k_ranks = k_ranks
        self.preprocessor = TestPreprocessor(label)

    def train(self, document_collection: DocumentCollection):
        for _ in tqdm(range(self.num_train_epochs)):
            for batch in range(self.num_train_epochs):
                sleep(0.5)
        self.model = True

    def predict(self, document: Document):
        indexes = list(range(len(document.blobs)))
        indices = list()
        if self.model:
            for i in range(self.k_ranks):
                indices.append(random.choice(indexes))
        else:
            raise AssertionError("No Model loaded!")
        return indices

    @classmethod
    def load(cls, load_path, label):
        load_model = os.path.join(load_path, "test_model.txt")
        init_obj = cls(label=label)
        with open(load_model, "r") as f:
            lines = f.readlines()
        print(lines)
        init_obj.model = True
        return init_obj

    def save(self, save_path):
        save_model = os.path.join(save_path, "test_model.txt")
        with open(save_model, "w") as f:
            f.write("this is a test.")


class TestPreprocessor(Preprocessor):

    def __init__(self, label, binary=True) -> None:
        self.label_retriever = LabelRetriever(label)
        self.binary = binary

    def prepare_labels(self, labels: List):
        if self.binary:
            labels = [int(label[0] > 0) for label in labels]
        return labels

    def preprocess(self, document_collection: DocumentCollection):
        raise NotImplementedError()

    def preprocess_document(self, document: Document):
        raise NotImplementedError()
