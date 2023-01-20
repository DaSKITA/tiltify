from time import sleep
import random
from tqdm import tqdm
import os
from tiltify.extractors.extraction_model import ExtractionModel
from tiltify.data_structures.document import Document, PredictedAnnotation
from tiltify.data_structures.document_collection import DocumentCollection


class TestModel(ExtractionModel):

    def __init__(self, num_train_epochs=2) -> None:
        self.num_train_epochs = num_train_epochs
        self.label_name = "Test_Right"
        self.model = False

    def train(self, document_collection: DocumentCollection):
        for _ in tqdm(range(self.num_train_epochs)):
            for batch in range(self.num_train_epochs):
                sleep(0.5)
        self.model = True

    def predict(self, document: Document):
        indexes = list(range(len(document.blobs)))
        indices = list()
        if self.model:
            for i in range(5):
                indices.append(random.choice(indexes))
        else:
            raise AssertionError("No Model loaded!")
        return indices

    @classmethod
    def load(cls, load_path):
        load_model = os.path.join(load_path, "test_model.txt")
        init_obj = cls()
        with open(load_model, "r") as f:
            lines = f.readlines()
        print(lines)
        init_obj.model = True
        return init_obj

    def save(self, save_path):
        save_model = os.path.join(save_path, "test_model.txt")
        with open(save_model, "w") as f:
            f.write("this is a test.")
