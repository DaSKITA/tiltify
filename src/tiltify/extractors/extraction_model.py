from abc import ABC, abstractmethod, abstractclassmethod


class ExtractionModel(ABC):

    model = None
    labels = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractclassmethod
    def load(self):
        pass
