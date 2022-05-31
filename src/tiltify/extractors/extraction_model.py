from abc import ABC, abstractmethod


class ExtractionModel(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
