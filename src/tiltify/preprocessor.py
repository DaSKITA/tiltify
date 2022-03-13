from abc import ABC, abstractmethod
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.data_structures.document import Document


class Preprocessor(ABC):

    @abstractmethod
    def preprocess(self, document_collection: DocumentCollection):
        pass
