from abc import ABC, abstractmethod
from tiltify.data_structures.document import Document
from tiltify.data_structures.document_collection import DocumentCollection


class ExtractorModel(ABC):
    """Serves as an Interface for Extractors.

    Args:
        ABC (_type_): _description_
        metaclass (_type_, optional): _description_. Defaults to ExtractorCombinedMeta.
    """

    exp_dir = None

    @abstractmethod
    def predict(self, document: Document):
        pass

    @abstractmethod
    def train(self, document_collection: DocumentCollection):
        pass

    @abstractmethod
    def load(self):
        pass
