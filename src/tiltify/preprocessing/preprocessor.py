from abc import ABC, abstractmethod
from typing import List
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.preprocessing.label_retriever import LabelRetriever
from tiltify.data_structures.document import Document


class Preprocessor(ABC):

    label_retriever = LabelRetriever()

    @abstractmethod
    def preprocess(self, document_collection: DocumentCollection):
        """Used for Training in the Objective

        Args:
            document_collection (DocumentCollection): _description_
        """
        pass

    @abstractmethod
    def preprocess_document(self, document: Document):
        """Used for inference in the Extractor Class.

        Args:
            document (Document): _description_
        """
        pass

    @abstractmethod
    def prepare_labels(self, labels: List[int]):
        pass
