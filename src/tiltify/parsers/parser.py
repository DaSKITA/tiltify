from abc import ABC, abstractmethod
from tiltify.data_structures.document import Document


class Parser(ABC):

    @abstractmethod
    def parse(self, text: str) -> Document:
        pass
