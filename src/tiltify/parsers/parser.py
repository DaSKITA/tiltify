from abc import ABC, abstractmethod
from typing import List

from tiltify.data_structures.document import Document


class Parser(ABC):

    @abstractmethod
    def parse(self, title: str, text: str, annotations: List) -> Document:
        pass
