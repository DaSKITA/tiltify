from tiltify.parsers.parser import Parser
from tiltify.data_structures.document import Document


class PolicyParser(Parser):

    def __init__(self) -> None:
        pass

    def parse(self, text: str) -> Document:
        return NotImplementedError()
