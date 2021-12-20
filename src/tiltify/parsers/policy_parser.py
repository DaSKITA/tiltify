from tiltify.parsers.parser import Parser
from tiltify.data_structures.blob import Blob
from tiltify.data_structures.document import Document


class PolicyParser(Parser):

    def __init__(self) -> None:
        pass

    def parse(self, title: str, text: str) -> Document:
        blobs = [Blob(blob) for blob in text.split('\n') if blob != '']
        doc = Document(title, blobs)
        return doc
