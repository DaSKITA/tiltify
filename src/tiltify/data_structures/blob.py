from typing import Union, List

from tiltify.data_structures.annotation import Annotation


class Blob:

    def __init__(self, text: str) -> None:
        """
        A blob is a parsable object from a given string. Whereas the parsed string represents a document.
        Blobs can thus be seen as paragraphs. The text gives the paragraph as a string. The annotations bear
        all found TILT annotations.

        Args:
            text (str): [description]
        """
        self.text = text
        self.annotations = []

    def __eq__(self, other):
        if self.text == other.text and self.annotations == other.annotations:
            return True
        return False

    def add_annotation(self, annotation: Union[Annotation, List[Annotation]]):
        if isinstance(annotation, list):
            self.annotations += annotation
        else:
            self.annotations.append(annotation)
