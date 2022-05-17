from typing import List, Union

from tiltify.data_structures.annotation import Annotation


class Blob:

    def __init__(self, text: str, annotations: List[Annotation] = None) -> None:
        """
        A blob is a parsable object from a given string. Whereas the parsed string represents a document.
        Blobs can thus be seen as paragraphs. The text gives the paragraph as a string. The annotations bear
        all found TILT annotations.

        Args:
            text (str): [description]
        """
        self.text = text
        if annotations:
            self.annotations = annotations
        else:
            self.annotations = []
        self.prediction_annotations = []

    def add_annotation(self, annotation: Union[Annotation, List[Annotation]]):
        if isinstance(annotation, list):
            self.annotations += annotation
        else:
            self.annotations.append(annotation)

    def get_annotations(self) -> List[str]:
        return [annotation for annotation in self.annotations]

    def add_prediction_annotation(self, annotation: Union[Annotation, List[Annotation]]):
        if isinstance(annotation, list):
            self.prediction_annotations += annotation
        else:
            self.prediction_annotations.append(annotation)

    def get_prediction_annotations(self) -> List[str]:
        return [annotation for annotation in self.prediction_annotations]
