from typing import List, Union

from tiltify.data_structures.blob import Blob
from tiltify.data_structures.annotation import PredictedAnnotation


class Document:

    def __init__(self, title: str, blobs: List[Blob] = None, language: str = None) -> None:
        """
        A Document is a list of blobs and represents a real document. In this case a privacy policy.
        It bears a title as well.

        Args:
            title (str): [description]
            blobs (Blob): [description]
        """
        self.title = title
        self.language = language
        if blobs:
            self.blobs = blobs
        else:
            self.blobs = []
        self.predicted_annotations = []

    def add_blob(self, blob: Union[Blob, List[Blob]]):
        if isinstance(blob, list):
            self.blobs += blob
        else:
            self.blobs.append(blob)

    def add_predicted_annotations(self, predicted_annotation: List[PredictedAnnotation]):
        self.predicted_annotations += predicted_annotation

    def get_predicted_annotation_by_annotation_label(self, annotation_label):
        predicted_annotation = [predicted_annotation for predicted_annotation in self.predicted_annotations
                                if predicted_annotation.label == annotation_label]
        if predicted_annotation != []:
            return predicted_annotation
        else:
            return None
