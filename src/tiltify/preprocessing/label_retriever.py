from typing import List, Union
from tiltify.data_structures.blob import Blob
from tiltify.config import LABEL_REPLACE
from tiltify.data_structures.annotation import Annotation


class LabelRetriever:

    def __init__(self) -> None:
        pass

    def retrieve_labels(self, document_blobs: List[Blob]):
        annotations = self.get_annotations(document_blobs)
        labels = self.map_annotations(annotations)
        return labels

    def get_annotations(self, document_blobs: List[Blob]) -> List[Union[List[Annotation], None]]:
        return [blob.get_annotations() for blob in document_blobs]

    def map_annotations(self, annotations: List[Union[List[Annotation], None]]):
        """Maps Annotations that are supported in the config.py to a Label. A Label is a list of Integers.
        Everything that can not be mapped is automatically written as [0]

        Args:
            annotations (List[Union[List[Annotation], None]]): _description_

        Returns:
            _type_: _description_
        """
        labels = []
        for annotation_list in annotations:
            if annotation_list:
                annotation = [LABEL_REPLACE.get(annotation.label, 0) for annotation in annotation_list]
            else:
                annotation = [0]
            labels.append(annotation)
        return labels
