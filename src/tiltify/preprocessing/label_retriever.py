from typing import List, Union
from tiltify.data_structures.blob import Blob
from tiltify.config import TILT_LABELS, SUPPORTED_LABELS
from tiltify.data_structures.annotation import Annotation


class LabelRetriever:

    def __init__(self, supported_labels: Union[List, None] = None) -> None:
        # TODO: training labers are different from prediction labels. Extractors are initialized with predict
        # labels. The mapping from train to predict happens here.
        self.train_label_mapper = {
            sup_label: train_label for sup_label, train_label in zip(SUPPORTED_LABELS, TILT_LABELS)}

        if supported_labels:
            supported_labels = [self.train_label_mapping[sup_label] for sup_label in supported_labels]
            self.tilt_labels_mapping = {tilt_label: idx for idx, tilt_label in enumerate(supported_labels)}
        else:
            self.tilt_labels_mapping = {tilt_label: idx for idx, tilt_label in enumerate(TILT_LABELS)}

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
                annotation = [
                    self.tilt_labels_mapping.get(annotation.label, 0) for annotation in annotation_list]
            else:
                annotation = [0]
            labels.append(annotation)
        return labels


if __name__ == "__main__":
    from tiltify.data_structures.document_collection import DocumentCollection

    label_retriever = LabelRetriever()
    document_collection = DocumentCollection.from_json_files()
    document = document_collection[0]

    labels = label_retriever.retrieve_labels(document.blobs)
