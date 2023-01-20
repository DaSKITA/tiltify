import re
from typing import List
from tiltify.data_structures.annotation import PredictedAnnotation
from tiltify.data_structures.document import Document


class AnnotationShaper:

    def __init__(self, extractor) -> None:
        self.label = extractor.extractor_label

    def form_predict_annotations(self, indices: List[int], document: Document, bare_document: str) -> List[PredictedAnnotation]:
        predicted_annotations = list()
        for idx in indices:
            extracted_text = document.blobs[idx].text
            span = re.search(extracted_text, bare_document).span()
            predicted_annotation = PredictedAnnotation(
                text=extracted_text,
                label=self.label,
                start=span[0],
                end=span[1]-1
            )
            predicted_annotations.append(predicted_annotation)
        return predicted_annotations
