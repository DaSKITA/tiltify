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
            try:
                start = bare_document.find(extracted_text)
            except AttributeError:
                print(extracted_text)
            predicted_annotation = PredictedAnnotation(
                text=extracted_text,
                label=self.label,
                start=start,
                end=start+len(extracted_text)
            )
            predicted_annotations.append(predicted_annotation)
        return predicted_annotations
