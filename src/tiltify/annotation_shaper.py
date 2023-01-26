from typing import List
from tiltify.data_structures.annotation import PredictedAnnotation
from tiltify.data_structures.document import Document


class AnnotationShaper:

    def __init__(self, label) -> None:
        self.label = label

    def form_predict_annotations(self, indices: List[int], document: Document, bare_document: str) -> List[PredictedAnnotation]:
        predicted_annotations = list()
        for idx in indices:
            try:
                extracted_text = document.blobs[idx].text
                start = bare_document.find(extracted_text)
            except AttributeError:
                print(extracted_text)
            predicted_annotation = PredictedAnnotation(
                text=extracted_text,
                # TODO: this only works if one label is supplied. For multiple labels this will fail
                label=self.label,
                start=start,
                end=start+len(extracted_text)
            )
            predicted_annotations.append(predicted_annotation)
        if predicted_annotations == []:
            predicted_annotations.append(PredictedAnnotation())
        return predicted_annotations
