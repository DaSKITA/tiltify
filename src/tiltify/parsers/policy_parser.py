from typing import List

from tiltify.parsers.parser import Parser
from tiltify.data_structures.annotation import Annotation
from tiltify.data_structures.blob import Blob
from tiltify.data_structures.document import Document


class ParserFilter:

    def __init__(self, symbols: List[str] = None) -> None:
        self.filter_list = ['\t', '\b']
        if symbols:
            self.filter_list.append(symbols)
        pass

    def filter(self, text: str) -> str:
        for symbols in self.filter_list:
            text.replace(symbols, '')
        return text


class PolicyParser(Parser):

    def __init__(self) -> None:
        self.filter = ParserFilter()

    def parse(self, document_name: str, text: str, annotations: List = None, language: str = None) -> Document:
        if not annotations:
            annotations = []
        processed_text = self.filter.filter(text)
        blobs_raw = [blob for blob in processed_text.split('\n') if blob != '']
        blobs_annotated = []

        # iterate over all blobs and pair them with their annotations
        for blob in blobs_raw:
            blob_annotations = []
            blob_start = len(text.split(blob)[0]) + 1
            blob_end = blob_start + len(blob) - 1
            for annotation in annotations:
                # gather annotations that belong to the current blob
                if blob_start <= annotation['annotation_start'] <= blob_end or \
                        blob_start <= annotation['annotation_end'] <= blob_end or \
                        annotation['annotation_start'] <= blob_start <= annotation['annotation_end'] or \
                        annotation['annotation_start'] <= blob_end <= annotation['annotation_end']:
                    blob_annotations.append(annotation)

            # zip the blob with gathered annotations
            blobs_annotated.append((blob, blob_annotations))

        # generate a List of Blob objects containing their responding annotations
        blobs = [Blob(blob_annotated[0],
                      [Annotation(annotation['annotation_text'],
                                  annotation['annotation_label'],
                                  annotation['annotation_start'],
                                  annotation['annotation_end'])
                       for annotation in blob_annotated[1]])
                 for blob_annotated in blobs_annotated]

        # create a Document object from the Blob objects
        doc = Document(document_name, blobs, language)
        return doc
