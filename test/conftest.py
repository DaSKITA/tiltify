import pytest
import json
import os
from tiltify.data_structures.blob import Blob
from tiltify.data_structures.document import Document
from tiltify.data_structures.annotation import Annotation
from tiltify.config import Path


@pytest.fixture
def document_object():
    with open(os.path.join(Path.root_path, 'data/test_data/document_object/carrefour_document_object.json'), 'r') as file:
        data = json.load(file)
    return Document(data['document']['document_name'], [Blob(blob["text"],
                                                    [Annotation(annotation['annotation_text'],
                                                                annotation['annotation_label'],
                                                                annotation['annotation_start'],
                                                                annotation['annotation_end'])
                                                     for annotation in blob["annotations"]])
                                               for blob in data['document']['blobs']])
