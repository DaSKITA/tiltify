import json
import os
import pytest
from os.path import dirname as dn

from tiltify.data_structures.annotation import Annotation
from tiltify.data_structures.blob import Blob
from tiltify.data_structures.document import Document
from tiltify.parsers.policy_parser import PolicyParser
from tiltify.config import Path


@pytest.fixture
def document_object():
    with open(os.path.join(Path.root_path, 'data/test_data/carrefour_document_object.json'), 'r') as file:
        data = json.load(file)
    return data['document']['document_name'], [Blob(blob["text"],
                                                    [Annotation(annotation['annotation_text'],
                                                                annotation['annotation_label'],
                                                                annotation['annotation_start'],
                                                                annotation['annotation_end'])
                                                     for annotation in blob["annotations"]])
                                               for blob in data['document']['blobs']]


@pytest.fixture
def policy():
    with open(os.path.join(Path.root_path, 'data/test_data/carrefour_document_annotations.json'), 'r') as file:
        data = json.load(file)
    return data['document']['document_name'], data['document']['text'], data['annotations']


def test_parser(document_object, policy):
    parser = PolicyParser()
    parsed_document = parser.parse(*policy)
    test_document = Document(*document_object)
    equality = True

    if parsed_document.title == test_document.title and len(parsed_document.blobs) == len(test_document.blobs):
        for parsed_blob, test_blob in zip(parsed_document.blobs, test_document.blobs):
            if parsed_blob.text != test_blob.text or len(parsed_blob.annotations) != len(test_blob.annotations):
                equality = False
            for parsed_annotation, test_annotation in zip(parsed_blob.annotations, test_blob.annotations):
                if parsed_annotation.annotated_text != test_annotation.annotated_text \
                        or parsed_annotation.label != test_annotation.label \
                        or parsed_annotation.start != test_annotation.start \
                        or parsed_annotation.end != test_annotation.end:
                    equality = False
    else:
        equality = False

    assert equality
