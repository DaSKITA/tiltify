import json
import os
import pytest
from os.path import dirname as dn

from tiltify.data_structures.annotation import Annotation
from tiltify.data_structures.blob import Blob

from tiltify.parsers.policy_parser import PolicyParser
from tiltify.config import Path






def test_parser(document_object, policy):
    parser = PolicyParser()
    parsed_document = parser.parse(*policy)
    test_document = document_object
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
