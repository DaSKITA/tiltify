import json
import os
import pytest
from os.path import dirname as dn

from tiltify.data_structures.blob import Blob
from tiltify.data_structures.document import Document
from tiltify.parsers.policy_parser import PolicyParser


@pytest.fixture
def document_object():
    print(dn(dn(__file__)))
    with open(os.path.join(dn(dn(__file__)), 'data/dhl_blobs.json'), 'r') as file:
        data = json.load(file)
    return data['name'], [Blob(blobs) for blobs in data['blobs']]


@pytest.fixture
def policy():
    with open(os.path.join(dn(dn(__file__)), 'data/dhl.json'), 'r') as file:
        data = json.load(file)
    return data['name'], data['text']


def test_parser(document_object, policy):
    parser = PolicyParser()
    parsed_document = parser.parse(*policy)
    test_document = Document(*document_object)
    assert (parsed_document == test_document)