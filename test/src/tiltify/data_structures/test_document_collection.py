from typing import Iterator
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.data_structures.document import Document


class TestDocumentCollection:

    def test_from_json_files(self):
        # set up
        document_collection = DocumentCollection.from_json_files("test_data/policies")

        # assert
        assert isinstance(document_collection, DocumentCollection)
        assert isinstance(document_collection, Iterator)
        assert document_collection[0] is not None
        assert isinstance([document for document in document_collection][0], Document)
