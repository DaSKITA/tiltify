import json
from tiltify.parsers.policy_parser import PolicyParser
from tiltify.data_structures.document import Document
from tiltify.data_loader import DataLoader
from typing import List


class DocumentCollection:

    json_parser = PolicyParser()
    data_loader = DataLoader()

    def __init__(self, documents: List[Document]) -> None:
        """This class provides policies in form of an iterator.
        Every Policy is represented in a Document Class. For later usage it is also possible
        to adjust the loading of each individual document. Therefore the usage of an iterator is preferred.

        Args:
            documents (List[Document]): _description_
        """
        self.documents = documents
        self.index = 0

    @classmethod
    def from_json_files(cls, folder_name: str = "annotated_policies", language: str = "de"):
        """Creates a Document Collection from Json Policy files.

        Args:
            folder_name (str, optional): _description_. Defaults to "annotated_policies".

        Returns:
            _type_: _description_
        """
        # Annotations are not parsed
        json_policies = cls.data_loader.get_json_policies(folder_name)
        json_policies = [
            json_policy for json_policy in json_policies if json_policy["language"] == language]
        document_list = [
            cls.json_parser.parse(**json_policy["document"], annotations=json_policy["annotations"])
            for json_policy in json_policies]
        return cls(document_list)

    @classmethod
    def from_pickle_files(self):
        pass

    def __next__(self) -> Document:
        try:
            document = self.documents[self.index]
        except IndexError:
            raise StopIteration()
        self.index += 1
        return document

    def __getitem__(self, key) -> Document:
        if isinstance(key, list):
            selected_collection = []
            for idx in key:
                selected_collection.append(self.documents[idx])
            return selected_collection
        else:
            return self.documents[key]

    def __iter__(self):
        """Necessary to identify it as an Iterable.

        Returns:
            _type_: _description_
        """
        return self

    def __len__(self):
        return len(self.documents)
