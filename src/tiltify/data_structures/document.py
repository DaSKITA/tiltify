from typing import List, Union

from tiltify.data_structures.blob import Blob


class Document:

    def __init__(self, title: str, blobs: List[Blob] = None) -> None:
        """
        A Document is a list of blobs and represents a real document. In this case a privacy policy.
        It bears a title as well.

        Args:
            title (str): [description]
            blobs (Blob): [description]
        """
        self.title = title
        if blobs:
            self.blobs = blobs
        else:
            self.blobs = []

    def add_blob(self, blob: Union[Blob, List[Blob]]):
        if isinstance(blob, list):
            self.blobs += blob
        else:
            self.blobs.append(blob)
