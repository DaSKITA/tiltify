from tiltify.data_structures.blob import Blob


class Document:

    def __init__(self, title: str, blobs: Blob) -> None:
        """
        A Document is a list of blobs and represents a real document. In this case a privacy policy.
        It bears a title as well.

        Args:
            title (str): [description]
            blobs (Blob): [description]
        """
        self.title = title
        self.blobs = blobs
