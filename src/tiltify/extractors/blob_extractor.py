from abc import ABC, abstractmethod
import os
from typing import Any
from tiltify.data_structures.document import Document
from tiltify.config import Path


class BlobExtractor(ABC):

    exp_dir = None

    @abstractmethod
    def predict(self, document: Document):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load(self):
        pass


class BlobExtractorMetaClass(type):

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Metaclass Magic, have to revisit how I did that.
        Returns:
            Any: [description]
        """
        instance = super().__call__(*args, **kwargs)
        instance = cls.set_and_create_exp_dir(instance)
        return instance

    @staticmethod
    def set_and_create_exp_dir(instance):
        instance.exp_dir = os.path.join(Path.exp_dir, instance.__class__.__name__)
        return instance
