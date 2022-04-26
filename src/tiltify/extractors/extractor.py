from abc import ABCMeta, abstractmethod, ABC
import os
from typing import Any
from tiltify.data_structures.document import Document
from tiltify.config import Path


class ExtractorMetaClass(type):

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Metaclass Magic. Changes the instantiation of a class (not an object). After class creation default
        values for abstract attributes are set. If you need to set a new default value make sure to add a
        static method an exectute that method after class creation.
        Returns:
            Any: [description]
        """
        instance = super().__call__(*args, **kwargs)
        instance = cls.set_and_create_exp_dir(instance)
        return instance

    @staticmethod
    def set_and_create_exp_dir(instance):
        instance.exp_dir = os.path.join(Path.experiment_path, instance.__class__.__name__)
        if not os.path.exists(instance.exp_dir):
            os.makedirs(instance.exp_dir)
        return instance


class ExtractorCombinedMeta(ABCMeta, ExtractorMetaClass):
    """
    ABC Object generally use ABCMeta for
    initialization. The ExtractorMetaclass sets default values for every Class that inherits from Extractor as
    the initialization method of the class is changed again through this metaclass. The combined object is
    needed to meet the requirements of ABC and add additional functionality of ExtractorMetaClass."""
    pass


class Extractor(ABC, metaclass=ExtractorCombinedMeta):
    """Serves as an Interface for Extractors.

    Args:
        ABC (_type_): _description_
        metaclass (_type_, optional): _description_. Defaults to ExtractorCombinedMeta.
    """

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
