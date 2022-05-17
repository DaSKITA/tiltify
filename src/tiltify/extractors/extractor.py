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


class ModelCombinedMeta(ABCMeta, ExtractorMetaClass):
    """
    ABC Objects generally use ABCMeta for
    initialization. The ExtractorMetaclass sets default values for every Class that inherits from Extractor as
    the initialization method of the class is changed again through this metaclass. The combined object is
    needed to meet the requirements of ABC and add additional functionality of ExtractorMetaClass."""
    pass


class Extractor:

    model_registry = {

    }

    def __init__(self, model_str: str = None) -> None:
        model_cls = self.model_registry[model_str]  # use an enum
        self.exp_dir = os.path.join(Path.experiment_path, model_cls.__name__)

    def extract(self, document: Document) -> Document:
        predicted_doc = self.model.predict(document)
        return predicted_doc

    def train(self):
        # happens on another machine
        pass
