import os
import shutil

from tiltify.config import Path
from tiltify.extractors.extractor import Extractor, ExtractorManager
from tiltify.models.gaussian_nb_model import GaussianNBModel


def mock_init_extractors(self, extractor_config: dict) -> None:
    for model_type, labels in extractor_config:
        extraction_model_cls = self._model_registry.get(model_type)

        model_path = os.path.join(
            Path.root_path, f"test/src/tiltify/model_files/")

        extractor = Extractor(extraction_model_cls=extraction_model_cls, extractor_label=labels, model_path=model_path)
        self._extractor_registry.append(labels, extractor)


def test_extractor_train(document_collection_object):
    # prepare
    label = "Right to Withdraw Consent"
    model_base_path = os.path.join(Path.root_path, "test/src/tiltify/model_files/")
    ExtractorManager._init_extractors = mock_init_extractors
    extractor_manager = ExtractorManager([("GaussianNB", label)])

    # perform
    extractor_manager.train_online(document_collection_object, label)
    test_model = GaussianNBModel.load(model_base_path + "Extractor/Right to Withdraw Consent/", label).model

    # assert
    assert test_model.theta_ is not None

    # cleanup
    shutil.rmtree(model_base_path)
