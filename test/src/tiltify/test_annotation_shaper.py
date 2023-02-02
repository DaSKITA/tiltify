from tiltify.extractors.extractor import Extractor
from tiltify.annotation_shaper import AnnotationShaper


def test_extractor_predict(document_object, policy):
    # prepare
    label = "Test_Right"
    extractor = Extractor("Test", "Test_Right")
    bare_text = policy[1]
    annotation_shaper = AnnotationShaper(label)
    extractor.train()

    # perform
    predictions = extractor.predict(label, document_object, bare_text)
    # assert
    assert all([pred_annot.start is not None for pred_annot in predictions])
