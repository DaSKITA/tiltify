from tiltify.extractors.extractor import Extractor
from tiltify.annotation_shaper import AnnotationShaper


def test_extractor_predict(document_object, policy):
    # prepare
    extractor = Extractor("Test", "Test_Right")
    bare_text = policy[1]
    annotation_shaper = AnnotationShaper(extractor)
    extractor.train()

    # perform
    predictions = extractor.predict(document_object)
    predicted_annotations = annotation_shaper.form_predict_annotations(
        predictions, document_object, bare_text)

    # assert
    assert all([pred_annot.start is not None for pred_annot in predicted_annotations])
