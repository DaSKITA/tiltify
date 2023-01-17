from tiltify.extractors.extractor import Extractor


def test_extractor_predict(document_object):
    # prepare
    extractor = Extractor("Test")
    extractor.train()

    # perform
    predictions = extractor.predict(document_object)

    # assert
    assert predictions
    assert all([isinstance(prediction.blob_text, str) for prediction in predictions])
