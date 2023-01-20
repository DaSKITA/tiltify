from tiltify.extractors.extractor import Extractor


def test_extractor_predict(document_object):
    # prepare
    extractor = Extractor("Test", "Test_Right")
    extractor.train()

    # perform
    predictions = extractor.predict(document_object)

    # assert
    assert predictions
