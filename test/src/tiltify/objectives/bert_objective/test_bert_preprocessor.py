import pytest
from tiltify.objectives.bert_objective.bert_preprocessor import BERTPreprocessor


@pytest.fixture
def text_example():
    n_examples = 10
    sentences = ["string" for _ in range(n_examples)]
    labels = [0] * int(n_examples/2) + [1] * int(n_examples/2)
    return sentences, labels


class TestBERTPreprocessor:

    def test_upsample(self, text_example):
        sentences, labels = text_example
        n_examples = len(sentences)
        preprocessor = BERTPreprocessor(n_upsample=0.5)

        sentences, labels = preprocessor.upsample(sentences, labels)

        # assert
        assert len(sentences) > n_examples
        assert len(labels) > n_examples

    def test_downsample(self, text_example):
        sentences, labels = text_example
        n_examples = len(sentences)
        preprocessor = BERTPreprocessor(n_downsample=0.5)

        sentences, labels = preprocessor.downsample(sentences, labels)

        # assert
        assert len(sentences) < n_examples
        assert len(labels) < n_examples

