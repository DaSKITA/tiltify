import pytest
from tiltify.preprocessing.bert_preprocessor import BERTPreprocessor


@pytest.fixture
def text_example():
    n_examples = 10
    sentences = ["string" for _ in range(n_examples)]
    labels = [0] * int(n_examples/2) + [1] * int(n_examples/2)
    return sentences, labels


class TestBERTPreprocessor:
    pass
