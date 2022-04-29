from typing import Dict, List, Tuple

import numpy as np
import spacy
from torch.utils.data import Dataset

from tiltify.config import LABEL_REPLACE
from tiltify.preprocessor import Preprocessor
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.data_structures.blob import Blob


class W2VDataset(Dataset):
    """
    A dataset class used to load the contents of a CSV
    dataset file.

    ...

    Attributes
    ----------
    vec : np.ndarray
        a Numpy array containing all the sentence word2vec values of the dataset
    labels : np.ndaaray
        a Numpy array containing all the target values of the dataset

    Methods
    -------
    __len__()
        returns the length of the dataset
    __getitem__(idx)
        returns the tokenized sentence and the id of the
        sentence at index idx
    """

    def __init__(self, sentences: List, labels: List):
        """
        Parameters
        ----------
        sentences : List
            a List containing the word2vec embeddings
        labels : List
            a List containing the labels related to sentences
        """
        self.sentences = sentences
        self.labels = labels

    def __len__(self) -> int:
        """returns the length of the dataset"""
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict:
        """returns the tokenized sentence and the id of the sentence at index idx

        Parameters
        ----------
        idx : int
            specifies the index of the data to be returned
        """
        output_dict = dict(
            sentence=self.sentences[idx]
        )
        if self.labels is not None:
            output_dict["labels"] = self.labels[idx]
        return output_dict


class W2VPreprocessor(Preprocessor):

    def __init__(self, en: bool = False, binary: bool = False, remove_stopwords: bool = False):
        # requires prior install of spacy packages:
        # python -m spacy download en_core_web_lg
        # python -m spacy download de_core_news_lg
        if en:
            self.nlp = spacy.load("en_core_web_lg")
        else:
            self.nlp = spacy.load("de_core_news_lg")
        self.binary = binary
        self.remove_stopwords = remove_stopwords

    def preprocess(self, document_collection: DocumentCollection):
        w2v_dataset = self._create_w2v_dataset(document_collection)
        return w2v_dataset

    def _create_w2v_dataset(self, document_collection: DocumentCollection):
        sentence_list = []
        label_list = []
        # TODO: adjust this one as tokenized sentences appended might cause bugs, also adjust for per document preds
        for document in document_collection:
            sentences = document.blobs
            vectorized_sentences = self._vectorize_sentences(sentences)
            labels = self._get_labels(sentences)
            sentence_list.append(vectorized_sentences)
            label_list.append(self._prepare_labels(labels))
        return W2VDataset(sentence_list, label_list)

    def _vectorize_sentences(self, sentences: List[Blob]) -> List[np.ndarray]:
        """Tokenizes and pads a list of sentences using the model specified.

        Parameters
        ----------
        sentences : List[str]
            a list containing all sentences that are to be vectorized
            in this function

        Returns
        -------
        List[np.ndarray]
            List of Numpy arrays containing word2vec vectors

        """
        vectors = []
        for blob in sentences:
            sentence = self.nlp(blob.text)
            if self.remove_stopwords:
                sentence = self.nlp(" ".join([str(word) for word in sentence if str(word) not in self.nlp.Defaults.stop_words]))
            vectors.append(sentence.vector)
        return vectors

    def _get_labels(self, document_blobs: List[Blob]) -> List[str]:
        # TODO: case where one blob has multiple annotations -> needs to be accessed in the model
        labels = [blob.get_annotations()[0] if blob.get_annotations() else None for blob in document_blobs]
        return labels

    def _prepare_labels(self, labels: List):
        if self.binary:
            label_data = [0 if entry is None else 1 for entry in labels] if labels else None
        else:
            label_data = [LABEL_REPLACE[entry] for entry in labels] if labels else None
        return label_data
