from typing import Dict, List
import spacy
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from tiltify.preprocessing.preprocessor import Preprocessor
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.data_structures.blob import Blob
from tiltify.preprocessing.label_retriever import LabelRetriever
from tiltify.data_structures.document import Document


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
            a (n, 300) List containing the word2vec embeddings as torch.Tensor
        labels : List
            a (n, 1) List containing the labels related to the sentence embeddings
        """
        assert len(sentences) == len(labels)
        self.sentences = torch.stack([torch.from_numpy(array) for array in sentences])
        self.labels = torch.Tensor(labels)

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
        return dict(
            sentences=self.sentences[idx],
            labels=self.labels[idx]
        )


class W2VPreprocessor(Preprocessor):

    def __init__(
            self, en: bool = False, binary: bool = False, remove_stopwords: bool = False,
                label: list = None, weighted_sampling: bool = True, batch_size: int = 25):
        # requires prior install of spacy packages:
        # python -m spacy download en_core_web_lg
        # python -m spacy download de_core_news_lg
        # TODO: this is not glove (https://spacy.io/models/en)
        if en:
            self.nlp = spacy.load("en_core_web_lg")
        else:
            self.nlp = spacy.load("de_core_news_lg")
        self.binary = binary
        self.remove_stopwords = remove_stopwords
        self.label_retriever = LabelRetriever(supported_label=label)
        self.weighted_sampling = weighted_sampling
        self.batch_size = batch_size

    def preprocess(self, document_collection: DocumentCollection):
        sentence_list = []
        label_list = []
        for document in tqdm(document_collection):
            embeddings, labels = self.preprocess_document(document)
            sentence_list.append(embeddings)
            label_list.append(labels)

        # flatten lists
        sentence_list = sum(sentence_list, [])
        label_list = sum(label_list, [])
        dataset = W2VDataset(sentence_list, label_list)
        data_loader = self.create_dataloader(dataset, weighted_sampling=self.weighted_sampling)
        return data_loader

    def preprocess_document(self, document: Document):
        labels = None
        document_blobs = document.blobs
        vectorized_sentences = self._vectorize_document(document_blobs)
        labels = self.label_retriever.retrieve_labels(document.blobs)
        labels = self.prepare_labels(labels)
        return vectorized_sentences, labels

    def _vectorize_document(self, document_blobs: List[Blob]) -> torch.Tensor:
        """Tokenizes and pads a list of sentences using spacy.

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
        for blob in document_blobs:
            blob_text = self.nlp(blob.text)
            if self.remove_stopwords:
                blob_text = self.nlp(" ".join([str(word) for word in blob_text if str(word) not in self.nlp.Defaults.stop_words]))
            vectors.append(blob_text.vector)
        return vectors

    def prepare_labels(self, labels: List):
        if self.binary:
            labels = [int(label[0] > 0) for label in labels]
        return labels

    def create_dataloader(self, dataset, weighted_sampling=True) -> DataLoader:
        sampler = None
        if weighted_sampling:
            sampler = self._create_sampler(dataset)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=sampler)
        return loader

    def _create_sampler(self, dataset: Dataset) -> WeightedRandomSampler:
        class_weights = [
            1 / torch.sum(dataset.labels == label).float() for label in dataset.labels.unique(sorted=True)]
        sample_weights = torch.Tensor([class_weights[int(t)] for t in dataset.labels])
        return WeightedRandomSampler(sample_weights, num_samples=sample_weights.shape[0])
