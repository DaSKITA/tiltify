from typing import Dict, List
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer, BatchEncoding
import pandas as pd

from tiltify.config import LABEL_REPLACE, BASE_BERT_MODEL
from tiltify.preprocessor import Preprocessor
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.data_structures.blob import Blob


class TiltDataset(Dataset):
    """
    A dataset class used to load the contents of a CSV
    dataset file.

    ...

    Attributes
    ----------
    st : BatchEncoding
        huggingface transformers data structure which contains
        the padded and tokenized sentences
    id : Tensor
        a Tensor containing all the sentence ids related to the
        padded and tokenized sentences in st
    labels : Tensor
        a Tensor containing all the target values related to the
        padded and tokenized sentences in st

    Methods
    -------
    __len__()
        returns the length of the dataset
    __getitem__(idx)
        returns the tokenized sentence and the id of the
        sentence at index idx
    """

    def __init__(self, sentences: Dict, labels: List):
        """
        Parameters
        ----------
        data : Dict
            a dict containing both the padded and tokenized
            sentences and the ids of those sentences
        """
        self.input_ids = sentences["input_ids"]
        self.attention_mask = sentences["attention_mask"]
        self.token_type_ids = sentences["token_type_ids"]
        self.labels = torch.Tensor(labels)

    def __len__(self) -> int:
        """returns the length of the dataset"""
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        """returns the tokenized sentence and the id of the sentence at index idx

        Parameters
        ----------
        idx : int
            specifies the index of the data to be returned
        """
        output_dict = dict(
            input_ids=self.input_ids[idx],
            attention_mask=self.attention_mask[idx],
            token_type_ids=self.token_type_ids[idx]
        )
        if self.labels is not None:
            output_dict["labels"] = self.labels[idx]
        return output_dict


class BERTPreprocessor(Preprocessor):

    def __init__(
            self, bert_model: str = None, binary: bool = False) -> None:
        if bert_model:
            self.bert_model = bert_model
        else:
            self.bert_model = BASE_BERT_MODEL
        self.binary = binary

    def preprocess(self, document_collection: DocumentCollection):
        tilt_dataset = self._create_tilt_dataset(document_collection)
        return tilt_dataset

    def preprocess_pandas(self, pandas_path: str):
        """
        This function needs to be deleted when pandas is not needed anymore.
        Args:
            pandas_path (_type_): _description_
        """
        df_dataset = pd.read_csv(pandas_path)
        sentences = list(df_dataset['sentence'])
        # get target values if is training dataset
        labels = df_dataset.get("label", None).to_list()
        if labels == []:
            labels = None
        labels = self._prepare_labels(labels)
        sentences_tokenized = self._tokenize_sentences(sentences)
        return TiltDataset(sentences=sentences_tokenized, labels=labels)

    def _create_tilt_dataset(self, document_collection: DocumentCollection):
        sentences = []
        labels = []
        # TODO: adjust this one as tokenized sentences appended might cause bugs, also adjust for per document preds
        for document in document_collection:
            labels = self._get_labels(document.blobs)
            sentence = document.blobs
            tokenized_sentence = self._tokenize_sentences(sentence)
            sentences.append(tokenized_sentence)
            labels.append(labels)
        return TiltDataset(sentences, labels)

    def _tokenize_sentences(self, sentences: List[str]) -> BatchEncoding:
        """Tokenizes and pads a list of sentences using the model specified.

        Parameters
        ----------
        sentences : List[str]
            a list containing all sentences that are to be tokenized
            in this function
        bert_base_model : str
            specifies which huggingface transformers model is to be
            used for the tokenization

        Returns
        -------
        BatchEncoding
            huggingface transformers data structure which contains
            the padded and tokenized sentences

        """

        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        return bert_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    def _get_labels(self, document_blobs: List[Blob]) -> List[int]:
        # TODO: case where one blob has multiple annotations -> needsto be accessed in the model
        labels = [blob.get_annotations()[0] for blob in document_blobs]
        return labels

    def _prepare_labels(self, labels: List):
        if self.binary:
            label_data = [0 if entry == "None" else 1 for entry in labels]
        else:
            label_data = [LABEL_REPLACE[entry] for entry in labels]
        return label_data
