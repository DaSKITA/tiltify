from typing import Dict, List, Union
from torch.utils.data import Dataset, Subset
from torch import Tensor
from transformers import BertTokenizer, BatchEncoding
from collections import defaultdict

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

    def __init__(self, data, binary=False):
        """
        Parameters
        ----------
        data : Dict
            a dict containing both the padded and tokenized
            sentences and the ids of those sentences
        """

        self.sentences = data['sentences']
        self.id = Tensor(data['id'])

        # add target values if existent
        if binary:
            label_data = [0 if entry == "None" else 1 for entry in data['labels']]
        else:
            label_data = [LABEL_REPLACE[entry] for entry in data['labels']]
        if label_data != []:
            self.labels = Tensor(label_data).long()
        else:
            self.labels = None

    def __len__(self) -> int:
        """returns the length of the dataset"""
        return len(self.id)

    def __getitem__(self, idx) -> Dict:
        """returns the tokenized sentence and the id of the sentence at index idx

        Parameters
        ----------
        idx : int
            specifies the index of the data to be returned
        """

        item = {key: val[idx, :] for key, val in self.sentences.items()}
        item['id'] = self.id[idx]

        # add target values if existent
        if self.labels is not None:
            item['label'] = self.labels[idx]

        return item


class TiltFinetuningDataset(Dataset):
    """ A dataset class used to wrap a TiltDataset
        for fine-tuning a transformers model with the Trainer API.
        Does not return sentence ids when indexed.
        ...

        Attributes
        ----------
        dataset : Dataset


        Methods
        -------
        __len__()
            returns the length of the dataset
        __getitem__(idx)
            returns the tokenized sentence and at index idx
    """

    def __init__(self, dataset: Union[TiltDataset, Subset]):
        """
        Parameters
        ----------
        dataset : Subset
            a TiltDataset (or subset) which should be wrapped
        """
        self.dataset = dataset

    def __len__(self) -> int:
        """returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        """returns the tokenized sentence at index idx

        Parameters
        ----------
        idx : int
            specifies the index of the data to be returned
        """
        item = self.dataset[idx]
        # remove id entry
        item.pop('id', None)
        return item


class BERTPreprocessor(Preprocessor):

    def __init__(
        self, bert_model: str = None, binary: bool = False, upsample: float = None,
            downsample: float = None) -> None:
        if bert_model:
            self.bert_model = bert_model
        else:
            self.bert_model = BASE_BERT_MODEL
        self.binary = binary
        self.upsample = upsample
        self.downsample = downsample

    def preprocess(self, document_collection: DocumentCollection):
        tilt_dataset = self._create_tilt_dataset(document_collection)
        return tilt_dataset

    def _create_tilt_dataset(self, document_collection: DocumentCollection):
        dataset_dict = defaultdict(list)
        for document in document_collection:
            labels = self._get_labels(document.blobs)
            sentences = document.blobs
            if self.upsample:
                sentences, labels = self.upsample(sentences, labels)
            if self.downsample:
                sentences, labels = self.downsample(sentences, labels)
            tokenized_sentences = self._tokenize_sentences(sentences)

            dataset_dict["sentences"].append(tokenized_sentences)
            dataset_dict["labels"].append(labels)
        dataset_dict["id"] = [idx for idx in range(len(dataset_dict["sentence"]))]
        return TiltDataset(dataset_dict, binary=self.binary)

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

    def upsample(self):
        pass

    def downsample(self):
        pass
