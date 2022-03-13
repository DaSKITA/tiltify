from typing import Dict, List
from torch.utils.data import Dataset
from torch import Tensor
from transformers import BertTokenizer, BatchEncoding
from collections import defaultdict
import pandas as pd
import random

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
        return 100# len(self.id)

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


class BERTPreprocessor(Preprocessor):

    def __init__(
        self, bert_model: str = None, binary: bool = False, n_upsample: float = None,
            n_downsample: float = None) -> None:
        if bert_model:
            self.bert_model = bert_model
        else:
            self.bert_model = BASE_BERT_MODEL
        self.binary = binary
        self.n_upsample = n_upsample
        self.n_downsample = n_downsample

    def preprocess(self, document_collection: DocumentCollection):
        tilt_dataset = self._create_tilt_dataset(document_collection)
        return tilt_dataset

    def preprocess_pandas(self, pandas_path):
        """This function needs to be deleted when pandas is not needed anymore.

        Args:
            pandas_path (_type_): _description_
        """
        df_dataset = pd.read_csv(pandas_path)

        # retrieving the contents of the dataset
        sent_ids = list(df_dataset['id'])
        sentences = list(df_dataset['sentence'])

        # get target values if is training dataset
        try:
            labels = list(df_dataset['label'])
        except KeyError:
            labels = None

        sentences_tokenized = self._tokenize_sentences(sentences)

        # saving the ids and tokenized sentences into the according dataset object
        test_data_dict = {'id': sent_ids, 'sentences': sentences_tokenized}

        # add target values if existent
        if labels:
            test_data_dict['labels'] = labels

        if self.n_upsample:
            sentences, labels = self.upsample(sentences, labels)
        if self.n_downsample:
            sentences, labels = self.downsample(sentences, labels)
        return TiltDataset(test_data_dict, binary=self.binary)

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

    def upsample(self, sentences, labels):
        minority = [idx for idx, label in enumerate(labels) if label == 1]
        upsampled_minorities = random.choices(minority, k=int(self.n_upsample * len(minority)))
        sentences += [sentences[idx] for idx in upsampled_minorities]
        labels += [labels[idx] for idx in upsampled_minorities]
        return sentences, labels

    def downsample(self, sentences, labels):
        majority = [idx for idx, label in enumerate(labels) if label == 0]
        minority = [idx for idx, label in enumerate(labels) if label == 1]
        downsampled_majorities = random.choices(majority, k=int(self.n_downsample* len(majority)))
        sentences = [sentences[idx] for idx in downsampled_majorities] + [sentences[idx] for idx in minority]
        labels = [labels[idx] for idx in downsampled_majorities] + [labels[idx] for idx in minority]
        return sentences, labels
