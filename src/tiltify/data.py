from typing import List, Dict, Union

import pandas as pd
from torch.utils.data import Dataset, Subset
from torch import Tensor
from transformers import BertTokenizer, BatchEncoding


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

    def __init__(self, data):
        """
        Parameters
        ----------
        data : Dict
            a dict containing both the padded and tokenized
            sentences and the ids of those sentences
        """

        self.st = data['st']
        self.id = Tensor(data['id'])

        # add target values if existent
        try:
            self.labels = Tensor(data['labels'])
        except KeyError:
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

        item = {key: val[idx, :] for key, val in self.st.items()}
        item['id'] = self.id[idx]

        # add target values if existent
        if self.labels is not None:
            item['label'] = self.labels[idx]

        return item


class TiltFinetuningDataset(Dataset):
    """
      A dataset class used to wrap a TiltDataset
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


def tokenize_sentences(sentences: List[str], bert_base_model: str) -> BatchEncoding:
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

    bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model)
    return bert_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")


def get_dataset(dataset_file_path: str, bert_base_model: str) -> TiltDataset:
    """Loads a dataset file into a DataLoader object.
    
    A given CSV dataset file is read and tokenized. The processed
    contents are then loaded into a pytorch DataLoader object,
    which is returned in the end. 

    Parameters
    ----------
    dataset_file_path : str
        the path to a valid CSV dataset file.
        Check the README for more information
    bert_base_model : str
        specifies which huggingface transformers model is to be
        used for the tokenization 
    
    Returns
    -------
    TiltDataset
        dataset object which contains the padded and
        tokenized sentences
    
    """

    df_dataset = pd.read_csv(dataset_file_path)

    # retrieving the contents of the dataset
    sent_ids = list(df_dataset['id'])
    sentences = list(df_dataset['sentence'])

    # get target values if is training dataset
    try:
        labels = list(df_dataset['label'])
    except KeyError:
        labels = None

    sentences_tokenized = tokenize_sentences(sentences, bert_base_model)

    # saving the ids and tokenized sentences into the according dataset object
    test_data_dict = {'id': sent_ids, 'st': sentences_tokenized}

    # add target values if existent
    if labels:
        test_data_dict['labels'] = labels

    return TiltDataset(test_data_dict)

