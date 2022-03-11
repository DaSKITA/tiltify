from typing import Dict, List, Tuple, Union
import pandas as pd
from torch.utils.data import Dataset, Subset, random_split
from torch import Generator, Tensor
from transformers import BertTokenizer, BatchEncoding

from tiltify.config import LABEL_REPLACE, RANDOM_SPLIT_SEED, Path


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

        self.st = data['st']
        self.id = Tensor(data['id'])

        # add target values if existent
        try:
            if binary:
                label_data = [0 if entry == "None" else 1 for entry in data['labels']]
            else:
                label_data = [LABEL_REPLACE[entry] for entry in data['labels']]
            self.labels = Tensor(label_data).long()
        except KeyError:
            self.labels = None
            print('KeyError in labels creation')

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


def get_dataset(dataset_file_path: str, bert_base_model: str, binary: bool) -> TiltDataset:
    """Loads a dataset file into a TiltDataset object.

    A given CSV dataset file is read and tokenized. The processed
    contents are then loaded into a pytorch TiltDataset object,
    which is returned in the end.

    Parameters
    ----------
    dataset_file_path : str
        the path to a valid CSV dataset file.
        Check the README for more information
    bert_base_model : str
        specifies which huggingface transformers model is to be
        used for the tokenization
    binary : bool
        decides if the dataset classes should differentiate between different 'Right To's or not

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

    return TiltDataset(test_data_dict, binary=binary)


def get_train_test_split(dataset: TiltDataset, n_test: float) -> Tuple[Subset, Subset]:
    """Get a train/test split for the given dataset

    Parameters
    ----------
    dataset : TiltDataset
        a dataset containing tokenized sentences
    n_test : float
        the ratio of the items in the dataset to be used for evaluation
    """

    # determine sizes
    test_size = round(n_test * len(dataset))
    train_size = len(dataset) - test_size
    # calculate the split
    train, test = random_split(dataset, [train_size, test_size],
                               generator=Generator().manual_seed(RANDOM_SPLIT_SEED))
    return train, test


def get_finetuning_datasets(dataset_file_path: str, bert_base_model: str, split_ratio: float = 0.33, val: bool = False,
                            binary: bool = False) -> Tuple[TiltFinetuningDataset, TiltFinetuningDataset,
                                                           TiltFinetuningDataset]:
    dataset = get_dataset(dataset_file_path, bert_base_model, binary=binary)

    if split_ratio:
        train_ds, val_test_ds = get_train_test_split(dataset, split_ratio)
        train_ft_ds = TiltFinetuningDataset(train_ds)
        if val:
            val_ds, test_ds = get_train_test_split(val_test_ds, 0.5)
            val_ft_ds = TiltFinetuningDataset(val_ds)
            test_ft_ds = TiltFinetuningDataset(test_ds)
        else:
            val_ft_ds = None
            test_ft_ds = TiltFinetuningDataset(val_test_ds)
    else:
        train_ft_ds = TiltFinetuningDataset(dataset)
        val_ft_ds = None
        test_ft_ds = None

    return train_ft_ds, val_ft_ds, test_ft_ds
