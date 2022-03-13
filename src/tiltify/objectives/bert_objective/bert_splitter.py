from torch.utils.data import random_split, Subset
from torch import Generator
from tiltify.config import RANDOM_SPLIT_SEED
from typing import Tuple, Union, Dict
from tiltify.objectives.bert_objective.bert_preprocessor import TiltDataset
from torch.utils.data import Dataset


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


class BERTSplitter:

    def __init__(self, split_ratio: float = None, val: bool = True) -> None:
        self.split_ratio = split_ratio
        self.val = val

    def split(self, preprocessed_data):
        return self.get_finetuning_datasets(preprocessed_data)

    def get_train_test_split(self, dataset: TiltDataset, split_ratio: float = None) -> Tuple[Subset, Subset]:
        # TODO: splits need to be handled outside of the preprocessor
        """Get a train/test split for the given dataset

        Parameters
        ----------
        dataset : TiltDataset
            a dataset containing tokenized sentences
        n_test : float
            the ratio of the items in the dataset to be used for evaluation
            """

        # determine sizes
        test_size = round(split_ratio * len(dataset))
        train_size = len(dataset) - test_size
        # calculate the split
        train, test = random_split(
            dataset, [train_size, test_size], generator=Generator().manual_seed(RANDOM_SPLIT_SEED))
        return train, test

    def get_finetuning_datasets(
            self, tilt_dataset) -> Tuple[TiltFinetuningDataset, TiltFinetuningDataset, TiltFinetuningDataset]:

        if self.split_ratio:
            train_ds, val_test_ds = self.get_train_test_split(tilt_dataset, self.split_ratio)
            train_ft_ds = TiltFinetuningDataset(train_ds)
            if self.val:
                val_ds, test_ds = self.get_train_test_split(val_test_ds, 0.5)
                val_ft_ds = TiltFinetuningDataset(val_ds)
                test_ft_ds = TiltFinetuningDataset(test_ds)
            else:
                val_ft_ds = None
                test_ft_ds = TiltFinetuningDataset(val_test_ds)
        else:
            train_ft_ds = TiltFinetuningDataset(tilt_dataset)
            val_ft_ds = None
            test_ft_ds = None

        return train_ft_ds, val_ft_ds, test_ft_ds
