from torch.utils.data import random_split, Subset
from torch import Generator
from tiltify.config import RANDOM_SPLIT_SEED
from typing import Tuple
from tiltify.objectives.bert_objective.bert_preprocessor import TiltDataset, TiltFinetuningDataset


class BERTSplitter:

    def __init__(self) -> None:
        pass

    def split(self, preprocessed_data):
        pass

    def get_train_test_split(self, dataset: TiltDataset, n_test: float) -> Tuple[Subset, Subset]:
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
        test_size = round(n_test * len(dataset))
        train_size = len(dataset) - test_size
        # calculate the split
        train, test = random_split(
            dataset, [train_size, test_size], generator=Generator().manual_seed(RANDOM_SPLIT_SEED))
        return train, test

    def get_finetuning_datasets(
        self, tilt_dataset, split_ratio: float = 0.33,
            val: bool = False, binary: bool = False
            ) -> Tuple[TiltFinetuningDataset, TiltFinetuningDataset, TiltFinetuningDataset]:

        if split_ratio:
            train_ds, val_test_ds = self.get_train_test_split(tilt_dataset, split_ratio)
            train_ft_ds = TiltFinetuningDataset(train_ds)
            if val:
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
