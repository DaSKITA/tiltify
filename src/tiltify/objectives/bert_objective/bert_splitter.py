from torch.utils.data import random_split, Subset
from torch import Generator
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tiltify.objectives.bert_objective.bert_preprocessor import TiltDataset
from typing import Tuple


class BERTSplitter:

    def __init__(self, split_ratio: float = None, val: bool = True, batch_size: int = None) -> None:
        self.split_ratio = split_ratio
        self.val = val
        if batch_size:
            self.batch_size = batch_size

    def split(self, preprocessed_data: TiltDataset):
        set_list = self.get_finetuning_datasets(preprocessed_data)
        sampler = self._create_sampler(preprocessed_data)
        for split_set in set_list:
            if split_set:
                split_set = self._create_dataloader(split_set, sampler)
        return set_list

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
            dataset, [train_size, test_size], generator=Generator())
        return train, test

    def get_finetuning_datasets(
            self, tilt_dataset: TiltDataset) -> Tuple[TiltDataset, TiltDataset, TiltDataset]:

        if self.split_ratio:
            train_ds, val_test_ds = self.get_train_test_split(tilt_dataset, self.split_ratio)
            train_ft_ds = train_ds
            if self.val:
                val_ds, test_ds = self.get_train_test_split(val_test_ds, 0.5)
                val_ft_ds = val_ds
                test_ft_ds = test_ds
            else:
                val_ft_ds = None
                test_ft_ds = val_test_ds
        else:
            train_ft_ds = tilt_dataset
            val_ft_ds = None
            test_ft_ds = None
        return [train_ft_ds, val_ft_ds, test_ft_ds]

    def _create_dataloader(self, dataset: TiltDataset, sampler: WeightedRandomSampler) -> DataLoader:
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=sampler)
        return loader

    def _create_sampler(self, dataset: TiltDataset) -> WeightedRandomSampler:
        class_weights = [
            1/torch.sum(dataset.labels == label).float() for label in dataset.labels.unique(sorted=True)]
        return WeightedRandomSampler(class_weights, num_samples=self.batch_size)
