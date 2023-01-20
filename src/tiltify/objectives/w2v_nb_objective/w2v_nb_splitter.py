from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from tiltify.splitter import Splitter
from tiltify.objectives.w2v_nb_objective.w2v_nb_preprocessor import W2VDataset


class W2VSplitter(Splitter):

    def __init__(self, val: float = False, split_ratio: bool = None, batch_size: int = None):
        super().__init__(val, split_ratio)
        self.batch_size = batch_size if batch_size else None

    def split(self, preprocessed_data: W2VDataset):
        loader_list = []
        split_idx_list = self._perform_splitting(dataset=preprocessed_data)
        set_list = self._apply_splitted_idx(preprocessed_data, split_idx_list)
        for split_set in set_list:
            if split_set:
                loader_list.append(self._create_dataloader(split_set))
        return loader_list

    def _perform_splitting(self, dataset: W2VDataset) -> Tuple:
        index = list(range(len(dataset)))
        train, test = train_test_split(index, test_size=self.split_ratio)
        if self.val:
            val, test = train_test_split(test, test_size=0.5)
            return train, val, test
        else:
            return train, test

    def _create_dataloader(self, dataset: Dataset) -> DataLoader:
        sampler = self._create_sampler(dataset)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=sampler)
        return loader

    def _create_sampler(self, dataset: Dataset) -> WeightedRandomSampler:
        class_weights = [
            1 / torch.sum(dataset.labels == label).float() for label in dataset.labels.unique(sorted=True)]
        sample_weights = torch.Tensor([class_weights[int(t)] for t in dataset.labels])
        return WeightedRandomSampler(sample_weights, num_samples=sample_weights.shape[0])

    def _apply_splitted_idx(self, dataset: W2VDataset, split_idx_list: Tuple[List]) -> List[Dataset]:
        dataset_list = []
        for split_idx in split_idx_list:
            dataset_dict = dataset[split_idx]
            dataset_list.append(W2VDataset(**dataset_dict))
        return dataset_list