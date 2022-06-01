from tiltify.objectives.bert_objective.bert_preprocessor import TiltDataset
from typing import Tuple, List
from sklearn.model_selection import train_test_split

from tiltify.data_structures.document_collection import DocumentCollection


class DocumentCollectionSplitter:

    def __init__(self, val: float = False, split_ratio: bool = None) -> None:
        self.split_ratio = split_ratio
        self.val = val

    def split(self, document_collection: DocumentCollection):
        loader_list = []
        split_idx_list = self._perform_splitting(document_collection)
        set_list = self._apply_splitted_idx(document_collection, split_idx_list)
        for split_set in set_list:
            if split_set:
                loader_list.append(self._create_dataloader(split_set))
        return loader_list

    def _perform_splitting(self, document_collection: DocumentCollection) -> Tuple[List]:
        index = list(range(len(document_collection)))
        train, test = train_test_split(index, test_size=self.split_ratio)
        if self.val:
            val, test = train_test_split(test, test_size=0.5)
            return train, val, test
        else:
            return train, test

    def _apply_splitted_idx(self, document_collection: DocumentCollection, split_idx_list: List) -> List[TiltDataset]:
        dataset_list = []
        for split_idx in split_idx_list:
            subset = document_collection[split_idx]
            dataset_list.append(subset)
        return dataset_list
