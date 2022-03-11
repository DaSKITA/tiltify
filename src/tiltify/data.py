from typing import Dict, List, Tuple, Union
import pandas as pd
from torch.utils.data import Dataset, Subset, random_split
from torch import Generator, Tensor
from transformers import BertTokenizer, BatchEncoding

from tiltify.config import LABEL_REPLACE, RANDOM_SPLIT_SEED, Path



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
