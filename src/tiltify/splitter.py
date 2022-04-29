from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class Splitter:

    def __init__(self, val: bool = False, split_ratio: bool = None) -> None:
        self.split_ratio = split_ratio
        self.val = val

    def split(self, preprocessed_data: Dataset):
        train, test = train_test_split(preprocessed_data, test_size=self.split_ratio)
        if self.val:
            val, test = train_test_split(test, test_size=0.5)
            return train, val, test
        else:
            return train, test
