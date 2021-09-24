from typing import List, Dict
import argparse

import pandas as pd

from config import DEFAULT_DATASET_PATH, DEFAULT_TEST_SPLIT_RATIO


def get_cli_args() -> argparse.Namespace:
    """Parses the given client arguments.
    
    Reacts to the argument --input, which specifies the path
    to a CSV dataset.
    """

    parser = argparse.ArgumentParser(description='Evaluate or train model on dataset.')
    parser.add_argument('--input', default=DEFAULT_DATASET_PATH,
                        help='path of the test data file')
    parser.add_argument('--test_split', default=DEFAULT_TEST_SPLIT_RATIO,
                        help='share of the dataset to be used as test data in case of training')
    return parser.parse_args()


def save_predictions(predictions: List[Dict], file_path: str):
    """Saves a list of predictions as a CSV file in the specified path.

    Parameters
    ----------
    predictions : List[Dict]
        a list of complexity score predictions, where
        each prediction consists of the according
        sentence id and the predicted MOS
    file_path : str
        a path specifying the location of the result CSV file   
    """

    df_predictions = pd.DataFrame.from_dict(predictions)
    df_predictions.to_csv(file_path, index=False)

