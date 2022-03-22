import os
from pathlib import Path

# default global variables
EVAL_RESULT_PATH = "eval.csv"

BASE_BERT_MODEL = "dbmdz/bert-base-german-cased"
FINETUNED_BERT_MODEL_PATH = "model/bert-finetuned"

DEFAULT_TEST_SPLIT_RATIO = 0.33
RANDOM_SPLIT_SEED = 0

LABEL_REPLACE = {'None': 0, 'Right to Information': 1, 'Right to Deletion': 2, 'Right to Data Portability': 3,
                 'Right to Withdraw Consent': 4, 'Right to Complain': 5, 'Right to Complain - Supervisor Authority': 5}


class Path:

    root_path = os.path.abspath(Path(os.path.dirname(__file__)).parent.parent)
    data_path = os.path.join(root_path, "data")
    policy_path = os.path.join(data_path, "official_policies")
    annotated_policy_path = os.path.join(data_path, "annotated_policies")
    default_dataset_path = os.path.join(root_path, "data/de_sentence_data.csv")
    experiment_path = os.path.join(root_path, "experiments")
