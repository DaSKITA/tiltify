import os

from datetime import timedelta
from pathlib import Path

# default global variables
EVAL_RESULT_PATH = "eval.csv"

BASE_BERT_MODEL = "dbmdz/bert-base-german-cased"
FINETUNED_BERT_MODEL_PATH = "model/bert-finetuned"

DEFAULT_TEST_SPLIT_RATIO = 0.33
RANDOM_SPLIT_SEED = 0

LABEL_REPLACE = {None: 0, 'rightToInformation--Description': 1, 'rightToRectificationOrDeletion--Description': 2, 'rightToDataPortability--Description': 3,
                 'rightToWithdrawConsent--Description': 4, 'rightToComplain--Description': 5, 'rightToComplain--Right to Complain - Supervisor Authority': 6}


class Path:

    root_path = os.path.abspath(Path(os.path.dirname(__file__)).parent.parent)
    data_path = os.path.join(root_path, "data")
    policy_path = os.path.join(data_path, "official_policies")
    annotated_policy_path = os.path.join(data_path, "annotated_policies")
    default_dataset_path = os.path.join(root_path, "data/de_sentence_data.csv")
    experiment_path = os.path.join(root_path, "experiments")


class FlaskConfig(object):

    # Secrets
    BASE_PATH = os.path.abspath(os.path.dirname(__file__))
    DEPLOYMENT = os.environ.get("DEPLOYMENT", False)

    JWT_SECRET_KEY = os.environ["JWT_SECRET_KEY"]
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_HEADER_TYPE = ""

    SECRET_KEY = os.environ["FLASK_SECRET_KEY"]
