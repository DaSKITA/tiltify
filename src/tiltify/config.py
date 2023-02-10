import os
from datetime import timedelta
from pathlib import Path

# default global variables
EVAL_RESULT_PATH = "eval.csv"

BASE_BERT_MODEL = "dbmdz/bert-base-german-cased"
FINETUNED_BERT_MODEL_PATH = "model/bert-finetuned"

DEFAULT_TEST_SPLIT_RATIO = 0.33
RANDOM_SPLIT_SEED = 0

# this should be possible with the exraction manager
EXTRACTOR_CONFIG = [
    #("GaussianNB", "Right to Withdraw Consent"),
    #("BinaryBert", "Right to Deletion"),
    ("Test", "Right to Information")
    #("SentenceBert", "Right to Complain")

    # ("BinaryBert", ["Right to Withdraw Consent", "Right to Complain"])
]

# Used for training
TILT_LABELS = [
    'rightToInformation--Description',
    'rightToRectificationOrDeletion--Description',
    'rightToDataPortability--Description',
    'rightToWithdrawConsent--Description',
    'rightToComplain--Description',
    'rightToComplain--Right to Complain - Supervisor Authority'
]

# used for prediction
SUPPORTED_LABELS = [
    'Right to Information',
    'Right to Deletion',
    'Right to Data Portability',
    'Right to Withdraw Consent',
    'Right to Complain'
]


class Path:

    root_path = os.path.abspath(Path(os.path.dirname(__file__)).parent.parent)
    data_path = os.path.join(root_path, "data")
    policy_path = os.path.join(data_path, "official_policies")
    annotated_policy_path = os.path.join(data_path, "annotated_policies")
    default_dataset_path = os.path.join(root_path, "data/de_sentence_data.csv")
    experiment_path = os.path.join(root_path, "experiments")


class FlaskConfig(object):

    try:
        # Secrets
        BASE_PATH = os.path.abspath(os.path.dirname(__file__))
        DEPLOYMENT = os.environ.get("DEPLOYMENT", False)
        if DEPLOYMENT is False:
            DEBUG = True

        JWT_SECRET_KEY = os.environ["JWT_SECRET_KEY"]
        JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
        JWT_HEADER_TYPE = ""

        SECRET_KEY = os.environ["FLASK_SECRET_KEY"]
    except KeyError:
        print("Environment Variables not found. Entering Test Mode!")
