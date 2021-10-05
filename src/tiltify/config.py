# default global variables
DEFAULT_DATASET_PATH = "/Users/farazo/Documents/DaSKITA/playground/tiltify/data/sentence_data.csv"
EVAL_RESULT_PATH = "eval.csv"

BASE_BERT_MODEL = "bert-base-uncased"
FINETUNED_BERT_MODEL_PATH = "model/bert-finetuned"

DEFAULT_TEST_SPLIT_RATIO = 0.33
RANDOM_SPLIT_SEED = 0

LABEL_REPLACE = {'None': 0, 'Right to Information': 1, 'Right to Deletion': 2, 'Right to Data Portability': 3,
                 'Right to Withdraw Consent': 4, 'Right to Complain': 5, 'Right to Complain - Supervisor Authority': 5}