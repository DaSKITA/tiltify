from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from data import get_dataset
from utils import get_cli_args, save_predictions
from config import DEFAULT_DATASET_PATH, EVAL_RESULT_PATH, BASE_BERT_MODEL, FINETUNED_BERT_MODEL_PATH


def make_predictions(test_dl: DataLoader, model: BertForSequenceClassification) -> List:
    """Calculate predictions for the given DataLoader object
    
    Using a huggingface transformers model the data given is
    processed and the resulting complexity predictions are
    returned as a list of dicts.
    
    Parameters
    ----------
    test_dl : DataLoader
        a pytorch DataLoader object containing the preprocessed
        dataset as batches
    model : BertForSequenceClassification
        a huggingface transformers BERT sequence classification
        object that is used to make predictions on the
        complexity of the given data
    """

    predictions = []  # for every batch all resulting predictions are saved by appending to this list

    # select GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # switch to eval mode
    model.eval()

    # iterate batches
    for batch in test_dl:
        # move data to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        # calculate the predictions
        prediction = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction = prediction.logits.detach().numpy()

        # store the predictions into the specified list
        for idx in range(prediction.shape[0]):
            predictions.append({'sent_id': int(batch['id'][idx]), 'mos': float(prediction[idx])})

    return predictions


if __name__ == "__main__":
    # gather command line arguments
    cli_args = get_cli_args()
    eval_data_file_path = cli_args.input

    # load the dataset
    print(f"Loading evaluation data from '{eval_data_file_path}' ...")
    dataset = get_dataset(eval_data_file_path, BASE_BERT_MODEL)
    test_dl = DataLoader(dataset)
    print(f"Evaluation data loaded!")

    # load the model to be used
    print(f"Loading model ...")
    model = BertForSequenceClassification.from_pretrained(FINETUNED_BERT_MODEL_PATH)
    print(f"Model loaded!")

    # use the model to make predictions on the dataset and save the results
    print(f"Evaluating model...")
    predictions = make_predictions(test_dl, model)
    save_predictions(predictions, EVAL_RESULT_PATH)
    print(f"Evaluation results saved in '{EVAL_RESULT_PATH}'.")

