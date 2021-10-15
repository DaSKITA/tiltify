from typing import Tuple

import torch
from torch.utils.data import Subset, random_split
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

from tiltify.data import TiltDataset, TiltFinetuningDataset, get_dataset
from tiltify.utils import get_cli_args
from tiltify.config import BASE_BERT_MODEL, DEFAULT_DATASET_PATH, DEFAULT_TEST_SPLIT_RATIO,\
    FINETUNED_BERT_MODEL_PATH, RANDOM_SPLIT_SEED


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
                               generator=torch.Generator().manual_seed(RANDOM_SPLIT_SEED))
    return train, test


def finetune_and_evaluate_model(model: BertForSequenceClassification, dataset: TiltDataset,
                                test_split_ratio: float = None):
    """Fine-tune (and evaluate) model using the given dataset.
    Uses the Trainer API of the transformers library.

    Parameters
    ----------
    dataset : TiltDataset
        a dataset containing tokenized sentences
    model : BertForSequenceClassification
        a huggingface transformers BERT sequence classification
        object that is fine-tuned using the dataset
    """
    if test_split_ratio:
        train_ds, test_ds = get_train_test_split(dataset, test_split_ratio)
        train_ft_ds = TiltFinetuningDataset(train_ds)
        test_ft_ds = TiltFinetuningDataset(test_ds)
    else:
        train_ft_ds = TiltFinetuningDataset(dataset)
        test_ft_ds = None

    training_args = TrainingArguments("finetune_trainer",
                                      evaluation_strategy="epoch",
                                      logging_strategy="epoch",
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16)

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ft_ds, eval_dataset=test_ft_ds)
    trainer.train()
    trainer.evaluate()

    model.save_pretrained(FINETUNED_BERT_MODEL_PATH)


def default_train(dataset_file_path: str = DEFAULT_DATASET_PATH, test_split_ratio: float = DEFAULT_TEST_SPLIT_RATIO)\
        -> BertForSequenceClassification:
    # load the dataset
    print(f"Loading dataset from '{dataset_file_path}' ...")
    dataset = get_dataset(dataset_file_path, BASE_BERT_MODEL)

    # load the base model to be fine-tuned
    print(f"Loading base model ...")
    # 'num_labels = 6' for classification task
    model = BertForSequenceClassification.from_pretrained(BASE_BERT_MODEL, num_labels=6)
    print(f"Model loaded!")

    # fine-tune the model and evaluate each epoch if test split is given
    print(f"Fine-tuning model...")
    finetune_and_evaluate_model(model, dataset, test_split_ratio)
    print(f"Fine-tuned model saved to '{FINETUNED_BERT_MODEL_PATH}'.")

    return model


if __name__ == "__main__":
    # gather command line arguments
    cli_args = get_cli_args()
    dataset_file_path = cli_args.input
    test_split_ratio = float(cli_args.test_split)

    # load the dataset
    print(f"Loading dataset from '{dataset_file_path}' ...")
    dataset = get_dataset(dataset_file_path, BASE_BERT_MODEL)
    # load the base model to be fine-tuned
    print(f"Loading base model ...")
    # 'num_labels = 1' for regression task
    model = BertForSequenceClassification.from_pretrained(BASE_BERT_MODEL, num_labels=1)
    print(f"Model loaded!")

    # fine-tune the model and evaluate each epoch if test split is given
    print(f"Fine-tuning model...")
    finetune_and_evaluate_model(model, dataset, test_split_ratio)
    print(f"Fine-tuned model saved to '{FINETUNED_BERT_MODEL_PATH}'.")


