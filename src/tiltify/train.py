from tiltify.config import BASE_BERT_MODEL, DEFAULT_DATASET_PATH, DEFAULT_TEST_SPLIT_RATIO
from tiltify.data import get_finetuning_datasets
from tiltify.models import TiltifyBERT


def default_train(dataset_file_path: str = DEFAULT_DATASET_PATH, test_split_ratio: float = DEFAULT_TEST_SPLIT_RATIO,
                  metric_func_str: str = 'frp', freeze_layer_count: int = None) -> TiltifyBERT:

    # load the datasets
    train_dataset, val_dataset = get_finetuning_datasets(dataset_file_path, BASE_BERT_MODEL, test_split_ratio)
    model = TiltifyBERT(num_labels=1, metric_func_str=metric_func_str, freeze_layer_count=freeze_layer_count)
    model.train(train_dataset, val_dataset)
    return model


if __name__ == "__main__":
    default_train()


