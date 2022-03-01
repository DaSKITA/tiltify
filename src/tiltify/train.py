from tiltify.config import BASE_BERT_MODEL, DEFAULT_TEST_SPLIT_RATIO, Path
from tiltify.data import get_finetuning_datasets
from tiltify.models import BinaryTiltifyBERT


def default_train(dataset_file_path: str = Path.default_dataset_path, test_split_ratio: float = DEFAULT_TEST_SPLIT_RATIO,
                  metric_func_str: str = 'frp', freeze_layer_count: int = None) -> BinaryTiltifyBERT:

    # load the datasets
    train_ds, _, test_ds = get_finetuning_datasets(dataset_file_path, BASE_BERT_MODEL, test_split_ratio)
    model = BinaryTiltifyBERT(metric_func_str=metric_func_str, freeze_layer_count=freeze_layer_count)
    model.train(train_ds, test_ds)
    return model


if __name__ == "__main__":
    default_train()


