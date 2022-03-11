import os

import click
from rapidflow.experiments.experiment import Experiment

from tiltify.config import BASE_BERT_MODEL, Path
from tiltify.data import get_finetuning_datasets
from tiltify.objective import BERTBinaryObjective, BERTRightToObjective


@click.command()
@click.option('--binary', default=False, help='Using this argument invokes the binary classification of RightTo\
examples in general, instead of classifying them distinctly.', is_flag=True)
def train_bert(binary):
    train, val, test = get_finetuning_datasets(Path.default_dataset_path, BASE_BERT_MODEL, val=True, binary=binary)
    experiment = Experiment(experiment_path=os.path.abspath(''))

    if binary:
        print("BINARY")
        experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
    else:
        print("NON-BINARY")
        experiment.add_objective(BERTRightToObjective, args=[train, val, test])

    experiment.run(k=2, trials=2, num_processes=1)


if __name__ == "__main__":
    train_bert()

