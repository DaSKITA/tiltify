import os
import click
from rapidflow.experiments.experiment import Experiment

from tiltify.config import BASE_BERT_MODEL, Path
from tiltify.preprocessing.bert_preprocessor import BERTPreprocessor
from tiltify.objectives.bert_objective.bert_binary_objective import BERTBinaryObjective#, BERTRightToObjective
from tiltify.preprocessing.document_collection_splitter import DocumentCollectionSplitter
from tiltify.data_structures.document_collection import DocumentCollection


@click.command()
@click.option('--binary', default=False, help='Using this argument invokes the binary classification of RightTo\
examples in general, instead of classifying them distinctly.', is_flag=True)
@click.option("--k", default=1, type=int, help="Number of Experiment repitions")
@click.option("--trials", default=50, type=int, help="Number of Hyperparameter Settings to run")
@click.option("--num_processes", default=None, type=int, help="Number of processes for  running the experiment.")
@click.option("--batch_size", default=32, type=int, help="Batch Size")
@click.option("--split_ratio", default=0.4, type=float, help="Split ratio between [test, val] and train.")
@click.option("--labels", default=None, type=list, help="The Label the Model should be trained on. Defaults to all supported by Tiltify App.")
def train_bert(binary, k, trials, batch_size, split_ratio, num_processes, labels):
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    document_collection = DocumentCollection.from_json_files()
    preprocessor = BERTPreprocessor(
        bert_model=BASE_BERT_MODEL, binary=binary, labels=labels)
    preprocessed_dataaset = preprocessor.preprocess(document_collection)
    bert_splitter = DocumentCollectionSplitter(val=True, split_ratio=split_ratio, batch_size=batch_size)
    train, val, test = bert_splitter.split(preprocessed_dataaset)
    experiment = Experiment(experiment_path=exp_dir, title="Binary-Classification", model_name="Bert-FF")

    if binary:
        experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
    #else:
    #    experiment.add_objective(BERTRightToObjective, args=[train, val, test])

    experiment.run(k=k, trials=trials, num_processes=num_processes)


if __name__ == "__main__":
    train_bert()
