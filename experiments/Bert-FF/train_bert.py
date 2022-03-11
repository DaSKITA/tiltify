import os
import click
from rapidflow.experiments.experiment import Experiment

from tiltify.config import BASE_BERT_MODEL
from tiltify.objectives.bert_objective.bert_preprocessor import BERTPreprocessor
from tiltify.objectives.bert_objective.bert_binary_objective import BERTBinaryObjective, BERTRightToObjective
from tiltify.data_structures.document_collection import DocumentCollection


@click.command()
@click.option('--binary', default=False, help='Using this argument invokes the binary classification of RightTo\
examples in general, instead of classifying them distinctly.', is_flag=True)
def train_bert(binary):
    document_collection = DocumentCollection.from_json_files()
    preprocessor = BERTPreprocessor(bert_model=BASE_BERT_MODEL, binary=binary)
    preprocessed_dataaset = preprocessor.preprocess(document_collection=document_collection)
    bert_splitter = BertSplitter(val=True)
    train, val, test = bert_splitter.split(preprocessed_dataaset)
    experiment = Experiment(experiment_path=os.path.abspath(''))

    if binary:
        experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
    else:
        experiment.add_objective(BERTRightToObjective, args=[train, val, test])

    experiment.run(k=2, trials=2, num_processes=1)


if __name__ == "__main__":
    train_bert()
