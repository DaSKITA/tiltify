import os
import click
from rapidflow.experiments.experiment import Experiment

from tiltify.config import BASE_BERT_MODEL, Path
from tiltify.objectives.bert_objective.bert_preprocessor import BERTPreprocessor
from tiltify.objectives.bert_objective.bert_binary_objective import BERTBinaryObjective, BERTRightToObjective
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.objectives.bert_objective.bert_splitter import BERTSplitter


@click.command()
@click.option('--binary', default=False, help='Using this argument invokes the binary classification of RightTo\
examples in general, instead of classifying them distinctly.', is_flag=True)
def train_bert(binary):
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    # document_collection = DocumentCollection.from_json_files()
    pandas_path = os.path.join(Path.data_path, "de_sentence_data.csv")
    preprocessor = BERTPreprocessor(bert_model=BASE_BERT_MODEL, binary=binary)
    preprocessed_dataaset = preprocessor.preprocess_pandas(pandas_path=pandas_path)
    bert_splitter = BERTSplitter(val=True, split_ratio=0.33)
    train, val, test = bert_splitter.split(preprocessed_dataaset)
    experiment = Experiment(experiment_path=exp_dir, title="Binary-Classification", model_name="Bert-FF")

    if binary:
        experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
    else:
        experiment.add_objective(BERTRightToObjective, args=[train, val, test])

    experiment.run(k=1, trials=1, num_processes=1)


if __name__ == "__main__":
    train_bert(binary=True)
