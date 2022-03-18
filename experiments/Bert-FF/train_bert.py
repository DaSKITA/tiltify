import os
import click
from rapidflow.experiments.experiment import Experiment

from tiltify.config import BASE_BERT_MODEL, Path
from tiltify.objectives.bert_objective.bert_preprocessor import BERTPreprocessor
from tiltify.objectives.bert_objective.bert_binary_objective import BERTBinaryObjective, BERTRightToObjective
from tiltify.objectives.bert_objective.bert_splitter import BERTSplitter
from tiltify.objectives.bert_objective.bert_sampler import BERTSampler


# @click.command()
# @click.option('--binary', default=False, help='Using this argument invokes the binary classification of RightTo\
# examples in general, instead of classifying them distinctly.', is_flag=True)
# @click.option("--n_upsample", default=None, help="Enable upsampling for underrepresented labels", type=float)
# @click.option("--n_downsample", default=None, help="Defines the percentage by which the overrepresented class \
#     is downsampled", type=float)
# @click.option("--k", default=1, type=int, help="Number of Experiment repitions")
# @click.option("--trials", default=50, type=int, help="Number of Hyperparameter Settings to run")
# @click.option("--num_processes", default=None, type=int, help="Number of processes for  running the experiment.")
def train_bert(binary, n_upsample, n_downsample, k, trials, num_processes):
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    # document_collection = DocumentCollection.from_json_files()
    pandas_path = os.path.join(Path.data_path, "de_sentence_data.csv")
    preprocessor = BERTPreprocessor(
        bert_model=BASE_BERT_MODEL, binary=binary, n_upsample=n_upsample, n_downsample=n_downsample)
    preprocessed_dataaset = preprocessor.preprocess_pandas(pandas_path=pandas_path)
    bert_splitter = BERTSplitter(val=True, split_ratio=0.4)
    train, val, test = bert_splitter.split(preprocessed_dataaset)
    bert_sampler = BERTSampler(n_downsample=n_downsample, n_upsample=n_upsample)
    train = bert_sampler.sample(train)
    val = bert_sampler.sample(val)
    test = bert_sampler.sample(test)
    experiment = Experiment(experiment_path=exp_dir, title="Binary-Classification", model_name="Bert-FF")

    if binary:
        experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
    else:
        experiment.add_objective(BERTRightToObjective, args=[train, val, test])

    experiment.run(k=k, trials=trials, num_processes=num_processes)


if __name__ == "__main__":
    train_bert(binary=True, n_upsample=0.5, n_downsample=0.5, k=1, trials=1, num_processes=1)
