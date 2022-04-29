import os
import click
import json
from rapidflow.experiments.experiment import Experiment

from tiltify.config import BASE_BERT_MODEL, Path
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.objectives.w2v_nb_objective.w2v_nb_preprocessor import W2VPreprocessor
from tiltify.objectives.w2v_nb_objective.w2v_nb_splitter import W2VSplitter
from tiltify.objectives.w2v_nb_objective.w2v_nb_objective import W2VBinaryObjective, W2VRightToObjective


@click.command()
@click.option('--binary', default=False, help='Using this argument invokes the binary classification of RightTo\
examples in general, instead of classifying them distinctly.', is_flag=True)
@click.option('--debug', default=False, help='Debug Option for smaller DocumentCollection', is_flag=True)
@click.option("--k", default=1, type=int, help="Number of Experiment repitions")
@click.option("--trials", default=50, type=int, help="Number of Hyperparameter Settings to run")
@click.option("--num_processes", default=None, type=int, help="Number of processes for  running the experiment.")
@click.option("--batch_size", default=32, type=int, help="Batch Size")
@click.option("--split_ratio", default=0.4, type=float, help="Split ratio between [test, val] and train.")
def train_w2v(binary, debug, k, trials, batch_size, split_ratio, num_processes):
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    if debug:
        data_loading_path = os.path.join(DocumentCollection.data_loader.data_path, "annotated_policies")
        file_names = [file_name for file_name in os.listdir(data_loading_path) if file_name.endswith(".json")][:20]
        loaded_policies = []
        for file_name in file_names:
            with open(os.path.join(data_loading_path, file_name), "r") as f:
                loaded_policies.append(json.load(f))
        document_list = [DocumentCollection.json_parser.parse(**json_policy["document"], annotations=json_policy["annotations"]) for json_policy in loaded_policies]
        document_collection = DocumentCollection(document_list)
    else:
        document_collection = DocumentCollection.from_json_files()
    preprocessor = W2VPreprocessor(binary=binary)
    preprocessed_dataset = preprocessor.preprocess(document_collection=document_collection)
    splitter = W2VSplitter(split_ratio=split_ratio, val=True)
    train, val, test = splitter.split(preprocessed_dataset)
    experiment = Experiment(experiment_path=exp_dir, title="Binary-Classification", model_name="W2V-NB")

    if binary:
        experiment.add_objective(W2VBinaryObjective, args=[train, val, test])
    else:
        experiment.add_objective(W2VRightToObjective, args=[train, val, test])

    experiment.run(k=k, trials=trials, num_processes=num_processes)


if __name__ == "__main__":
    train_w2v()
