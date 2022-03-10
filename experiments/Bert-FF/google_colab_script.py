import os

from tiltify.config import BASE_BERT_MODEL, Path
from tiltify.data import get_finetuning_datasets
from tiltify.objective import BERTBinaryObjective
from rapidflow.experiments.experiment import Experiment


if __name__ == "__main__":

    exp_path = os.path.join(Path.experiment_path, "Bert-FF")
    train, val, test = get_finetuning_datasets(Path.default_dataset_path, BASE_BERT_MODEL, val=True, binary=True)
    experiment = Experiment(title="reannotated_documents_first", experiment_path=exp_path, model_name="Bert-FF")
    experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
    experiment.run(k=1, trials=1, num_processes=1)
