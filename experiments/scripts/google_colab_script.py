import os

from tiltify.config import BASE_BERT_MODEL, Path
from tiltify.data import get_finetuning_datasets
from tiltify.objective import BERTBinaryObjective
from torch.multiprocessing import set_start_method
from rapidflow.experiments.experiment import Experiment


if __name__ == "__main__":

    train, val, test = get_finetuning_datasets(Path.DEFAULT_DATASET_PATH, BASE_BERT_MODEL, val=True)
    experiment = Experiment(experiment_path=os.path.abspath(''))
    experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
    experiment.run(k=2, trials=2, num_processes=1)
