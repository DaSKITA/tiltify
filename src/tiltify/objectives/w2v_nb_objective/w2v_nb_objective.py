from typing import Dict

import numpy as np
import tqdm
from rapidflow.metrics_handler import MetricsHandler
from rapidflow.objective import Objective
from sklearn.naive_bayes import GaussianNB

from tiltify.objectives.w2v_nb_objective.w2v_nb_preprocessor import W2VDataset


class W2VBinaryObjective(Objective):

    def __init__(self, train_dataloader: W2VDataset, val_dataloader: W2VDataset, test_dataloader: W2VDataset):
        super().__init__()
        self.train_dataloader, self.val_dataloader, self.test_dataloader = \
            train_dataloader, val_dataloader, test_dataloader
        self.labels = 2

    def train(self, trial=None) -> Dict:
        # Hyperparameter setup
        first_class_prior = trial.suggest_float("priors", 0.5, 0.95, log=True)
        hyperparameters = dict(
            priors=[first_class_prior, 1 - first_class_prior],
            num_train_epochs=100
        )

        # Model setup
        model = GaussianNB(priors=hyperparameters["priors"])
        self.track_model(model, hyperparameters)
        for epoch in tqdm.tqdm(range(hyperparameters["num_train_epochs"])):
            for batch in self.train_dataloader:
                self.model.partial_fit(X=batch["sentences"], y=batch["labels"], classes=np.array(range(self.labels)))

        # Validation
        val_labels = []
        val_preds = []
        for batch in self.val_dataloader:
            val_labels += batch["labels"]
            predictions = self.model.predict(batch["sentences"])
            val_preds += predictions.tolist()
        metrics_handler = MetricsHandler()
        metrics = metrics_handler.calculate_classification_metrics(val_labels, val_preds)
        return metrics['macro avg f1-score']

    def test(self):
        test_labels = []
        test_preds = []
        for batch in self.test_dataloader:
            test_labels += batch["labels"]
            test_preds += self.model.predict(batch["sentences"]).tolist()
        metrics_handler = MetricsHandler()
        metrics = metrics_handler.calculate_classification_metrics(test_labels, test_preds)
        return metrics


class W2VRightToObjective(W2VBinaryObjective):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = 6
