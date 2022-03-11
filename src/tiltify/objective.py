from typing import Dict

import torch

from rapidflow.metrics_handler import MetricsHandler
from rapidflow.objective import Objective
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

from tiltify.config import BASE_BERT_MODEL
from tiltify.data import TiltFinetuningDataset


class BERTBinaryObjective(Objective):

    def __init__(self, train_dataset: TiltFinetuningDataset, val_dataset: TiltFinetuningDataset,
                 test_dataset: TiltFinetuningDataset):
        super().__init__()
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        self.labels = 2

    @staticmethod
    def _metric_func(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        metrics_handler = MetricsHandler()
        classification_metrics = metrics_handler.calculate_classification_metrics(labels, preds)
        return classification_metrics

    def train(self, trial=None) -> Dict:
        hyperparameters = dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            num_train_epochs=1,
            weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-5, log=True),
        )
        # model setup
        model = BertForSequenceClassification.from_pretrained(BASE_BERT_MODEL, num_labels=self.labels)
        self.track_model(model, hyperparameters)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # train
        training_args = TrainingArguments("finetune_trainer",
                                          evaluation_strategy="epoch",
                                          logging_strategy="epoch",
                                          per_device_train_batch_size=32,
                                          per_device_eval_batch_size=32,
                                          **self.hyperparameters)

        trainer = Trainer(model=self.model,
                          args=training_args,
                          train_dataset=self.train_dataset,
                          eval_dataset=self.val_dataset,
                          compute_metrics=self._metric_func)

        trainer.train()
        metrics = trainer.evaluate()
        return metrics['eval_macro avg f1-score']

    def test(self):
        testing_args = TrainingArguments("finetune_trainer",
                                         evaluation_strategy="epoch",
                                         logging_strategy="epoch",
                                         per_device_train_batch_size=32,
                                         per_device_eval_batch_size=32)

        trainer = Trainer(model=self.model,
                          args=testing_args,
                          eval_dataset=self.test_dataset,
                          compute_metrics=self._metric_func)

        return trainer.evaluate()


class BERTRightToObjective(BERTBinaryObjective):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = 6
