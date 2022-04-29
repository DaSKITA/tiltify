from typing import Dict

import torch
from rapidflow.metrics_handler import MetricsHandler
from rapidflow.objective import Objective
from transformers import BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
import tqdm

from tiltify.config import BASE_BERT_MODEL
from tiltify.objectives.bert_objective.bert_preprocessor import TiltDataset


class BERTBinaryObjective(Objective):

    def __init__(
            self, train_dataloader: TiltDataset, val_dataloader: TiltDataset, test_dataloader: TiltDataset):
        """https://huggingface.co/docs/transformers/training

        Args:
            train_dataset (TiltDataset): _description_
            val_dataset (TiltDataset): _description_
            test_dataset (TiltDataset): _description_
        """
        super().__init__()
        self.train_dataloader, self.val_dataloader, self.test_dataloader = \
            train_dataloader, val_dataloader, test_dataloader
        self.labels = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, trial=None) -> Dict:
        hyperparameters = dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            num_train_epochs=5,
            weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-5, log=True),
        )
        num_training_steps = hyperparameters["num_train_epochs"] * len(self.train_dataloader)
        # model setup
        model = BertForSequenceClassification.from_pretrained(BASE_BERT_MODEL, num_labels=self.labels)
        self.track_model(model, hyperparameters)
        self.model.to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = AdamW(
            self.model.parameters(), lr=hyperparameters["learning_rate"],
            weight_decay=hyperparameters["weight_decay"])
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        # train
        self.model.train()
        for epoch in tqdm.tqdm(range(hyperparameters["num_train_epochs"])):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = model(**batch)
                loss = criterion(outputs.logits.max(dim=1)[0], labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        val_labels = []
        val_preds = []
        self.model.eval()
        for batch in self.val_dataloader:
            val_labels += batch.pop("labels").tolist()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                output = self.model(**batch)
            logits = output.logits
            predictions = torch.argmax(logits, dim=-1)
            val_preds += predictions.detach().cpu().tolist()
        metrics_handler = MetricsHandler()
        metrics = metrics_handler.calculate_classification_metrics(val_labels, val_preds)
        return metrics['macro avg f1-score']

    def test(self):
        test_labels = []
        test_preds = []
        self.model.eval()
        for batch in self.test_dataloader:
            test_labels += batch.pop("labels").tolist()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                output = self.model(**batch)
            logits = output.logits
            predictions = torch.argmax(logits, dim=-1)
            test_preds += predictions.detach().cpu().tolist()
        metrics_handler = MetricsHandler()
        metrics = metrics_handler.calculate_classification_metrics(test_labels, test_preds)
        return metrics


class BERTRightToObjective(BERTBinaryObjective):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = 6
