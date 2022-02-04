import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

from tiltify.config import BASE_BERT_MODEL, FINETUNED_BERT_MODEL_PATH
from tiltify.utils import classification_report_metric, precision_recall_fscore_metric


class BinaryTiltifyBERT:
    def __init__(self, metric_func_str: str = 'frp',  freeze_layer_count: int = None):
        self.model = BertForSequenceClassification.from_pretrained(BASE_BERT_MODEL, num_labels=2)
        if freeze_layer_count:
            # We freeze here the embeddings of the model
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False

            if freeze_layer_count != -1:
                # if freeze_layer_count == -1, we only freeze the embedding layer
                # otherwise we freeze the first `freeze_layer_count` encoder layers
                for layer in self.model.bert.encoder.layer[:freeze_layer_count]:
                    for param in layer.parameters():
                        param.requires_grad = False
        self.freeze_layer_count = freeze_layer_count
        self.metric_func = {
            'frp': precision_recall_fscore_metric,
            'cr': classification_report_metric
        }[metric_func_str]

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)

    def train(self, train_dataset, val_dataset):
        training_args = TrainingArguments("finetune_trainer",
                                          evaluation_strategy="epoch",
                                          logging_strategy="epoch",
                                          per_device_train_batch_size=32,
                                          per_device_eval_batch_size=32,
                                          num_train_epochs=10)

        trainer = Trainer(model=self.model,
                          args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=val_dataset,
                          compute_metrics=self.metric_func)
        trainer.train()
        trainer.evaluate()

        self.model.save_pretrained(FINETUNED_BERT_MODEL_PATH)
        return self.model

    @staticmethod
    def hyperparameter_space(trial):
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            num_train_epochs=trial.suggest_int("num_train_epochs", 1, 20)
        )

    @staticmethod
    def model_init():
        model = BertForSequenceClassification.from_pretrained(BASE_BERT_MODEL, num_labels=2)
        for param in model.base_model.parameters():
            param.requires_grad = False
        return model

    def hyperparameter_search(self, train_dataset, val_dataset):
        training_args = TrainingArguments(
            output_dir=FINETUNED_BERT_MODEL_PATH,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            eval_steps=1000,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            save_total_limit=1
        )

        trainer = Trainer(
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.metric_func,
            model_init=self.model_init
        )

        best_run = trainer.hyperparameter_search(direction="minimize", hp_space=self.hyperparameter_space,
                                                 compute_objective=lambda x: x["eval_loss"], n_trials=100)

        return best_run
