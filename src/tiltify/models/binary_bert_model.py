from transformers import BertForSequenceClassification, get_scheduler
import torch
from torch.optim import AdamW
from tqdm import tqdm

from tiltify.config import BASE_BERT_MODEL
from tiltify.data_structures.document import Document
from tiltify.extractors.extraction_model import ExtractionModel
from tiltify.preprocessing.bert_preprocessor import BERTPreprocessor
from tiltify.data_structures.document_collection import DocumentCollection


class BinaryBERTModel(ExtractionModel):

    def __init__(
        self, learning_rate=1e-3, weight_decay=1e-5, num_train_epochs=5, batch_size=20,
            label=None) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.model = BertForSequenceClassification.from_pretrained(BASE_BERT_MODEL, num_labels=1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocessor = BERTPreprocessor(
            bert_model=BASE_BERT_MODEL, binary=True, batch_size=batch_size, label=label)
        self.sigmoid = torch.nn.Sigmoid()

    def train(self, document_collection: DocumentCollection):
        data_loader = self.preprocessor.preprocess(document_collection)
        self.model.to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        num_training_steps = self.num_train_epochs * len(data_loader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        for _ in tqdm(range(self.num_train_epochs)):
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = criterion(outputs.logits.flatten(), labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

    def predict(self, document: Document):
        document_loader = self.preprocessor.preprocess_document(document)
        logits = []
        for batch in document_loader:
            _ = batch.pop("labels")
            preprocessed_document = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                output = self.model(**preprocessed_document)
            estimates = self.sigmoid(output.logits)
            logits.append(estimates)
        logits = torch.cat(logits, dim=0)
        logits = logits.detach().cpu().tolist()
        logits = sum(logits, [])
        return logits

    @classmethod
    def load(cls, load_path, label):
        model = BertForSequenceClassification.from_pretrained(load_path, num_labels=1, local_files_only=True)
        model.eval()
        init_obj = cls(label=label)
        init_obj.model = model
        init_obj.model.to(init_obj.device)
        return init_obj

    def save(self, save_path):
        self.model.save_pretrained(save_path)

    def to_device(self, device: str):
        self.model.to(torch.device(device))
