from transformers import BertForSequenceClassification, get_scheduler
import torch
from torch.optim import AdamW
from tqdm import tqdm

from tiltify.config import BASE_BERT_MODEL, LABEL_REPLACE
from tiltify.extractors.extraction_model import ExtractionModel
from tiltify.data_structures.document import Document, PredictedAnnotation
from tiltify.objectives.bert_objective.bert_preprocessor import BERTPreprocessor
from tiltify.data_structures.document_collection import DocumentCollection


class BinaryBERTModel(ExtractionModel):

    def __init__(self, learning_rate, weight_decay, num_train_epochs, batch_size, k_ranks=None) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.model = BertForSequenceClassification.from_pretrained(BASE_BERT_MODEL, num_labels=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # External Handling?
        self.preprocessor = BERTPreprocessor(bert_model=BASE_BERT_MODEL, binary=True, batch_size=batch_size)
        self.label_name = "Consumer_Right"
        if k_ranks:
            self.k_ranks = k_ranks
        else:
            self.k_ranks = len([key for key in LABEL_REPLACE.keys() if key])

    def train(self, document_collection: DocumentCollection):
        data_loader = self.preprocessor.preprocess(document_collection)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        num_training_steps = self.num_train_epochs * len(data_loader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.model.train()
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
        preprocessed_document = self.preprocessor.preprocess_document(document)
        logits = []
        for batch in preprocessed_document:
            batch.pop("labels")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                output = self.model(**batch)
            logits_batch = output.logits.detach().cpu()
            logits.append(logits_batch)
        logits = torch.cat(logits, dim=0)
        logits, indices = self.form_k_ranks(logits)
        indices = indices.tolist()
        predicted_annotations = [
            PredictedAnnotation(blob_idx=idx[0], blob_text=document.blobs[idx[0]], label=self.label_name)
            for idx in indices]
        return predicted_annotations

    def form_k_ranks(self, logits):
        ranked_logits, indices = torch.sort(logits, descending=True, dim=0)
        return ranked_logits[:self.k_ranks], indices[:self.k_ranks]

    def save(self):
        pass

    def set_k_ranks(self, k_ranks):
        self.k_ranks = k_ranks


if __name__ == "__main__":
    from tiltify.data_structures.document_collection import DocumentCollection
    document = DocumentCollection.from_json_files()[0]
    model = BinaryBERTModel(0.1, 1e-4, 1, 1)
    result = model.predict(document)
    print(result)
