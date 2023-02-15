import os
from sentence_transformers import SentenceTransformer, losses, util, InputExample
import itertools
from torch.utils.data import DataLoader
import torch

from tiltify.extractors.extraction_model import ExtractionModel
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.data_structures.document import Document
from tiltify.preprocessing.label_retriever import LabelRetriever


class SentenceBert(ExtractionModel):

    def __init__(self, label, num_train_epochs=5, en=False, batch_size=50, binary=True) -> None:
        self.label = label
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        if en:
            self.pretrained_model = 'dbmdz/bert-base-english-cased'
        else:
            self.pretrained_model = 'dbmdz/bert-base-german-cased'
        self.preprocessor = SentenceBertPreprocessor(self.label, binary)
        self.encoded_query = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, document_collection: DocumentCollection):
        self.model = SentenceTransformer(self.pretrained_model, device=self.device)
        triplet_corpus = self.preprocessor.preprocess(document_collection)
        triplet_corpus = DataLoader(
            triplet_corpus, shuffle=True, batch_size=self.batch_size,
            collate_fn=self.model.smart_batching_collate)
        train_loss = losses.TripletLoss(model=self.model, triplet_margin=5)
        self.model.fit(
            train_objectives=[(triplet_corpus, train_loss)], epochs=self.num_train_epochs, warmup_steps=100)
        self.encoded_query = self.model.encode(self.preprocessor.query)

    def predict(self, document: Document) -> list[float]:
        blob_texts = [blob.text for blob in document.blobs]
        encoded_corpus = self.model.encode(blob_texts, convert_to_tensor=True)
        cos_scores = util.cos_sim(self.encoded_query, encoded_corpus)[0]
        return cos_scores.detach().cpu().tolist()

    @classmethod
    def load(cls, load_path, label):
        load_path = os.path.join(load_path, "sbert")
        model = SentenceTransformer.load(load_path)
        init_obj = cls(label)
        init_obj.model = model
        init_obj.encoded_query = model.encode(init_obj.preprocessor.query)
        return init_obj

    def save(self, save_path):
        save_path = os.path.join(save_path, "sbert")
        self.model.save(save_path)


class SentenceBertPreprocessor:

    queries = {
        'Right to Information': "Das Bestehen eines Rechts auf Auskunft seitens des Verantwortlichen über die"
                                "betreffenden personenbezogenen Daten.",
        'Right to Deletion': "Die betroffene Person hat das Recht, von dem Verantwortlichen unverzüglich die"
                                "Berichtigung sie betreffender unrichtiger personenbezogener Daten zu"
                                "verlangen."
                                "Die betroffene Person hat das Recht, von dem Verantwortlichen zu verlangen, "
                                "dass sie betreffende"
                                "personenbezogene Daten unverzüglich gelöscht werden.",
        'Right to Data Portability': "Die betroffene Person hat das Recht, die sie betreffenden "
                                        "personenbezogenen Daten, die sie einem Verantwortlichen"
                                        "bereitgestellt hat, in einem strukturierten, gängigen und "
                                        "maschinenlesbaren Format zu erhalten.",
        'Right to Withdraw Consent':  "Das Bestehen eines Rechts, die Einwilligung jederzeit zu widerrufen.",
        'Right to Complain': "Das Bestehen eines Beschwerderechts bei einer Aufsichtsbehörde."
    }

    def __init__(self, label, binary=True) -> None:
        self.label_retriever = LabelRetriever(label)
        self.query = self.queries[label]
        self.binary = binary

    def preprocess(self, document_collection: DocumentCollection):
        triplet_corpus = []
        for document in document_collection:
            triplet_document = self.preprocess_document(document)
            triplet_corpus.append(triplet_document)
        triplet_corpus = sum(triplet_corpus, [])
        return triplet_corpus

    def preprocess_document(self, document: Document):
        labels = self.label_retriever.retrieve_labels(document.blobs)
        labels = self.prepare_labels(labels)  # binarize labels
        positive_data = []
        negative_data = []
        for blob, label in zip(document.blobs, labels):
            if label > 0:
                positive_data.append(blob.text)
            else:
                negative_data.append(blob.text)

        # no labels because triplet loss doesnt expect a label (https://arxiv.org/pdf/1908.10084.pdf)
        triplet_data = [
            InputExample(texts=combination)
            for combination in itertools.product([self.query], positive_data, negative_data)]
        return triplet_data

    def prepare_labels(self, labels: list):
        """Binarize Labels if necessray.
        Only Labels that are above 1 are matched. As the preprocessor tries to identify relevant blobs.

        Args:
            labels (List): _description_

        Returns:
            _type_: _description_
        """
        if self.binary:
            labels = [int(label[0] > 0) for label in labels]
        return labels

    @staticmethod
    def _sbert_colate(input_list):
        x = []
        for x_ in input_list:
            x.append(x_)
        return x


if __name__ == "__main__":
    document_collection = DocumentCollection.from_json_files()
    extractor = SentenceBert(label="Right to Information")
    extractor.train(document_collection)
    predicted_indices = extractor.predict(document_collection[0])
    print(predicted_indices)
