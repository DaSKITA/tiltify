from tqdm import tqdm
import os
import pickle

from tiltify.extractors.extraction_model import ExtractionModel
from tiltify.preprocessing.w2v_nb_preprocessor import W2VPreprocessor
from tiltify.data_structures.document import Document
from tiltify.data_structures.document_collection import DocumentCollection
from sklearn.naive_bayes import GaussianNB


class GaussianNBModel(ExtractionModel):
    _params_names = ["class_count_", "class_prior_", "classes_", "epsilon_", "sigma_", "theta_"]

    def __init__(
        self, prior=None, label=None, num_train_epochs=100, remove_stopwords=False,
            n_classes=2, en=False, weighted_sampling=True) -> None:
        self.num_train_epochs = num_train_epochs
        self.label = label
        if prior:
            self.prior = prior
        else:
            self.prior = [.5, .5]
        if n_classes > 2:
            self.binary = False
        else:
            self.binary = True
        self.label = label
        classes = self.label if isinstance(self.label, list) else [self.label]
        self.classes = {None: 0}
        self.classes.update({name: idx+1 for idx, name in enumerate(classes)})
        self.preprocessor = W2VPreprocessor(en=en, binary=self.binary, remove_stopwords=remove_stopwords,
                                            weighted_sampling=weighted_sampling)

    def train(self, document_collection: DocumentCollection):
        data_loader = self.preprocessor.preprocess(document_collection)
        self.model = GaussianNB(priors=self.prior)

        for epoch in tqdm(range(self.num_train_epochs)):
            for batch in data_loader:
                self.model.partial_fit(
                    X=batch["sentences"], y=batch["labels"], classes=list(self.classes.values()))

    def predict(self, document: Document):
        preprocessed_document, _ = self.preprocessor.preprocess_document(document)
        logits = self.model.predict_proba(preprocessed_document)
        idx = self.classes.get(self.label, None)
        label_logits = logits[:, idx]
        return label_logits.tolist()

    @classmethod
    def load(cls, load_path, label):
        load_path = os.path.join(load_path, "gaussian_NB.p")
        with open(load_path, "rb") as f:
            model = pickle.load(f)
        init_obj = cls(label=label)
        init_obj.model = model
        return init_obj

    def save(self, save_path):
        save_path = os.path.join(save_path, "gaussian_NB.p")
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)


if __name__ == "__main__":
    document_collection = DocumentCollection.from_json_files()[:2]
    model = GaussianNBModel(prior=[0.9, 0.1], label="Right to Withdraw")
    model.train(document_collection)
    predictions = model.predict(document_collection[0])
    print(predictions)
