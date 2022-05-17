from tiltify.extractors.extractor import Extractor
from tiltify.data_structures.document import Document
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.objectives.bert_objective.bert_preprocessor import BERTPreprocessor
from tiltify.objectives.bert_objective.bert_splitter import BERTSplitter
from tiltify.objectives.bert_objective.bert_binary_objective import BERTBinaryObjective
from tiltify.config import BASE_BERT_MODEL
from rapidflow.experiments.experiment import Experiment


# TODO: should be a generic class that is created via components

class BinaryBERTExtractor(Extractor):

    def __init__(self) -> None:
        self.preprocessor = BERTPreprocessor(
            bert_model=BASE_BERT_MODEL, binary=True)
        self.title = "Binary Classification"
        # self.model_name = self.model.__class__.__name__
        self.model = None

    def train(
            self, document_collection: DocumentCollection, split_ratio, batch_size, k, trials, num_processes):
        # TODO: maybe this can be more generic
        # TODO: change processing for document collection
        preprocessed_dataset = self.preprocessor.preprocess(document_collection=document_collection)
        bert_splitter = BERTSplitter(val=True, split_ratio=split_ratio, batch_size=batch_size)
        train, val, test = bert_splitter.split(preprocessed_dataset)
        experiment = Experiment(
            experiment_path=self.exp_dir, title=self.title, model_name=self.model_name)
        experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
        return experiment

    def predict(self, document: Document):
        return super().predict(document)

    def load(self):
        pass


if __name__ == "__main__":
    extractor = BinaryBERTExtractor()
    print(extractor.exp_dir)
