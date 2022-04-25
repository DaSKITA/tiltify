import os

from tiltify.extractors.blob_extractor import BlobExtractor, BlobExtractorMetaClass
from tiltify.data_structures.document import Document
from tiltify.objectives.bert_objective.bert_preprocessor import BERTPreprocessor
from tiltify.objectives.bert_objective.bert_splitter import BERTSplitter
from tiltify.objectives.bert_objective.bert_binary_objective import BERTBinaryObjective
from tiltify.config import BASE_BERT_MODEL, Path


class BinaryBERTExtractor(BlobExtractor, metalass=BlobExtractorMetaClass):

    def __init__(self, testarg) -> None:
        self.testarg = testarg

    def train(self):
        # document_collection = DocumentCollection.from_json_files()
        pandas_path = os.path.join(Path.data_path, "de_sentence_data.csv")
        preprocessor = BERTPreprocessor(
            bert_model=BASE_BERT_MODEL, binary=True)
        preprocessed_dataaset = preprocessor.preprocess_pandas(pandas_path=pandas_path)
        bert_splitter = BERTSplitter(val=True, split_ratio=split_ratio, batch_size=batch_size)
        train, val, test = bert_splitter.split(preprocessed_dataaset)
        experiment = Experiment(experiment_path=self.exp_dir, title="Binary-Classification", model_name="Bert-FF")
        experiment.add_objective(BERTBinaryObjective, args=[train, val, test])
        experiment.run(k=k, trials=trials, num_processes=num_processes)

    def predict(self, document: Document):
        return super().predict(document)



if __name__ == "__main__":
    extractor = BinaryBERTExtractor()
    print(extractor.exp_dir)
