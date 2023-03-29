from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.models.binary_bert_model import BinaryBERTModel
from tiltify.models.gaussian_nb_model import GaussianNBModel
from tiltify.models.sentence_bert import SentenceBert


document_collection = DocumentCollection.from_json_files()

for model_cls in [BinaryBERTModel, GaussianNBModel, SentenceBert]:
    all_labels = []
    model = model_cls(label="Right to Withdraw Consent")
    for document in document_collection:
        labels = model.preprocessor.label_retriever.retrieve_labels(document.blobs)
        labels = model.preprocessor.prepare_labels(labels)
        present = sum(labels) > 0
        all_labels.append(present)
    print(f"{model_cls.__name__} Number of Real Consumer Right Docs: {sum(all_labels)} out of {len(document_collection)}")
