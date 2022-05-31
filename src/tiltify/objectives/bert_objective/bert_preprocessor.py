from typing import Dict, List
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer, BatchEncoding

from tiltify.config import BASE_BERT_MODEL
from tiltify.preprocessing.preprocessor import Preprocessor
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.data_structures.document import Document


class TiltDataset(Dataset):
    """
    A dataset class used to load the contents of a CSV
    dataset file.

    ...

    Attributes
    ----------
    st : BatchEncoding
        huggingface transformers data structure which contains
        the padded and tokenized blobs
    id : Tensor
        a Tensor containing all the sentence ids related to the
        padded and tokenized blobs in st
    labels : Tensor
        a Tensor containing all the target values related to the
        padded and tokenized blobs in st

    Methods
    -------
    __len__()
        returns the length of the dataset
    __getitem__(idx)
        returns the tokenized sentence and the id of the
        sentence at index idx
    """

    def __init__(self, blobs: Dict, labels: List):
        """
        Parameters
        ----------
        data : Dict
            a dict containing both the padded and tokenized
            blobs and the ids of those blobs
        """
        self.input_ids = blobs["input_ids"]
        self.attention_mask = blobs["attention_mask"]
        self.token_type_ids = blobs["token_type_ids"]
        self.labels = torch.Tensor(labels)

    def __len__(self) -> int:
        """returns the length of the dataset"""
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        """returns the tokenized sentence and the id of the sentence at index idx

        Parameters
        ----------
        idx : int
            specifies the index of the data to be returned
        """
        output_dict = dict(
            input_ids=self.input_ids[idx],
            attention_mask=self.attention_mask[idx],
            token_type_ids=self.token_type_ids[idx]
        )
        if self.labels is not None:
            output_dict["labels"] = self.labels[idx]
        return output_dict


class BERTPreprocessor(Preprocessor):

    def __init__(
            self, bert_model: str = None, binary: bool = False) -> None:
        if bert_model:
            self.bert_model = bert_model
        else:
            self.bert_model = BASE_BERT_MODEL
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        self.binary = binary

    def preprocess(self, document_collection: DocumentCollection):
        """This preprocessing function creates a corpus where all documents form a list of sentences and labels.
        Documents are not provided by document for document. The document level representation is not used.

        Args:
            document_collection (DocumentCollection): _description_

        Returns:
            _type_: _description_
        """
        preprocessed_corpus = []
        corpus_labels = []
        corpus_blobs = []
        for document in document_collection:
            document_labels = self.label_retriever.retrieve_labels(document.blobs)
            document_labels = self.prepare_labels(document_labels)
            corpus_blobs += [blob.text for blob in document.blobs]
            corpus_labels.append(document_labels)
        corpus_labels = sum(corpus_labels, [])
        preprocessed_corpus = self._tokenize_blobs(corpus_blobs)
        return TiltDataset(preprocessed_corpus, corpus_labels)

    def preprocess_document(self, document: Document):
        """Used for inference. If a single document is provided this document has to be used for prediction.

        Args:
            document (Document): _description_

        Returns:
            _type_: _description_
        """
        return self._tokenize_blobs([blob.text for blob in document.blobs])

    def prepare_labels(self, labels: List):
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

    def _tokenize_blobs(self, blobs: List[str]) -> BatchEncoding:
        """Tokenizes and pads a list of blobs using the model specified.

        Parameters
        ----------
        blobs : List[str]
            a list containing all blobs that are to be tokenized
            in this function
        bert_base_model : str
            specifies which huggingface transformers model is to be
            used for the tokenization

        Returns
        -------
        BatchEncoding
            huggingface transformers data structure which contains
            the padded and tokenized blobs

        """
        return self.bert_tokenizer(
            blobs, padding=True, truncation=True, return_tensors="pt")


if __name__ == "__main__":
    document_collection = DocumentCollection.from_json_files()
    preprocessor = BERTPreprocessor(
        bert_model=BASE_BERT_MODEL, binary=True)
    preprocessed_dataaset = preprocessor.preprocess(document_collection)
    print("Done")
