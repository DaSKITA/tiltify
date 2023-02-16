from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
ctx = mp.get_context("spawn")

from tiltify.data_structures.document_collection import DocumentCollection


class LearningManager:

    def __init__(self, extractor_registry, storage_size=15) -> None:
        self._extractor_registry = extractor_registry
        self._annotation_storage = {}
        self.storage_size = storage_size

    def is_storage_full(self, label):
        return len(self._annotation_storage.setdefault(label, [])) >= self.storage_size

    def process_mini_batch(self, label):
        # This function should merge the according batch into a single DocumentCollection object
        # maybe even write a merge function in DocumentCollection or
        # utils, where a batch can be merged into a single DocumentCollection?
        document_collection_batched = DocumentCollection()
        for batch_dc in self._annotation_storage.setdefault(label, []):
            try:
                idx = [doc.title for doc in document_collection_batched].index(batch_dc[0].title)
                # TODO: add_annotations function doesnt exist, it would be cool to have functionalities
                # as seen in PolicyParser for adding annotations to an existing DocumentCollection/Document object
                document_collection_batched[idx].add_annotations(batch_dc["annotations"])
            except ValueError:
                # TODO: handle batch_dc not in document_collection_batched e.g. add batch_dc as Document
                pass
        return document_collection_batched

    def train(self, labels):
        for label in labels:
            print(f"Training Model for {label}...")
            extractor = self._extractor_registry[label]
            if extractor:
                extractor.train()
                extractor.save()

    def train_all(self):
        for extractor, extractor_label in self._extractor_registry:
            print(f"Training {extractor_label} Model...")
            extractor.train()
            extractor.save()

    def _train_online_extractor(self, document_collection):
        # TODO: actually train extractors and call reload API path
        # see train_routine.py
        pass

    def train_online_collection(self, document_collection):
        # TODO: more than one label?
        label = document_collection[0]["annotations"]["annotation_label"]
        for extractor, extractor_label in self._extractor_registry:
            if extractor_label == label:
                self._annotation_storage.setdefault(extractor_label, []).append(document_collection)
                if self.is_storage_full(extractor_label):
                    with ProcessPoolExecutor(mp_context=ctx) as executor:
                        executor.submit(self._train_online_extractor(), self.process_mini_batch(extractor_label))
