import requests
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
ctx = mp.get_context("fork")

from tiltify.config import INTERNAL_KEY, TILTIFY_ADD, TILTIFY_PORT
from tiltify.data_structures.document_collection import DocumentCollection


class LearningManager:

    def __init__(self, extractor_registry, storage_size=15) -> None:
        self._extractor_registry = extractor_registry
        self._annotated_collection_storage = {}
        self.storage_size = storage_size

    def get_storage(self):
        return self._annotated_collection_storage

    def is_storage_full(self, label):
        return len(self._annotated_collection_storage.setdefault(label, [])) >= self.storage_size

    def process_mini_batch(self, label):
        document_collection_batched = DocumentCollection([])
        for batch_dc in self._annotated_collection_storage.setdefault(label, []):
            doc = batch_dc[0]
            try:
                idx = [doc.title for doc in document_collection_batched].index(doc.title)
                document_collection_batched[idx].copy_annotations(doc)
            except ValueError:
                document_collection_batched.documents.append(doc)
        return document_collection_batched  # if batch_dc else None

    @staticmethod
    def storage_train(data, extractor, label):
        print("starting concurrent training")
        extractor.train_online(document_collection=data)
        print("training finished")
        extractor.save()
        print("model saved")

        payload = {
            "key": INTERNAL_KEY,
            "extractor_label": label
        }

        print("starting request")
        requests.post(f"http://{TILTIFY_ADD}:{TILTIFY_PORT}" + "/api/reload", json=payload, timeout=3000,
                      headers={'Content-Type': 'application/json'})
        pass

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

    def train_online(self, document_collection, label):
        # TODO: more than one label?
        extractor = self._extractor_registry[label]
        if extractor:
            self._annotated_collection_storage.setdefault(label, []).append(document_collection)
            print(f"received annotation for {label}. now at {len(self._annotated_collection_storage[label])}/{self.storage_size} entries")
            if self.is_storage_full(label):
                print("starting concurrent training now!")
                with ProcessPoolExecutor(mp_context=ctx) as executor:
                    executor.submit(self.storage_train,
                                    self.process_mini_batch(label),
                                    extractor,
                                    label)


