from concurrent.futures import ProcessPoolExecutor


class LearningManager:

    def __init__(self, extractor_registry, max_batch_size=10) -> None:
        self._extractor_registry = extractor_registry
        self.max_batch_size = max_batch_size

    @staticmethod
    def _learning_surveilance(extractor_registry):
        

    def get_number_of_annotations(label):
        pass
