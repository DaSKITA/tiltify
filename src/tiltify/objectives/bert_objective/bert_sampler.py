import random
from typing import Tuple, List
from tiltify.objectives.bert_objective.bert_splitter import TiltFinetuningDataset


class BERTSampler:
    def __init__(self, n_upsample: float, n_downsample: float) -> None:
        self.n_upsample = n_upsample
        self.n_downsample = n_downsample

    def upsample(self, sentences, labels):
        minority = [idx for idx, label in enumerate(labels) if label == 1]
        upsampled_minorities = random.choices(minority, k=int(self.n_upsample * len(minority)))
        sentences += [sentences[idx] for idx in upsampled_minorities]
        labels += [labels[idx] for idx in upsampled_minorities]
        return sentences, labels

    def downsample(self, sentences, labels):
        majority = [idx for idx, label in enumerate(labels) if label == 0]
        minority = [idx for idx, label in enumerate(labels) if label == 1]
        downsampled_majorities = random.choices(majority, k=int(self.n_downsample * len(majority)))
        sentences = [sentences[idx] for idx in downsampled_majorities] + [sentences[idx] for idx in minority]
        labels = [labels[idx] for idx in downsampled_majorities] + [labels[idx] for idx in minority]
        return sentences, labels

    def sample(self, dataset: TiltFinetuningDataset) -> Tuple[List[str], List[int]]:
        sentences = dataset.dataset.dataset.sentences
        labels = dataset.dataset.dataset.labels
        if self.n_upsample:
            sentences, labels = self.upsample(sentences, labels)
        if self.n_downsample:
            sentences, labels = self.downsample(sentences, labels)
        return sentences, labels
