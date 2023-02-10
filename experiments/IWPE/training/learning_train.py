from sklearn.metrics import classification_report
from tiltify.models.gaussian_nb_model import GaussianNBModel
from tiltify.models.sentence_bert import SentenceBert
from tiltify.models.test_model import TestModel
from tiltify.models.binary_bert_model import BinaryBERTModel
from collections import defaultdict
from tiltify.config import Path
import os
import json
import pathlib
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.extractors.k_ranker import KRanker
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_documents(document_collection, train_doc_size):
    doc_col_len = len(document_collection)
    doc_cnt = int(doc_col_len * train_doc_size)
    return document_collection[:(doc_cnt-1)]


def eval_model(model, doc_set, k_ranks):
    print(f"Starting evaluation of {model.__class__.__name__}...")
    metrics_dict = {}
    found_doc = []
    real_doc = []
    for document in tqdm(doc_set[:5]):
        labels = model.preprocessor.label_retriever.retrieve_labels(document.blobs)
        labels = model.preprocessor.prepare_labels(labels)
        logits = model.predict(document)
        log_based_preds = [logit > 0.5 for logit in logits]
        metrics_dict["classify_metrics"] = classification_report(labels, log_based_preds, output_dict=True, digits=2)
        for k_rank in k_ranks:
            ranker = KRanker(k_rank)
            doc_indexes, _ = ranker.form_k_ranks(logits)
            found_blob = sum([labels[index] > 0 for index in doc_indexes]) > 0
            real_blob = sum(labels) > 0
            found_doc.append(found_blob)
            real_doc.append(real_blob)
        metrics_dict[f"{k_rank}_k_rank_metrics"] = classification_report(real_doc, found_doc, output_dict=True, digits=2)
    return metrics_dict


config = {
    "k_ranks": [5, 10, 25],
    "label": "Right to Withdraw Consent",
    "repitions": 2,
    "step_size": 2,
    "random_state": 1337
}

step_size = config["step_size"]
train_doc_sizes = [size/10 for size in list(range(0, 10+step_size, step_size))][1:]
print(f"Training Sizes: {train_doc_sizes}")

document_collection = DocumentCollection.from_json_files()
doc_index = list(range(len(document_collection)))

train_docs, test_docs = train_test_split(
    doc_index, test_size=0.33, random_state=config["random_state"], shuffle=False)
train_docs = document_collection[train_docs]
test_set = document_collection[test_docs]
print(f"Corpus having: {len(test_docs)} Test Docs and {len(train_docs)} Train Docs.")


model_types = [
    TestModel,
    GaussianNBModel,
    SentenceBert,
    BinaryBERTModel
    ]

for model_type in model_types:
    print(f"Conducting experment for {model_type.__name__}...")
    model_cls = model_type
    model_kwargs = dict(
        label=config["label"]
    )
    exp_dir = os.path.join(Path.root_path, f"experiments/IWPE/training/{model_cls.__name__}")

    model = model_cls(**model_kwargs)
    results = defaultdict(dict)

    for k in range(config["repitions"]):
        result_dict = defaultdict(list)
        for train_doc_size in train_doc_sizes:
            save_dir = os.path.join(exp_dir, f"{train_doc_size}/{k}")
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

            train_set = get_documents(train_docs, train_doc_size)
            model.train(train_set)
            train_report = eval_model(model, train_set, config["k_ranks"])
            test_report = eval_model(model, test_set, config["k_ranks"])
            model.save(save_dir)

            # TODO: add more metrics and maybe also logits
            result_dict["train_size"].append(train_doc_size)
            result_dict["train_results"].append(train_report)
            result_dict["test_results"].append(test_report)
            results[k] = result_dict

            result_dir = os.path.join(exp_dir, "results.json")

            with open(result_dir, "w") as f:
                json.dump(results, f)

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f)

print("Done!")
