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
from datetime import datetime
from sklearn.metrics import brier_score_loss


def get_documents(document_collection, train_doc_size):
    doc_col_len = len(document_collection)
    doc_cnt = int(doc_col_len * train_doc_size)
    return document_collection[:(doc_cnt-1)]


def eval_model(model, doc_set, k_ranks):
    print(f"Starting evaluation of {model.__class__.__name__}...")
    metrics_dict = {}

    all_logits = []
    all_labels = []
    for document in tqdm(doc_set):
        labels = model.preprocessor.label_retriever.retrieve_labels(document.blobs)
        labels = model.preprocessor.prepare_labels(labels)
        logits = model.predict(document)
        # log_based_preds = [logit > 0.5 for logit in logits]
        all_logits.append(logits)
        all_labels.append(labels)

    for k_rank in k_ranks:
        found_doc = []
        real_doc = []
        for idx, logits in enumerate(all_logits):
            doc_labels = all_labels[idx]
            ranker = KRanker(k_rank)
            doc_indexes, _ = ranker.form_k_ranks(logits)
            found_blob = sum([doc_labels[index] > 0 for index in doc_indexes]) > 0
            real_blob = sum(doc_labels) > 0
            found_doc.append(found_blob)
            real_doc.append(real_blob)
        metrics_dict[f"{k_rank}_k_rank_metrics"] = classification_report(real_doc, found_doc, output_dict=True, digits=2, zero_division=0)
    print(f"!!!!!!!!!! Found: {sum(found_doc)}, Real: {sum(real_doc)}")
    metrics_dict["classify_metrics"] = []
    # metrics_dict["all_logits"] = all_logits
    # metrics_dict["all_labels"] = all_labels
    all_labels = sum(all_labels, [])
    all_logits = sum(all_logits, [])
    all_preds = [1 if logit > 0.5 else 0 for logit in all_logits]
    metrics_dict["classify_metrics"] = classification_report(all_labels, all_preds, output_dict=True, digits=2, zero_division=0)
    if isinstance(model, SentenceBert):
        all_labels = [-1 if i == 0 else i for i in all_labels]
    metrics_dict["brier_loss"] = brier_score_loss(all_labels, all_logits)
    return metrics_dict


config = {
    "k_ranks": [5, 10, 25],
    "repitions": 2,
    "step_size": 2,
    "random_state": 1337
}

step_size = config["step_size"]
train_doc_sizes = [1]  # [size/10 for size in list(range(0, 10+step_size, step_size))][1:]
print(f"Training Sizes: {train_doc_sizes}")

document_collection = DocumentCollection.from_json_files()
doc_index = list(range(len(document_collection)))

train_docs, test_docs = train_test_split(
    doc_index, test_size=0.33, random_state=config["random_state"], shuffle=False)
train_docs = document_collection[train_docs]
test_set = document_collection[test_docs]
print(f"Corpus having: {len(test_docs)} Test Docs and {len(train_docs)} Train Docs.")


model_types = [
    TestModel
    #BinaryBERTModel,
    #SentenceBert,
    #GaussianNBModel
    ]

right_tos = [
    'Right to Information',
    'Right to Deletion',
    'Right to Data Portability',
    'Right to Withdraw Consent',
    'Right to Complain'
]
for model_type in model_types:
    print(f"#### Conducting experment for {model_type.__name__}... ####")
    start_time = datetime.now()
    model_cls = model_type
    exp_dir = os.path.join(Path.root_path, f"experiments/IWPE/training/{model_cls.__name__}")
    result_dir = os.path.join(exp_dir, "right_to_results_2.json")
    if os.path.isfile(result_dir):
        with open(result_dir, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results in {result_dir}")
    else:
        results = defaultdict(dict)
    for k in range(config["repitions"]):
        for right_to in right_tos:
            try:
                result_dict = {}
                save_dir = os.path.join(exp_dir, f"{right_to}/{k}")
                pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
                model = model_cls(label=right_to)
                train_set = get_documents(train_docs, 1)
                print(f"Starting training of {model_cls.__name__}...")
                model.train(train_set)
                model.save(save_dir)
                train_report = eval_model(model, train_set, config["k_ranks"])
                test_report = eval_model(model, test_set, config["k_ranks"])

                # TODO: add more metrics and maybe also logits
                result_dict["train_results"] = train_report
                result_dict["test_results"] = test_report
                results[right_to][k] = result_dict

                with open(result_dir, "w") as f:
                    json.dump(results, f)
            except RuntimeError:
                print(f"{model_type} for {right_to} could not be trained. Skipping...")
    diff = datetime.now() - start_time
    config["duration"] = diff.total_seconds()
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f)
    print(f"### Experiment for {model_type.__name__} ended after {diff.total_seconds()} seconds. ###")

print("Done!")
