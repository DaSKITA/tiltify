import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from operator import itemgetter
from sentence_transformers import InputExample, losses, SentenceTransformer, util
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.preprocessing.label_retriever import LabelRetriever
from time import gmtime, strftime
from torch.utils.data import DataLoader
import pandas as pd
import multiprocessing as mp
ctx = mp.get_context('spawn')
import json
from pynvml import *
from pynvml.smi import nvidia_smi
nvmlInit()
from time import sleep
from datetime import datetime

# Compiled Excerpt from DSGVO, that is used to analyze "Right To"'s in a binary matter
# RightToInformation | DSGVO Art. 13 (2b)
# RightToRectificationOrDeletion | DSGVO Art. 16 + 17
# RightToDataPortability | DSGVO Art. 20
# RightToWithdrawConsent | DSGVO Art. 13 (2c)
# RightToComplain | DSGVO Art. 13 (2d)
query = "Die betroffene Person hat das Recht auf Auskunft seitens des Verantwortlichen über die etreffenden " \
        "personenbezogenen Daten. Die betroffene Person hat das Recht auf die Berichtigung sie betreffender " \
        "unrichtiger personenbezogener Daten und auf die unverzügliche Löschung sie betreffender personenbezogener " \
        "Daten durch den Verantwortlichen. Die betroffene Person hat das Recht, die sie betreffenden personenbezogenen " \
        "Daten in einem strukturierten, gängigen und maschinenlesbaren Format zu erhalten. Die betroffene Person hat " \
        "das Recht, die Einwilligung jederzeit zu widerrufen.Die betroffene Person hat das Recht auf Beschwerde bei " \
        "einer Aufsichtsbehörde."


def load_doc_col():
    # load complete DocumentCollection and create LabelRetriever
    unprocessed_doc_col = DocumentCollection.from_json_files()
    label_retriever = LabelRetriever()
    processed_doc_col = {}

    # preprocess DocumentCollection
    for idx, doc in enumerate(unprocessed_doc_col):
        labels = label_retriever.retrieve_labels(doc.blobs)
        processed_doc_col[idx] = ([blob.text for blob in doc.blobs], labels)

    return processed_doc_col


def split_doc_col_binary(doc_col, with_docs=False):
    positive_doc_ids = []
    for idx, doc_tuple in doc_col.items():
        if {1, 2, 3, 4, 5} & set([label for labels in doc_tuple[1] for label in labels]):
            positive_doc_ids.append(idx)

    # split indices array
    _, test_idx = train_test_split(positive_doc_ids, test_size=0.33, random_state=42)

    # create splitted lists of documents using the indices array
    train_docs = itemgetter(*[id for id in list(range(len(doc_col))) if id not in test_idx])(doc_col)
    test_docs = itemgetter(*test_idx)(doc_col)

    # flatten list of tuples of lists
    train_data = [[blob_text, doc_tuple[1][idx]] for doc_tuple in train_docs
                   for idx, blob_text in enumerate(doc_tuple[0])]
    test_data = [[blob_text, doc_tuple[1][idx]] for doc_tuple in test_docs
                  for idx, blob_text in enumerate(doc_tuple[0])]
    if not with_docs:
        return train_data, test_data
    else:
        return train_data, test_data, train_docs, test_docs


def plot_graph(title, pos, neg):
    fig = plt.figure()
    sns.distplot(neg, label="Negative Data")
    sns.distplot(pos, label="Positive Data")
    plt.legend()
    fig.suptitle(f"Similarity for {title}", fontsize=14)
    plt.xlabel('Cosine Similarity', fontsize=18)
    plt.ylabel('Density', fontsize=16)
    plt.text(plt.xlim()[0], plt.ylim()[1] * 0.85, f"\nPositive Data\nMean: {np.mean(pos)}\nMedian: {np.median(pos)}")
    plt.text(plt.xlim()[0], plt.ylim()[1] * 0.7, f"\nNegative Data\nMean: {np.mean(neg)}\nMedian: {np.median(neg)}")
    return fig


def form_triplets(query, data):
    positive_data = []
    negative_data = []
    for blobs, labels in data:
        for i, blob in blobs:
            if 0 not in labels[i]:
                positive_data.append(blob.text)
            else:
                negative_data.append(blob.text)

    triplet_data = [InputExample(texts=combination)
                    for combination in itertools.product([query], positive_data, negative_data)]
    return triplet_data


def evaluation(model, query, query_name, positive_data, negative_data, pp, label=None):
    # embed the query and data
    query_embedding = model.encode(query, convert_to_tensor=True)
    positive_embeddings = model.encode(positive_data, convert_to_tensor=True)
    negative_embeddings = model.encode(negative_data, convert_to_tensor=True)

    # Run the query against all positive and negative examples respectively
    pos_cos_scores = util.cos_sim(query_embedding, positive_embeddings)[0].cpu()
    neg_cos_scores = util.cos_sim(query_embedding, negative_embeddings)[0].cpu()

    # plotting
    plot1 = plot_graph(
        query_name + "_" + label if label else query_name, pos_cos_scores.numpy(), neg_cos_scores.numpy())
    pp.savefig(plot1)


def record_watt(queue, model_dir):
    power_draws = []
    while queue.get():
        inst = nvidia_smi.getInstance()
        memory = inst.DeviceQuery("memory.used")
        power_draw = inst.DeviceQuery("power.draw")
        timestamp = datetime.timestamp(datetime.now())
        stats = {"memory_usage": memory, "power_draw": power_draw, "timestamp": timestamp}
        power_draws.append(stats)
        if queue.empty():
            queue.put(True)
        sleep(5)
    with open(os.path.join(model_dir, "power_draws.json"), "w") as f:
        json.dump(power_draws, f)


def train_with_power_draw(model, train_dataloader, train_loss, model_dir):
    queue = ctx.Queue()
    queue.put(True)
    p = ctx.Process(target=record_watt, args=(queue, model_dir,))
    p.start()
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=100)
    queue.put(False)
    p.join()
    sleep(2)
    p.close()
    return model


if __name__ == "__main__":
    # directory structure
    directory_name = f'binary_triplet_semantic_search_results_{strftime("%Y-%m-%d_%H:%M:%S", gmtime())}'
    curr_dir = os.path.join(os.path.dirname(__file__), "triplet_training")
    exp_dir = os.path.join(curr_dir, directory_name)
    models_dir = os.path.join(exp_dir, 'models')
    os.makedirs(exp_dir)
    os.makedirs(models_dir)

    # plotting pdf creation
    pp = PdfPages(exp_dir + '/evaluation.pdf')

    # loading and preprocessing DocumentCollection
    doc_col = load_doc_col()
    power_draws = {}

    print(30*"#"+f" Starting with binary training "+30*"#")
    model_name = "binary_triplet_semantic_search_".lower().replace(" ", "_")
    model_dir = os.path.join(models_dir, model_name)
    os.makedirs(model_dir)
    positive_train_data, negative_train_data, positive_test_data, negative_test_data = [], [], [], []
    train_docs, test_docs = split_doc_col_binary(doc_col)

    # # for testing
    # test_docs = test_docs[:5]
    # train_docs = train_docs[:10]Q
    # test_docs[1][1] = [query_id]
    # train_docs[1][1] = [query_id]

    # Process train data
    for blob, labels in train_docs:
        if {1, 2, 3, 4, 5} & set(labels):
            positive_train_data.append(blob)
        else:
            negative_train_data.append(blob)
    train_data = [InputExample(texts=combination)
                  for combination in itertools.product([query], positive_train_data, negative_train_data)]

    # Process test_data
    for blob, labels in test_docs:
        if {1, 2, 3, 4, 5} & set(labels):
            positive_test_data.append(blob)
        else:
            negative_test_data.append(blob)

    # initiate model and measure pre-training performance
    model = SentenceTransformer('dbmdz/bert-base-german-cased')
    evaluation(model, query, "Binary 'Right To's Classification", positive_test_data, negative_test_data, pp, label="pre-training")

    # Train the model
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=10)
    train_loss = losses.TripletLoss(model, triplet_margin=5)
    # model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=100)
    model = train_with_power_draw(model, train_dataloader, train_loss, model_dir)
    evaluation(model, query, "Binary 'Right To's Classification", positive_test_data, negative_test_data, pp, label="post-training")

    # save trained model
    model.save(path=model_dir, model_name=model_name)
    pp.close()
