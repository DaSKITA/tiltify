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
import logging


# Excerpts from DSGVO, that define the analyzed rights
# RightToInformation | DSGVO Art. 13 (2b)
# RightToRectificationOrDeletion | DSGVO Art. 16 + 17
# RightToDataPortability | DSGVO Art. 20
# RightToWithdrawConsent | DSGVO Art. 13 (2c)
# RightToComplain | DSGVO Art. 13 (2d)
# These law texts are used as our queries
queries = {(1,
            "Right To Information"): "Das Bestehen eines Rechts auf Auskunft seitens des Verantwortlichen über die "
                                     "betreffenden personenbezogenen Daten.",
           (2,
            "Right To Rectification Or Deletion"): "Die betroffene Person hat das Recht, von dem Verantwortlichen "
                                                   "unverzüglich die Berichtigung sie betreffender unrichtiger "
                                                   "personenbezogener Daten zu verlangen. Die betroffene Person hat "
                                                   "das Recht, von dem Verantwortlichen zu verlangen, dass sie "
                                                   "betreffende personenbezogene Daten unverzüglich gelöscht werden.",
           (3,
            "Right To Data Portability"): "Die betroffene Person hat das Recht, die sie betreffenden personenbezogenen "
                                          "Daten, die sie einem Verantwortlichen bereitgestellt hat, in einem "
                                          "strukturierten, gängigen und maschinenlesbaren Format zu erhalten.",
           (4, "Right To Withdraw Consent"): "Das Bestehen eines Rechts, die Einwilligung jederzeit zu widerrufen.",
           (5, "Right To Complain"): "Das Bestehen eines Beschwerderechts bei einer Aufsichtsbehörde."}


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


def split_doc_col(doc_col, query_id, with_docs=False):
    positive_doc_ids = []
    for idx, doc_tuple in doc_col.items():
        if query_id in [label for labels in doc_tuple[1] for label in labels]:
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
            if query_id in labels[i]:
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


def record_watt(queue):
    while queue.get():
        inst = nvidia_smi.getInstance()
        memory = inst.DeviceQuery("memory.used")
        power_draw = inst.DeviceQuery("power.draw")
        timestamp = datetime.timestamp(datetime.now())
        stats = {"memory_usage": memory, "power_draw": power_draw, "timestamp": timestamp}
        stats = json.dumps(stats)
        logging.debug(stats)
        if queue.empty():
            queue.put(True)
        sleep(5)


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
    directory_name = f'triplet_semantic_search_results_{strftime("%Y-%m-%d_%H:%M:%S", gmtime())}'
    curr_dir = os.path.join(os.path.dirname(__file__), "triplet_training")
    exp_dir = os.path.join(curr_dir, directory_name)
    logging_file = os.path.join(exp_dir, "run.log")
    print(f"Writing Log to: {logging_file}")
    models_dir = os.path.join(exp_dir, 'models')
    os.makedirs(exp_dir)
    os.makedirs(models_dir)
    logging.basicConfig(
        filename=logging_file, encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')

    # plotting pdf creation
    pp = PdfPages(exp_dir + '/evaluation.pdf')

    # loading and preprocessing DocumentCollection
    doc_col = load_doc_col()
    power_draws = {}
    queue = ctx.Queue()
    queue.put(True)
    p = ctx.Process(target=record_watt, args=(queue,))
    p.start()

    for (query_id, query_name), query in queries.items():
        print(30*"#"+f" Starting with {query_name} "+30*"#")
        logging.debug(30*"#"+f" Starting with {query_name} "+30*"#")
        model_name = "triplet_semantic_search_" + query_name.lower().replace(" ", "_")
        model_dir = os.path.join(models_dir, model_name)
        os.makedirs(model_dir)
        positive_train_data, negative_train_data, positive_test_data, negative_test_data = [], [], [], []
        train_docs, test_docs = split_doc_col(doc_col, query_id)

        # for testing
        test_docs = test_docs[:5]
        train_docs = train_docs[:10]
        test_docs[1][1] = [query_id]
        train_docs[1][1] = [query_id]

        # Process train data
        for blob, labels in train_docs:
            if query_id in labels:
                positive_train_data.append(blob)
            else:
                negative_train_data.append(blob)
        train_data = [InputExample(texts=combination)
                      for combination in itertools.product([query], positive_train_data, negative_train_data)]

        # Process test_data
        for blob, labels in test_docs:
            if query_id in labels:
                positive_test_data.append(blob)
            else:
                negative_test_data.append(blob)

        # initiate model and measure pre-training performance
        model = SentenceTransformer('dbmdz/bert-base-german-cased')
        evaluation(model, query, query_name, positive_test_data, negative_test_data, pp, label="pre-training")

        # Train the model
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=10)
        train_loss = losses.TripletLoss(model, triplet_margin=5)
        logging.debug("Start Training")
        # model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=100)
        logging.debug("End Training")
        # model = train_with_power_draw(model, train_dataloader, train_loss, model_dir)
        evaluation(model, query, query_name, positive_test_data, negative_test_data, pp, label="post-training")

        # save trained model
        model.save(path=model_dir, model_name=model_name)
        break
    pp.close()
    queue.put(False)
    p.join()
    sleep(2)
    p.close()
