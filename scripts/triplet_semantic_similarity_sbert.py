import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from sentence_transformers import InputExample, losses, SentenceTransformer, util
from sklearn.model_selection import train_test_split
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.objectives.bert_objective.bert_preprocessor import BERTPreprocessor
from tiltify.preprocessing.label_retriever import LabelRetriever
from time import gmtime, strftime
from torch.utils.data import DataLoader

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
    unprocessed_doc_col = DocumentCollection.from_json_files()
    label_retriever = LabelRetriever()
    processed_doc_col = {}
    for idx, doc in enumerate(unprocessed_doc_col):
        labels = label_retriever.retrieve_labels(doc.blobs)
        processed_doc_col[idx] = ([blob.text for blob in doc.blobs], labels)
    return processed_doc_col


def split_doc_col(doc_col, query_id):
    positive_docs = []
    for idx, doc_tuple in doc_col.items():
        if query_id in [label for labels in doc_tuple[1] for label in labels]:
            positive_docs.append(idx)
    train_idx, test_idx = train_test_split(positive_docs, test_size=0.33, random_state=42)
    # TODO: implement split
    return ([], [])

def plot_graph(title, pos, neg):
    fig = plt.figure()
    sns.distplot(neg, label="Negative Data")
    sns.distplot(pos, label="Positive Data")
    plt.legend()
    fig.suptitle(f"Semantic Search Similarity for {title}", fontsize=20)
    plt.xlabel('Cosine Similarity', fontsize=18)
    plt.ylabel('Density', fontsize=16)
    plt.text(plt.xlim()[0], plt.ylim()[1] * 0.85, f"\nPositive Data\nMean: {np.mean(pos)}\nMedian: {np.median(pos)}")
    plt.text(plt.xlim()[0], plt.ylim()[1] * 0.7, f"\nNegative Data\nMean: {np.mean(neg)}\nMedian: {np.median(neg)}")
    return fig


if __name__ == "__main__":
    # plotting pdf creation
    pp = PdfPages(f'triplet_semantic_search_results_{strftime("%Y-%m-%d_%H:%M:%S", gmtime())}.pdf')

    # loading and preprocessing DocumentCollection
    doc_col = load_doc_col()

    for (query_id, query_name), query in queries.items():
        positive_data = []
        negative_data = []
        test_docs, train_docs = split_doc_col(doc_col, query_id)

        for blobs, labels in train_docs:
            for i, blob in blobs:
                if query_id in labels[i]:
                    positive_data.append(blob.text)
                else:
                    negative_data.append(blob.text)

        train_data = [InputExample(texts=combination) for combination in itertools.product([query], positive_data, negative_data)]

        # Train the model
        model = SentenceTransformer('dbmdz/bert-base-german-cased')
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
        train_loss = losses.TripletLoss(model, triplet_margin=5)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=100)

        # Embed the query and data
        query_embedding = model.encode(query, convert_to_tensor=True)
        positive_embeddings = model.encode(positive_data, convert_to_tensor=True)
        negative_embeddings = model.encode(negative_data, convert_to_tensor=True)

        # Run the query against all positive and negative examples respectively
        pos_cos_scores = util.cos_sim(query_embedding, positive_embeddings)[0]
        neg_cos_scores = util.cos_sim(query_embedding, negative_embeddings)[0]

        # plotting
        plot1 = plot_graph(query_name, neg_cos_scores.numpy(), pos_cos_scores.numpy())
        plot2 = plot_graph(query_name, pos_cos_scores.numpy(), neg_cos_scores.numpy())
        pp.savefig(plot1)
        pp.savefig(plot2)

    pp.close()
