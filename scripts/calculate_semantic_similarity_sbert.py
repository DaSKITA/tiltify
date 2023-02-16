import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sentence_transformers import SentenceTransformer, util
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.preprocessing.bert_preprocessor import BERTPreprocessor
from time import gmtime, strftime

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

# Load DocumentCollection, create Preprocessor & SBERT Embedder
doc_col = DocumentCollection.from_json_files()
preprocessor = BERTPreprocessor(bert_model="dbmdz/bert-base-german-cased", binary=False)
embedder = SentenceTransformer('dbmdz/bert-base-german-cased')


# plotting function
def plotGraph(title, pos, neg):
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


# plotting pdf creation
pp = PdfPages(f'semantic_search_results_{strftime("%Y-%m-%d_%H:%M:%S", gmtime())}.pdf')

for (query_id, query_name), query in queries.items():
    positive_data = []
    negative_data = []

    for doc in doc_col:
        _, labels = preprocessor.preprocess_document(doc)

        for i, blob in enumerate(doc.blobs):
            if query_id in labels[i]:
                positive_data.append(blob.text)
            else:
                negative_data.append(blob.text)

    # Embed the query and data
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    positive_embeddings = embedder.encode(positive_data, convert_to_tensor=True)
    negative_embeddings = embedder.encode(negative_data, convert_to_tensor=True)

    # Store sentences & embeddings on disc
    with open('pos_emb_' + str(query_id) + '.pkl', "wb") as fOut:
        pickle.dump({'sentences': positive_data, 'embeddings': positive_embeddings}, fOut,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open('neg_emb_' + str(query_id) + '.pkl', "wb") as fOut:
        pickle.dump({'sentences': negative_data, 'embeddings': negative_embeddings}, fOut,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # Run the query against all positive and negative examples respectively
    pos_cos_scores = util.cos_sim(query_embedding, positive_embeddings)[0]
    neg_cos_scores = util.cos_sim(query_embedding, negative_embeddings)[0]

    # plotting
    plot1 = plotGraph(query_name, neg_cos_scores.numpy(), pos_cos_scores.numpy())
    plot2 = plotGraph(query_name, pos_cos_scores.numpy(), neg_cos_scores.numpy())
    pp.savefig(plot1)
    pp.savefig(plot2)

pp.close()
