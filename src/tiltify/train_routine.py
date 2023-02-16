from flask_executor import Executor
from tiltify.main import app

app.config['EXECUTOR_TYPE'] = 'process'
app.config['EXECUTOR_MAX_WORKERS'] = 1
executor = Executor(app)


@executor.job
def train_model(extractor_manager, data_list: DocumentCollection, label: str):
    learning_manager = LearningManager()
    learning_manager.learn(data_list)

    model.train(document_collection=data_list)
    model.save()

    payload = {
        "extractor_label": label
    }
    requests.post(f"{TILTIFY_ADD}:{TILTIFY_PORT}" + "/api/reload", json=payload, timeout=3000,
                  headers={'Content-Type': 'application/json'})
