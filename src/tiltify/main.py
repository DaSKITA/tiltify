from flask import Blueprint, Flask, request
from flask_jwt_extended import create_access_token, jwt_required, JWTManager
from flask_restx import Api, fields, Resource

# Initialize Flask App
app = Flask(__name__)

from tiltify.config import EXTRACTOR_CONFIG, FlaskConfig, INTERNAL_KEY, TILTIFY_ADD, TILTIFY_PORT
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.extractors.extractor import ExtractorManager
from tiltify.parsers.policy_parser import PolicyParser

# initialize ExtractorManager, PolicyParser and load FlaskConfig
extractor_manager = ExtractorManager(EXTRACTOR_CONFIG)
extractor_manager.load_all()
policy_parser = PolicyParser()
app.config.from_object(FlaskConfig)

# API namespace
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization'
    }
}

# API setup
api_bp = Blueprint("api", __name__, url_prefix="/api/")
api = Api(api_bp, version='1.0', title='TILTify API', doc='/docs/',
          description='A simple API granting access to model training and Annotation predictions of TILTify',
          authorizations=authorizations, default='TILTify auth, train & predict',
          default_label='Contains paths to authorization for or training and predicting with TILTify',)
app.register_blueprint(api_bp)

jwt = JWTManager(app)

# API marshaling objects
password = api.model('AuthPassword', {
    'password': fields.String(required=True, description="Password")
})

document = api.model('Document', {
    'document_name': fields.String(required=True, description="Name of the company to which data belongs"),
    'text': fields.String(required=True, description="Data to be analyzed in the prediction")
})

annotation = api.model('Annotation', {
    'annotation_label': fields.String(required=True, description='Label to which this annotation belongs'),
    'annotation_text': fields.String(required=True, description='Text of annotation'),
    'annotation_start': fields.Integer(required=True, description='Starting position of text annotation in the document text'),
    'annotation_end': fields.Integer(required=True, description='Ending position of annotation in the document text')
})

annotated_document = api.model('PredictionDocument', {
    'document': fields.Nested(document, required=True, description='Document data'),
    'annotations': fields.List(fields.Nested(annotation), required=True,
                               description='List of annotations available for this document')
})

predict_annotation = api.model('PredictAnnotation', {
    'label': fields.String(required=True, description='Label to which this annotation belongs'),
    'start': fields.Integer(required=True, description='Starting position of annotation in the document text'),
    'end': fields.Integer(required=True, description='Ending position of annotation in the document text'),
    'text': fields.String(required=True, description='Text of annotation')
})

predict_input = api.model('PredictInput', {
    'document': fields.Nested(annotated_document, required=True),
    'labels': fields.List(fields.String(description='Label to calculate predictions for'), required=True)
})

predict_output = api.model('PredictOutput', {
    'predictions': fields.List(fields.Nested(predict_annotation), required=True)
})

train_input = api.model('TrainInput', {
    'documents': fields.List(fields.Nested(annotated_document), required=True),
    'labels': fields.List(fields.String(description='Labels to which the documents to be trained belong'))
})

reload_instructuin = api.model('ReloadInstruction', {
    'extractor_label': fields.String(description='Label specifying Extractor to update', required=True)
})


# API authentication
@api.route("/auth")
class Authentication(Resource):

    @api.expect(password)
    def post(self):
        password = request.json.get("password", None)

        if password == FlaskConfig.JWT_SECRET_KEY:
            access_token = create_access_token(identity="TILTer")
            return access_token, 200
        else:
            return {"msg": "Wrong password"}, 401


# API paths
@api.route('/train')
class Train(Resource):

    # Exchange with a CronJob and poll database directly

    @api.expect(train_input)
    @api.doc(security='apikey')
    @jwt_required()
    def post(self):
        # Websockets? : https://blog.miguelgrinberg.com/post/add-a-websocket-route-to-your-flask-2-x-application
        try:
            json_document_list = request.json.get('documents')
            labels = request.json.get("labels")
            document_collection = DocumentCollection.from_json_dict(json_document_list)
            extractor_manager.train_online(labels, document_collection)
        except Exception as e:
            return f"Error: {e}", 500
        return "Training started", 202


@api.route('/predict')
class Predict(Resource):

    @api.expect(predict_input)
    @api.marshal_with(predict_output)
    @api.doc(security='apikey')
    @jwt_required()
    def post(self):
        """
        :return: list of predictions.
        """
        predict_input = request.json.get("document")
        labels = request.json.get("labels")
        document = policy_parser.parse(
            **predict_input["document"], annotations=predict_input["annotations"])
        predictions = extractor_manager.predict(labels, document, predict_input["document"]["text"])
        predictions = {"predictions": [prediction.to_dict() for prediction in predictions]}
        return predictions, 200

@api.route('/reload')
class Reload(Resource):

    @api.expect(reload_instruction)
    def post(self):
        if request.json.get("key") == INTERNAL_KEY:
            extractor_label = request.json.get("extractor_label")
            ExtractorManager.load(label)
            return "Success!", 200
        else:
            return "Unauthorized", 401

if __name__ == "__main__":
    app.run(
        host=TILTIFY_ADD, port=TILTIFY_PORT, use_debugger=False, use_reloader=False,
        passthrough_errors=True)
