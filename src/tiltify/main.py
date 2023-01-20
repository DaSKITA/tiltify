from flask import Blueprint, Flask, request
from flask_jwt_extended import create_access_token, jwt_required, JWTManager
from flask_restx import Api, fields, Namespace, Resource

from tiltify.config import EXTRACTOR_MODEL, FlaskConfig
from tiltify.data_structures.document_collection import DocumentCollection
from tiltify.extractors.extractor import Extractor

# Initialize Flask App
extractor = Extractor(extractor_type=EXTRACTOR_MODEL, extractor_label="test_label")  # TODO: change to "BinaryBert" and real label
app = Flask(__name__)
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
    'annotation_start': fields.Integer(required=True, description='Starting position of annotation in the document text'),
    'annotation_end': fields.Integer(required=True, description='Ending position of annotation in the document text')
})

prediction_document = api.model('PredictionDocument', {
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
    'document': fields.Nested(prediction_document, required=True),
    'labels': fields.List(fields.String(description='Label to calculate predictions for'), required=True)
})

predict_output = api.model('PredictOutput', {
    'predictions': fields.List(fields.Nested(predict_annotation), required=True)
})

train_input = api.model('TrainInput', {
    'document': fields.List(fields.Nested(document), required=True)
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

    @api.expect(train_input)
    @api.doc(security='apikey')
    @jwt_required()
    def post(self):
        """
        :return: list of tasks.
        """
        try:
            json_document_list = request.json.get('document')
            document_collection = DocumentCollection.from_json_dict(json_document_list)
            extractor.train_online(document_collection)
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
        :return: list of tasks.
        """
        try:
            # TODO: get task and labels from payload and process
            task = None
            predictions = extractor.predict(task)
        except Exception as e:
            return f"Error: {e}", 500
        return predictions, 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0", port="5001", use_debugger=False, use_reloader=False,
        passthrough_errors=True)
