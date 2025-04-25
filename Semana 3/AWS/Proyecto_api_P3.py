#!/usr/bin/python
from flask import Flask, request
from flask_restx import Api, Resource, fields
# import joblib
# from m09_model_deployment import predict_proba

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='Song Popularity Prediction API',
    description='Song Popularity Prediction API')

ns = api.namespace('predict',
                   description='Predicción de popularidad de canciones')

# parser = api.parser()

# parser.add_argument(
#     'URL',
#     type=str,
#     required=True,
#     help='URL to be analyzed',
#     location='args')

input_model = api.model('SongInput', {
    'title': fields.String(required=True, description='Título de la canción'),
    'artist': fields.String(required=True, description='Artista'),
    'genre': fields.String(required=True, description='Género musical'),
    'duration_sec': fields.Float(required=True, description='Duración en segundos'),
    'danceability': fields.Float(required=True, description='Medida de "bailabilidad"'),
    'energy': fields.Float(required=True, description='Nivel de energía'),
    'valence': fields.Float(required=True, description='Positividad de la canción'),
    'tempo': fields.Float(required=True, description='Tempo (BPM)'),
    'acousticness': fields.Float(required=True, description='Porcentaje acústico'),
    'explicit': fields.Boolean(required=True, description='¿Tiene contenido explícito?')
})

# resource_fields = api.model('Resource', {
#     'result': fields.String,
# })

resource_fields = api.model('PredictionOutput', {
    'popularity': fields.Integer(description='Popularidad predicha (0 a 100)')
})


@ns.route('/')
class PhishingApi(Resource):
    @ns.expect(input_model)
    @ns.marshal_with(resource_fields)
    # @api.doc(parser=parser)
    # @api.marshal_with(resource_fields)
    def post(self):
        data = request.get_json()

        return {
            "result": predict_proba(data)
        }, 200


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
