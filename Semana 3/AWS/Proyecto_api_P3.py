#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import numpy as np
# import joblib
from Proyecto_model_deployment_P2 import predict_popularity

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='Song Popularity Prediction API',
    description='Song Popularity Prediction API')

ns = api.namespace('predict',
                   description='Predicción de popularidad de canciones')

# ¡Registrar el namespace!
api.add_namespace(ns)

# Crear el parser
parser = api.parser()

# parser = api.parser()

# parser.add_argument(
#     'URL',
#     type=str,
#     required=True,
#     help='URL to be analyzed',
#     location='args')

# Definir todas las variables esperadas
parser.add_argument('track_id', type=str, required=True,
                    help='ID de la canción')
parser.add_argument('artists', type=str, required=True, help='Artista')
parser.add_argument('album_name', type=str, required=True,
                    help='Álbum de la canción')
parser.add_argument('track_name', type=str, required=True,
                    help='Nombre de la canción')
parser.add_argument('duration_ms', type=float, required=True,
                    help='Duración en milisegundos')
parser.add_argument('explicit', type=float, required=True,
                    help='¿Tiene contenido explícito?')
parser.add_argument('danceability', type=float,
                    required=True, help='Medida de "bailabilidad"')
parser.add_argument('energy', type=float, required=True,
                    help='Medida de energía')
parser.add_argument('key', type=float, required=True,
                    help='Tono de la canción')
parser.add_argument('loudness', type=float, required=True,
                    help='Volumen de la canción')
parser.add_argument('mode', type=float, required=True,
                    help='Modo de la canción')
parser.add_argument('speechiness', type=float,
                    required=True, help='Medida de "habladuría"')
parser.add_argument('acousticness', type=float,
                    required=True, help='Medida de "acústica"')
parser.add_argument('instrumentalness', type=float,
                    required=True, help='Medida de "instrumentalidad"')
parser.add_argument('liveness', type=float, required=True,
                    help='Medida de "vivacidad"')
parser.add_argument('valence', type=float, required=True,
                    help='Medida de "valencia"')
parser.add_argument('tempo', type=float, required=True,
                    help='Tempo de la canción')
parser.add_argument('time_signature', type=float,
                    required=True, help='Firma de tiempo')
parser.add_argument('track_genre', type=str,
                    required=True, help='Género musical')


# resource_fields = api.model('Resource', {
#     'result': fields.String,
# })

resource_fields = api.model('PredictionOutput', {
    'popularity': fields.Float(description='Popularidad predicha (0 a 100)')
})


@ns.route('/')
class PopularityPredictor(Resource):
    # @ns.expect(input_model)
    # @ns.marshal_with(resource_fields)
    @ns.expect(parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        # data = request.get_json()
        args = parser.parse_args()
        print(args)
        args = dict(args)
        result = predict_popularity(args)
        print(result)
        print(type(result))

        if isinstance(result, np.ndarray):
            result = result.item()

        print(result)

        return {
            "popularity": result
        }, 200


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
