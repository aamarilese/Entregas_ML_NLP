#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import numpy as np
# import joblib
from Proyecto2_model_deployment_P2 import predict_genre

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='Film genre classification API',
    description='Film genre classification API')

ns = api.namespace('predict',
                   description='Clasificación de género de películas')

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
parser.add_argument('plot', type=str, required=True,
                    help='Argumento de entrada: sinopsis de la película')

# resource_fields = api.model('Resource', {
#     'result': fields.String,
# })

resource_fields = api.model('GenreProbabilities', {
    'genres': fields.Raw(
        description='Dictionary of predicted genres and their probabilities'
    )
})


@ns.route('/')
class GenreClassifier(Resource):
    @ns.expect(parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        plot = args.get('plot')

        if not plot:
            api.abort(400, "Missing 'plot' parameter.")

        try:
            # Ejecutar predicción
            df_probs = predict_genre(plot)

            # Ordenar por probabilidad
            sorted_genres = df_probs.T.sort_values(by=0, ascending=False)
            sorted_genres.columns = ['probability']

            # Filtrar si se desea (umbral)
            threshold = 0.1
            filtered = sorted_genres[sorted_genres['probability'] > threshold]

            # Convertir a diccionario
            genre_dict = filtered['probability'].round(3).to_dict()

            return {'genres': genre_dict}, 200

        except Exception as e:
            api.abort(500, f"Prediction error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
