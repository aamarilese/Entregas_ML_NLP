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
        result = predict_genre(args)
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
