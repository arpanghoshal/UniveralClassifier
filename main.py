from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from transformers import pipeline
import os

app = Flask(__name__)
api = Api(app)

os.environ['TRANSFORMERS_OFFLINE']='1'

class HelloWorld(Resource):
    def get(self):
        return {'about': 'Hello world'}
    def post(self):
        return {'about': 'Hello world with post'}

class UniversalDetector(Resource):
    def post(self):
        data = request.get_json()
        text = data.get('text', '')
        labels = data.get('labels', '')

        classifier = pipeline("zero-shot-classification",
                      model="model")
        labels = labels.split(',')
        answer = classifier(text, labels, multi_label=True)
        answer = dict((zip(answer['labels'], answer['scores'])))

        return {'output': answer}

api.add_resource(HelloWorld, '/')
api.add_resource(UniversalDetector, '/emotiondetector/<string:text>')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)