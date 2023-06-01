from pickle import FALSE
from flask import Flask, request, jsonify
app = Flask(__name__)
import recommend_ai

@app.route('/recommend/<int:id>', methods=['GET'])
def respond(id):
    response = {}

    # Check if the user sent a name at all
    if not id:
        response["ERROR"] = "No product's id found. Please send a id."
    # Check if the user entered a number
    elif str(id).isdigit() == FALSE:
        response["ERROR"] = "The product's id must be numeric."
    else:
        # print(recommend_AI.recommend(name))
        response["data"] = recommend_ai.recommend(id)
        # Return the response in json format
        return jsonify(response)

@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Welcome to our recommend-AI-api!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)