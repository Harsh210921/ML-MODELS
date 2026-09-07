from flask import Flask, request, jsonify
from flask import Flask
from flask_cors import CORS
app=Flask(__name__)
CORS(app)
@app.route("/test")
def test():
 return "THIS IS BACKEND : TEST SUCCESSFULL"

@app.route("/send", methods=["POST"])
def collect():
    data = request.get_json()

    print(data)
    return "DATA RECIEVED"
app.run()