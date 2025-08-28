from flask import Flask, render_template, request, jsonify, url_for
from ia import *

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/mensagem", methods=["POST"])
def mensagem():
    mensagem_usuario = request.json.get("mensagem")
    resposta = conversar(mensagem_usuario)
    print('Alguém enviou uma msg')
    return jsonify({"resposta": resposta})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
