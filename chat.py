from flask import request, jsonify
from flask_cors import CORS, cross_origin

from config import app
from qa_chain import qa_chain, conv_chain

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/ask-ai', methods=['POST'])
@cross_origin()
def show():
    query = request.json['query']
    response, source_docs = qa_chain(query)

    return jsonify(resp=response)


@app.route('/ask-ai-conv', methods=['POST'])
@cross_origin()
def ai_chat():
    query = request.json['query']
    chat_history = request.json['chat-history']

    response, source_docs = conv_chain(query, chat_history)

    return jsonify(resp=response, source_docs=source_docs)


if __name__ == "__main__":
    app.run(debug=True)
