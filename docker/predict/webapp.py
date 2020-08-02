from flask import Flask, request, jsonify
import torch
from model import MLP

app = Flask(__name__)


def predict(net, data):

    data = torch.FloatTensor(data)
    net.eval()
    output = net(data)
    output = torch.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)

    return output.detach().numpy().tolist()


@app.route('/', methods=["GET"])
def get_prediction():
    content = request.json
    data = content['data']
    output = predict(net=model, data=data)
    return jsonify({"output": output})


if __name__ == '__main__':

    model = torch.load("./MLP.pt")

    app.run(debug=False,  host='0.0.0.0', port=80)
