from flask import Flask, abort, request, redirect, url_for
from flask import render_template,jsonify
import pickle as pk
from config import Config
import numpy as np
app = Flask(__name__)
with open('./results/min_max.pkl','rb') as input_file :
    min_arr = pk.load(input_file)
    max_arr = pk.load(input_file)

@app.route('/getaction', methods=['POST'])
def predict_action(): 
    input = request.json['value']
    print (type(input))
    return input
    model = read_model(Config.file_save_classifier_model)
    label_list = Config.label
    print (min_arr)
    # input = [-0.20042,-9.6045,-0.055147,-0.49165,-0.73358,0.46562]
    for i in range(len(input)):
        input[i] = (input[i] - min_arr[i])/(max_arr[i] - min_arr[i])
    input = np.asarray([input])
    print (input.shape)
    # input = input.reshape(-1,1)
    action = label_list[model.predict(input)[0]]
    print (action)
    # lol

    outputs = {
        "action": action
    }
    return jsonify(outputs=outputs)
def read_model(file_save_classifier_model):
    with open(file_save_classifier_model,'rb') as input_file :
        model = pk.load(input_file)
    return model
if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0',port=5000)