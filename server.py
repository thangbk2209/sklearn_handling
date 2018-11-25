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
    input = request.json['acc']
    print (type(input))
    # return input
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

@app.route('/getgesture', methods=['POST'])
def detect_gesture(): 
    acc = request.json['acc']
    timestamps = request.json['timestamps']
    number_samples = len(acc)
    Ax = []
    Ay = []
    Az = []
    for i in range(number_samples):
        Ax.append(acc[i][3])
        Ay.append(acc[i][4])
        Az.append(acc[i][5])
    Ax = np.array(Ax)
    Ay = np.array(Ay)
    Az = np.array(Az)
    print (Ax,Ay,Az)
    action = find_max_change(Ax,Ay,Az)
    print (action)
    if (action == 'tay_len_xuong'):
        gesture = predict_gesture(Az,'up','down')
        return gesture 
    if (action == 'tay_trai_phai'):
        gesture = predict_gesture(Ay,'right','left')
        return gesture 
    elif (action == 'tay_ra_vao'):
        gesture = predict_gesture(Ax,'out','in')
        return gesture 
    
def predict_gesture(arr,label1,label2):
    print (label1,label2)
    max_arr = np.amax(arr)
    min_arr = np.amin(arr)
    max_index = np.where(arr == max_arr)
    min_index = np.where(arr == min_arr)
    if (max_index < min_index):
        return label2
    else:
        return label1

def find_min_max_arr(arr):
    return np.amin(arr), np.amax(arr)
def find_max_change(Ax,Ay,Az):
    min_ax,max_ax = find_min_max_arr(Ax)
    min_ay,max_ay = find_min_max_arr(Ay)
    min_az,max_az = find_min_max_arr(Az)
    x_change = max_ax - min_ax
    y_change = max_ay - min_ay
    z_change = max_az - min_az
    arr = [x_change,y_change,z_change]
    max_change = np.amax(arr)
    if (max_change == z_change):
        return 'tay_len_xuong'
    elif(max_change == y_change):
        return 'tay_trai_phai'
    else:
        return 'tay_ra_vao'
def read_model(file_save_classifier_model):
    with open(file_save_classifier_model,'rb') as input_file :
        model = pk.load(input_file)
    return model
if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0',port=5000)