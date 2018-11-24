import ast
import matplotlib.pyplot as plt 
import numpy as np
data_file = "./data/tay/tay_len_xuong.txt"
file = open('./data/tay_len_xuong.txt','a+', encoding="utf8")
arr_Az = []
arr_Ax = []
arr_Ay = []
with open(data_file, encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = ast.literal_eval(line)
        yaw = line['yaw']
        pitch = line['pitch']
        roll = line['roll']
        Ax = line['Ax']
        Ay = line['Ay']
        Az = line['Az']
        
        if(len(arr_Az) > 0):
            if(Az > arr_Az[-1]):
                label = 'hand_up'
            else:
                label = 'hand_down'
        else:
            label = 'unknown'
        arr_Az.append(Az)
        arr_Ax.append(Ax)
        arr_Ay.append(Ay)
        timestamps = line['timestamps']
        file.write(str(timestamps) + ',' + str(yaw) + ',' + str(pitch) + ',' + str(roll) + ',' + str(Ax) + ',' + str(Ay) + ',' +str(Az) + ',' +label  + '\n')
arr_Ax = np.asarray(arr_Ax)
arr_Ay = np.asarray(arr_Ay)
arr_Az = np.asarray(arr_Az)
plt.plot(arr_Ax) 
plt.plot(arr_Ay) 
plt.plot(arr_Az) 
plt.legend(['Ax', 'Ay', 'Az'], loc='upper right')
plt.show()