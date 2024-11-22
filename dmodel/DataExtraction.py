import os
import subprocess
import json
import numpy as np

current_data_extration_path = os.path.dirname(os.path.abspath(__file__))

# Run the script to get the results
def run_script(filename=""):
    resfile = "res" + filename
    script = current_data_extration_path + "/script.bat"

    script_dir = os.path.dirname(script)
    subprocess.run(["cmd", "/c", script], cwd=script_dir,stdout=subprocess.DEVNULL)

def get_values(line_len = 8,numbers = 97, nv = 14,flag = 1,file=None):

    filename =  file.split("//")[-1]
    resfile = "res_" + filename
    run_script(filename)


    if flag != 1:
        return resfile
    datas = []
    with open("test_model//"+resfile, 'r') as file:
        lines = file.readlines()

    start_index = []
    labelidx = 0
    for i, line in enumerate(lines):
        if line.strip().split('\t')[0] == "Variables:" and not labelidx:
            labelidx  = i
        if line.strip() == 'Values:':
            start_index.append(i+1)

    labels = []
    for i in range(nv):
        line = lines[labelidx+i].strip().split('\t')
        if i == 0:
            labels.append(line[2])
        else:
            labels.append(line[1])
            
    for k in range(len(start_index)-1):
        data_lines = lines[start_index[k]:]
        datas.append([])
        for i in range(numbers):
            values = []
            for j in range(7):
                line = data_lines[i * line_len + j]
                line = line.strip()
                value = line.split("\t")
                if j == 0:
                    value.pop(0)
                for d in range(2):
                    values.append(float(value[d]))
            datas[k].append(values)
    return np.array(datas),labels

def get_file_paths(directory):
    file_paths = []  
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            file_paths.append(full_path.replace("\\", "/"))
    return file_paths


def loadmeasures(idx):
    directory = 'Avail_Meas_Data/Data_for_Modeling'
    files = get_file_paths(directory)
    # for i in range(len(files)):
    #     print("{} : {}".format(i,files[i]))
    file = files[idx]
    datas = []
    with open(file, 'r') as file:
        lines = file.readlines()

    start_index = []
    for i, line in enumerate(lines):
        line = line.strip().split('\t')[0]
        if line and line[0] == '#':
            start_index.append(i + 1)

    for i in range(len(start_index)):
        j = 0
        ds = lines[start_index[i]:]
        datas.append([])
        while ds[j].strip() != "END_DB":
            line = ds[j].strip().split(" ")
            values = []
            for d in range(len(line)):
                if line[d] != "":
                    values.append(float(line[d]))
            datas[i].append(values)
            j += 1

    return  np.array(datas)