import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Extract data from .txt files
def Extracdatas(idxstr,resfile):
    with open(resfile, 'r') as file:
        lines = file.readlines()

    start_index = []
    for i, line in enumerate(lines):
        if idxstr in line:
            start_index.append(i)
    labels = []
    nlabel = lines[start_index[0]+2].split(":")[1]
    nlabel = int(nlabel)
    npoint = lines[start_index[0]+3].split(":")[1]
    labels.append(lines[start_index[0]+4].split("\t")[2])
    for i in range(int(nlabel)-1):
        labels.append(lines[start_index[0] + 5+i].split("\t")[3])

    datas = []
    for k in range(len(start_index)):
        data_lines = lines[start_index[k]+5+nlabel:]
        datas.append([])
        for i in range(int(npoint)):
            values = []
            for j in range(nlabel//2):
                line = data_lines[i * (nlabel//2+1) + j]
                line = line.strip()
                value = line.split("\t")
                if j == 0:
                    value.pop(0)
                for d in range(2):
                    values.append(float(value[d]))
            datas[k].append(values),labels
    return np.array(datas)

def Getdatas(file,idxstr):
    datas = Extracdatas(idxstr,file)
    return datas

######paras
DC_paras = ["voff", "nfactor", "u0", "ua", "Igsdio", "Njgs", "Igddio", "Njgd", "Rshg", "Eta0", "Vdscale", "Cdscd",
            "Rsc", "Rdc", "UTE", "RTH0", "LAMBDA", "Vsat", "Tbar"]

pbounds = {
    "voff": (-10, 0),
    "nfactor": (0, 5),
    "u0": (1e-3, 0.5),
    "ua": (1e-9, 5e-7),
    "Igsdio": (0.02, 0.5),
    "Njgs": (0.5, 13.5),
    "Igddio": (2.5, 63),
    "Njgd": (0.6, 15.6),
    "Rshg": (2e-4, 5e-3),
    "Eta0": (0.02, 0.52),
    "Vdscale": (1.1, 28.3),
    "Cdscd": (-5, 5),
    "Rsc": (9.2e-5, 2.3e-3),
    "Rdc": (2.4e-4, 6.1e-3),
    "UTE": (-3, -0.2),
    "RTH0": (4.8, 120),
    "LAMBDA": (4e-4, 0.011),
    "Vsat": (5e4, 1.3e6),
    "Tbar": (2e-9, 5.7e-8)
}

dc_netfiles = ["5DC_input.txt","5DC_transfer_lin.txt","5DC_trans_su.txt","5DC_trans.txt","5DC_output.txt"]


def generate_paras(pbounds):
    paras = {}
    for key, (low, high) in pbounds.items():
        value = np.random.uniform(low, high)
        value = float('%.4g' % value)
        paras[key] = value
    return paras

def Changeparas(paras,path="asmhemt.txt"):
    for i in range(1):
        with open(path, 'r+') as file:
            content = file.read()
        # 替换参数值
        for parameter, new_value in paras.items():
            parameter = parameter.lower()
            new_value = str(new_value).upper()
            content = re.sub(f'{parameter} =[\d\w\+\-E\.]*', f'{parameter} ={new_value}', content, flags=re.IGNORECASE)

        with open(path, 'w') as file:
            file.write(content)


######loss
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true+0.01)))

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))


def inputloss(mds,ds):

    loss1 = RMSE(mds[0,-10:,1],-ds[0,-10:,13])
    loss2 = RMSE(mds[0, -10:, 2], -ds[0,-10 :, 12])*4
    return  np.average([loss1,loss2])

def translinloss(mds,ds):
     loss1 = RMSE(mds[0,:,1],-ds[0,:,13])
     loss2 = RMSE(mds[1, :, 1], -ds[1, :, 13])
     loss3 = RMSE(mds[2, :, 1], -ds[2, :, 13])
     return np.average([loss1, loss2,loss3])

def transsubloss(mds, ds):
    losses = []
    for j in range(21):
        losses.append( RMSE(mds[j, :, 1], -ds[j, :, 13]) )
    return np.average(losses)


def transloss(mds,ds):
    losses = []
    for j in range(6):
        losses.append( RMSE(mds[j, :, 1], -ds[j, :, 13]) )
    for k in range(len(ds[:])):
        losses.append(RMSE(mds[:, k, 1],-ds[:,k,13]))
    return np.average(losses)

def outloss(mds,ds):
    losses = []
    for j in range(24):
        losses.append( RMSE(mds[j, :, 1], -ds[j, :, 13]) )

    return np.average(losses)

