import DataExtraction
import numpy as np
import os
from DC_modeling import Getdatas,Changeparas
import matplotlib.pyplot as plt
import tqdm

def generate_monte_carlo_samples(pbounds, num_samples):
    samples = []
    for _ in range(num_samples):
        paras = {}
        for key, (low, high) in pbounds.items():
            value = np.random.uniform(low, high)
            value = float('%.4g' % value)
            paras[key] = value
        samples.append(paras)
    return samples

# define your parameter space
pbounds = {

}

num_samples=80000
params = generate_monte_carlo_samples(pbounds, num_samples=num_samples)

# save .npy
np.save("params.npy", params)
params = np.load("params.npy",allow_pickle=True)

# initialize the dataset
filename = 'data_val.npy'
if os.path.exists(filename):
    ds = np.load(filename)
else:
    ds = np.empty((0, 76, 41, 2))

for i in tqdm.tqdm(range(len(params))):
    Changeparas(params[i])
    DataExtraction.run_script()
    datas = Getdatas("res.txt", "Plotname: DC")

    datas = np.array(datas)

    Id = -datas[:, :, 13]
    Gm = Id
    Gm = np.diff(Id, axis=1) / 0.1 # 0.1 is the step size
    zero_column = np.zeros((76, 1))
    Gm= np.hstack([Gm, zero_column])

    # Id and Gm (76,41,2)
    datas = np.dstack((Id, Gm))

    ds = np.append(ds, [datas], axis=0)

    if (i + 1) % 100 == 0 or (i + 1) == num_samples:
        np.save(filename, ds)
    
    np.save(filename, ds)