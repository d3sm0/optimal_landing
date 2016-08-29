from multiprocessing import Process
import matplotlib.pyplot as plt
import glob
import pickle
from tqdm import tqdm
import deep_control as dc

model_description = {"data": "simple",
                     "control": dc.nn.DTHETA,
                     "nlayers": 3,
                     "units": 12,
                     "output_mode": dc.nn.OUTPUT_LOG, 
                     "dropout": False,
                     "batch_size": 8,
                     "epochs": 320,
                     "lr": 0.001,
                     "input_vars" : 5,
                     "hidden_nonlinearity": "ReLu"}

data = ["simple_qc_clean", "simple_clean"]
data = ["simple_clean"]
nls = [ (dc.nn.OUTPUT_LOG, "ReLu"),
       (dc.nn.OUTPUT_BOUNDED, "ReLu"),
       (dc.nn.OUTPUT_NO, "ReLu"),
       (dc.nn.OUTPUT_LOG, "tanh"),
       (dc.nn.OUTPUT_NO, "tanh"),
       (dc.nn.OUTPUT_BOUNDED, "tanh")
]

nlayers=[1,4]
us = [32]
#controls = [dc.nn.DTHETA, dc.nn.THRUST]
controls = [1]
th =  12
params = []
for l in nlayers:
 for d in data:
  for u in us:
    for c in controls:
        for nl in nls:
           md = model_description.copy()
           md['nlayers'] = l
           md['control'] = c
           md['output_mode'] = nl[0]
           md['hidden_nonlinearity'] = nl[1]
           md['units'] = u
           md['data'] = d
           params.append(md)
i=0
for  i in range(int(len(params)/th)):
    ps = []
    for j in range(th):
           p = Process(target=dc.nn.train, args=(params[i*th+j],))
           ps.append(p)
           p.start()

    for pi in ps:
        pi.join()

for j in range(int(len(params)%th)):
    ps = []
    for j in range(th):
           print(j)
           p = Process(target=dc.nn.train, args=(params[i*th+j],))
           ps.append(p)
           p.start()

    for pi in ps:
        pi.join()
