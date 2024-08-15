from json import JSONEncoder
import json
import plotting
import numpy as np
import pandas as pd

N = 100
ntwk_iters = 15

rho_list = [0, 5, 10, 15, 20]
Tau_list = [(2,20),(2,10),(2,5),(2,2),(5,2),(10,2),(20,2)]
new_rho_list = [x/(ntwk_iters) for x in rho_list]
new_Tau_list = [np.round((a)/(a+b),2) for (a,b) in Tau_list]

filename = 'sim_output/test.json'

with open(filename, "r") as read_file:
    decodedArray = json.load(read_file)
    arr1 = np.asarray(decodedArray["util"])

plotting.heat_map(arr1, new_rho_list, new_Tau_list, type = 'triangles', title = 'test', save = True)
