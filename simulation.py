''' code to run simulation '''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import formation, recommendations, metrics, plotting
from copy import deepcopy
from json import JSONEncoder
import json

N = 100                          # number of nodes init
ntwk_iters = 15                  # network iters, how many nodes to add
total_nodes = N+ntwk_iters
extra_iters = 50                # extra iterations after all nodes have been added
sim_iters = 1                   # total number of times to run each iter
alpha = .5                      # node types

rho_list = [0, 10]
#rho_list = [30, 40]
#rho_list = [0]
#rho_list = [20]
Tau_list = [(2,20),(20,2)]
#Tau_list = [(20,2)]
#Tau_list = [(5,2),(10,2),(20,2)]

results_arr = np.empty((len(rho_list), len(Tau_list), sim_iters))

G_old = None

# rho is public entity resource constraint
for idx_r, rho in enumerate(rho_list):
    # tau_a and tau_b give the dist for trust
    for idx_t, (Tau_a, Tau_b) in enumerate(Tau_list):
        for sim_it in range(sim_iters):
            print(rho, Tau_a, Tau_b, sim_it)
            # run many iterations of the network formation model
            for i in range(ntwk_iters+extra_iters):
                # if it's the first iteration then create the graph
                if i == 0:
                    G = formation.new(N, alpha, Tau_a, Tau_b)
                    formation.christakis(G,i)
                elif i <= ntwk_iters:
                    formation.node_enters(G, alpha, Tau_a, Tau_b, i)
                    formation.christakis(G, i)

                # store the current G before anything happens
                G_old = deepcopy(G)

                # public entity provides provides service
                recommendations.provide_service(G, rho, selfishness = .5)

            util_list = [node[1]['utility_from_food'] for node in G.nodes.data()]
            results_arr[idx_r][idx_t][sim_it] = np.mean(util_list)
print(results_arr)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Serialization
filename = 'sim_output/test.json'
numpyData = {"util": results_arr}
with open(filename, "w") as write_file:
    json.dump(numpyData, write_file, cls=NumpyArrayEncoder)
print("Done writing serialized NumPy array into file")
