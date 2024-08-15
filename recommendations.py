''' code for public entity to make recommendations'''

import random
import metrics
import networkx as nx
import numpy as np


# New function to provide service
# should happen according to entity_metric somehow

def provide_service(G, rho, selfishness):
    ''' public entity provides a service that impacts the utility_from_food of nodes '''
    
    selected_nodes = select_nodes(G, rho)  # Select nodes
    change_vector = []

    for node in list(G.nodes()):
        if node in selected_nodes:
            # Change node attribute based on quality of service
            change = metrics.util(node, G, rho, selfishness=.5)
            G.nodes[node]['utility_from_food'] = change
            change_vector.append(change)
        else:
            # if node wasn't given service, change is 0
            G.nodes[node]['utility_from_food'] = 0
            change_vector.append(0)

    trust_update(G, change_vector)

# New function to select nodes
# this should also happen according to entity_metric
def select_nodes(G, num_nodes):
    ''' select random nodes to provide the service to '''
    return random.sample(list(G.nodes), num_nodes)

# trust changes as a social phenomenon
def trust_update(G, change_vector, lambda_value=0.5):
    for v_idx, v in enumerate(list(G.nodes())):
        current_trust = G.nodes[v]['trust']
        service_quality = change_vector[v_idx]
        neighbors = list(G.neighbors(v))

        if len(neighbors) != 0:
            avg_neighbors_service_quality = np.mean([G.nodes[neighbor]['utility_from_food'] for neighbor in neighbors])
        else:
            avg_neighbors_service_quality = 0

        # agent's trust is the avg of their own service quality and their neighbors'
        trust_update = (service_quality + avg_neighbors_service_quality) / 2

        prev_trust = G.nodes[v]['prev_trust']
        delta_trust = lambda_value * (current_trust - prev_trust) + (1 - lambda_value) * trust_update

        new_trust = max(0, min(1, current_trust + delta_trust))
        G.nodes[v]['prev_trust'] = current_trust
        G.nodes[v]['trust'] = new_trust
