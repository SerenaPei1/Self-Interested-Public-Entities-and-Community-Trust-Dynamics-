''' code for network formation model'''

import networkx as nx
import numpy as np
from random import choice, choices, shuffle
from itertools import combinations


def new(N, alpha, Tau_a, Tau_b):
    ''' new graph with N many nodes '''

    G = nx.Graph()

    for node in range(N):
        tau = np.random.beta(Tau_a, Tau_b)
        type = choices(['orange','blue'], [alpha, 1-alpha])

        # add what iteration they arrived as well
        G.add_node(node+1, trust = tau, prev_trust = tau, type = type[0], arrived = 0, utility_from_food = 0) # Add new attribute

    return G

def node_enters(G, alpha, Tau_a, Tau_b, it):
    ''' add a new node and mark it as new '''

    tau = np.random.beta(Tau_a, Tau_b)
    type = choices(['orange','blue'], [alpha, 1-alpha])

    G.add_node(len(G.nodes())+1, trust = tau, prev_trust = tau, type = type[0], arrived = it, utility_from_food = 0) # Add new attribute

def christakis(G, it):
    ''' run the christakis network formation model '''

    # each node gets a pairing
    node_list = list(G.nodes())
    shuffle(node_list)

    for node in node_list:
        # get pairing
        node_pair = get_pairing(G, node, it)

        # if the edge already exists, we might delete it
        if G.has_edge(node, node_pair):
            if edge_util(G, node, node_pair) < 0 or edge_util(G, node_pair, node) < 0:
                G.remove_edge(node, node_pair)

        # if it didn't exist, we'll see if it should
        else:
            if edge_util(G, node, node_pair) > 0 and edge_util(G, node_pair, node) > 0:
                G.add_edge(node, node_pair)

def get_pairing(G, node, it):
    ''' gets random (or not so random) pairing for christakis model '''
    if G.nodes[node]['arrived'] == it:
        node_pair = choice([x for x in list(G.nodes()) if x != node])

    else:
        choice_set = [x for x in list(G.nodes()) if nx.has_path(G, x, node) and nx.shortest_path_length(G, x, node) == 2]
        if len(choice_set) == 0:
            node_pair = choice([x for x in list(G.nodes()) if x != node])
        else:
            node_pair = choice(choice_set)

    return node_pair

def edge_util(G, u, v):
    ''' returns utility to u from forming an edge with v '''

    b1 = -1
    b2 = 0
    omega = .25
    a1 = -.25
    a2 = 0
    a3 = 2.75
    a4 = 1.25
    eps = np.random.logistic()

    x_u = 1 if G.nodes[u]['type'] == 'orange' else 0
    x_v = 1 if G.nodes[v]['type'] == 'orange' else 0
    deg_v = G.degree(v)
    if nx.has_path(G, u, v):
        uv_2 = 1 if nx.shortest_path_length(G, u, v) == 2 else 0
        uv_3 = 1 if nx.shortest_path_length(G, u, v) == 3 else 0
    else:
        uv_2 = 0
        uv_3 = 0

    util = b1 + b2*x_v - (omega*(x_u-x_v)**2) + a1*deg_v + (a2*(deg_v)**2) + a3*uv_2 + a4*uv_3 + eps
    return util
