''' definitions for fairness metrics '''
import networkx as nx
import numpy as np


def util(v, G, rho, selfishness):
    # TODO: make this depend on some quality of v
    # for now, dummy metric
    satisfaction_prop = np.random.uniform(.4, .6)
    quality = entity_decision(G, rho, selfishness)

    return satisfaction_prop*quality


# the public entity disburses its resources to community members in a way that minimally erodes trust
# selfishness here represents the entity's willingness to provide poor quality service
def entity_decision(G, rho, selfishness):

    return .5
