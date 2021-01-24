import numpy as np
import networkx as nx
import csv
import time
import sys
import argparse
import os
import random
import protocols
import utils
import copy

sys.path.extend(["graphical_model/"])

from factor import Factor
from graphical_model import GraphicalModel

def ith_object_name(prefix, i):
    return prefix + str(int(i))

def ijth_object_name(prefix, i, j):
    return prefix + "(" + str(int(i)) + "," + str(int(j)) + ")"

def pandemic_model(J, h, infected):

    healthy_nodes = list(set(range(len(J))).difference(set(infected)))
    new_J = np.zeros((len(healthy_nodes), len(healthy_nodes)))
    for i in range(len(healthy_nodes)):
        for j in range(i + 1, len(healthy_nodes)):
            new_J[i, j] = J[healthy_nodes[i], healthy_nodes[j]]
            new_J[j, i] = J[healthy_nodes[j], healthy_nodes[i]]

    new_h = np.zeros(len(healthy_nodes))
    for i in range(len(healthy_nodes)):
        new_h[i] = h[healthy_nodes[i]]
        for j in range(len(infected)):
            new_h[i] += J[healthy_nodes[i], infected[j]]

    graph = nx.from_numpy_matrix(new_J)
    model = GraphicalModel()
    model_size = len(graph.nodes())

    for i in range(model_size):
        model.add_variable(ith_object_name("V", i))

    for i in range(model_size):
        for j in range(i + 1, model_size):
            beta = new_J[i, j]
            log_values = np.array([beta, -beta, -beta, beta]).reshape([2, 2])
            factor = Factor(
                name=ijth_object_name("F", i, j),
                variables=[ith_object_name("V", i), ith_object_name("V", j)],
                log_values=log_values,
            )
            model.add_factor(factor)

    for i in range(model_size):
        factor = Factor(
            name=ith_object_name("B", i),
            variables=[ith_object_name("V", i)],
            log_values=np.array([new_h[i], -new_h[i]]),
        )
        model.add_factor(factor)

    return model

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-type", default="grid", help="type of graphical model")
parser.add_argument(
    "-alg", "--algorithms", nargs="+", default=["gbr"], help="algorithms to be tested"
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

model_protocol = protocols.model_protocol_dict[args.model_type]
inference_protocols = [protocols.inference_protocol_dict[name] for name in args.algorithms]

ibound = 20
J = np.array([[0,34,147,82,62,153,123,81,37,156,44,29,41,258,146,0,62,175,28,66],
[34,0,410,43,28,134,340,68,11,27,72,18,39,464,129,56,142,134,54,49],
[147,410,0,130,40,520,622,64,36,19,69,56,68,277,39,40,84,58,17,52],
[82,43,130,0,80,55,81,46,95,142,90,54,43,359,133,15,55,136,0,59],
[62,28,40,80,0,38,165,63,112,158,45,23,96,360,225,50,91,153,23,28],
[153,134,520,55,38,0,644,18,59,35,45,47,54,420,192,47,63,80,36,19],
[123,340,622,81,165,644,0,68,50,41,0,145,21,20,12,89,28,36,60,1],
[81,68,64,46,63,18,68,0,102,180,117,174,109,465,244,50,96,147,64,65],
[37,11,36,95,112,59,50,102,0,98,51,89,74,179,155,50,22,57,43,17],
[156,27,19,142,158,35,41,180,98,0,123,196,209,273,186,61,50,136,212,70],
[44,72,69,90,45,45,0,117,51,123,0,266,72,212,95,111,39,59,99,4],
[29,18,56,54,23,47,145,174,89,196,266,0,112,719,413,253,123,216,12,52],
[41,39,68,43,96,54,21,109,74,209,72,112,0,369,168,44,96,102,87,34],
[258,464,277,359,360,420,20,465,179,273,212,719,369,0,350,219,110,164,409,42],
[146,129,39,133,225,192,12,244,155,186,95,413,168,350,0,191,135,123,183,6],
[0,56,40,15,50,47,89,50,50,61,111,253,44,219,191,0,72,102,141,24],
[62,142,84,55,91,63,28,96,22,50,39,123,96,110,135,72,0,103,135,0],
[175,134,58,136,153,80,36,147,57,136,59,216,102,164,123,102,103,0,301,180],
[28,54,17,0,23,36,60,64,43,212,99,12,87,409,183,141,135,301,0,182],
[66,49,52,59,28,19,1,65,17,70,4,52,34,42,6,24,0,180,182,0]])

h = 0.1 * np.ones(len(J))
mu = 0.0005
J = J*np.log(1/(1-mu))
infected = [0]

model = pandemic_model(J, h, infected)
true_logZ = model_protocol["true_inference"](model)

# compute PF
for ip in inference_protocols:
    if ip["use_ibound"]:
        alg = ip["algorithm"](model, ibound)
    else:
        alg = ip["algorithm"](model)

    tic = time.time()
    logZ = alg.run(**ip["run_args"])
    print("logZ is equal to: ", logZ)
    err = np.abs(true_logZ - logZ)
    toc = time.time()

    print("Alg: {:15}, Error: {:15.4f}, Time: {:15.2f}".format(ip["name"], err, toc - tic))

# compute marginals
yet_healthy = list(set(range(len(J))).difference(set(infected)))
# write data to file
case = 'seattle'
init_inf = [0]
H_a = 0.1
MU = 0.0005
filename = "{}_{}_marg_prob_init_inf={}_H_a={}_MU={}.csv".format("PM", case, init_inf, H_a, MU)

utils.append_to_csv(filename, ['Tract', 'logZi/logZ'])

for node in yet_healthy:
    this_infected = copy.copy(infected)
    this_infected.append(node)
    model = pandemic_model(J, h, this_infected)

    for ip in inference_protocols:
        if ip["use_ibound"]:
            alg = ip["algorithm"](model, ibound)
        else:
            alg = ip["algorithm"](model)

        tic = time.time()
        logZ_node = alg.run(**ip["run_args"])
        print("Node ", node, " is infected with probability: ", logZ_node/logZ)
        toc = time.time()
        utils.append_to_csv(filename,[node,logZ_node/logZ])