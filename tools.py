import numpy as np
import csv
import time
import sys
#sys.path.extend(['./', './seattle/'])
import argparse
import os
import random
import protocols
import utils
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from graphical_model import *
from factor import *
from generate_model import generate_complete_gmi, generate_complete
from bucket_elimination import BucketElimination
from bucket_renormalization import BucketRenormalization
import itertools
import traceback

from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.patches import Ellipse
import numpy.random as rnd
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def time_it(func):
    def wrapper():
        t1 = time.time()
        func()
        t2 = time.time()
        print('time taken: {}'.format(t2-t1))
    return wrapper

def extract_data_top20(case, MU=0.002):
    # Read Data
    data = pd.read_csv('./{}/{}_top20_travel_numbers.csv'.format(case, case), header=None).values
    int_data = []
    for row in data:
        int_data.append([int(entry) for entry in row])
    data = np.array(int_data)

    # alternate way of estimating J
    J = data*np.log(1/(1-MU))
    return J

def extract_data(case, MU=0.002):
    # Read Data
    data = pd.read_csv('./{}/{}_travel_numbers.csv'.format(case, case), header=None).values
    print(data)
    # estimating J
    J = data*np.log(1/(1-MU))
    print(J)
    # MU = 1-np.exp(-J/np.max(data))
    return J

def ith_object_name(prefix, i):
    return prefix + str(int(i))
def ijth_object_name(prefix, i,j):
    return prefix + '(' + str(int(i)) + ',' + str(int(j)) + ')'

def generate_graphical_model(case, J, H_a):
    '''
    This is done in 3 steps:
        1) add all variable names to the GM
        2) add all 2-factors to the GM
        3) add 1-factors to each variable
    '''
    # A collection of factors
    model = GraphicalModel()

    # first define variable names:
    model.add_variables_from( [ith_object_name('V', node) for node in range(len(J))] )

    # define interaction strength factors
    for i in range(len(J)):
        for j in range(i+1, len(J)):
            Jab = J[i,j]
            log_values = np.array([Jab, -Jab, -Jab, Jab]).reshape([2,2])
            factor = Factor(
                name = ijth_object_name('F', i, j),
                variables = [ith_object_name('V', i), ith_object_name('V', j)],
                log_values = log_values)
            model.add_factor(factor)

    # define magnetic fields
    for node in range(len(J)):
        h_a = -H_a
        log_values = np.array([-h_a, h_a])
        factor = Factor(
            name = ith_object_name('B', node),
            variables = [ith_object_name('V', node)],
            log_values = log_values)
        model.add_factor(factor)

    return model

def condition_on_seeds_from(model, seed):
    copy = model.copy()

    healthy = [var for var in copy.variables if var not in seed]
    # modify magentic fields of neighbors of each seed
    for inf in seed:
        for sus in healthy:
            t = np.sort([int(inf[1:]), int(sus[1:])])
            fac = model.get_factor('F({},{})'.format(t[0], t[1]))
            copy.get_factor(sus.replace('V','B')).log_values -= fac.log_values[0]

    # remove all appropriate variables and edges
    for var in seed:
        adj_factors = copy.get_adj_factors(var)
        copy.remove_variable(var)
        copy.remove_factors_from(adj_factors)

    return copy

# def compute_PF_of_sub_GM(model, index, init_inf, ibound):
#     '''
#         Each computation done in parallel consists of
#         1) updating the neighbors' magnetic field
#         2) computing the partition function of the modified GM
#     '''
#     var = model.variables[index] # collect the ith variable name
#
#     # this modifies the GM copy by
#     # removing var and factors connected to it
#     # updating the neighbors' magnetic fields
#     N = len(model.variables)
#
#     conditioned = condition_on_seeds_from(model, [var], in_place=False)
#     factors = conditioned.get_factors_from([ith_object_name('B',i) for i in range(N) if i not in init_inf+[var]])
#     magnetizations = [fac.log_values[0] for fac in factors]
#     print(list(zip(factors, magnetizations)))
#     quit()
#
#     try:
#         # compute partition function of the modified GM
#         t1 = time.time()
#         logZ_copy = BucketRenormalization(conditioned, ibound=ibound).run()
#         t2 = time.time()
#         print('partition function for {} is complete: {} (time taken: {}) - ibound = {}'.format(var, logZ_copy, t2 - t1, ibound))
#         return [var, logZ_copy, t2 - t1]
#     except Exception as e:
#         print(e)
#         return []

def compute_marginals(case, model, params, alg="GBR"):
    init_inf, H_a, MU, ibound = params


    N = len(model.variables)

    # condition_on_seeds_from(model, init_inf)
    init_inf = [ith_object_name('V',var) for var in init_inf]
    conditioned_on_init = condition_on_seeds_from(model, init_inf)
    if alg == "GBR":
        logZ = BucketRenormalization(conditioned_on_init, ibound=ibound).run()
        filename = "{}_ibound={}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format("GBR",ibound, case, init_inf, H_a, MU)
    elif alg == "BE":
        logZ = BucketElimination(conditioned_on_init).run()
        filename = "{}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format("BE", case, init_inf, H_a, MU)
    else:
        raise("Algorithm not defined")

    # filename = "{}_ibound={}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format("GBR",ibound, case, init_inf, H_a, MU)
    utils.append_to_csv(filename, ['Tract', 'CALI'])

    # P = lambda i: np.exp(-H_a)*np.exp(logZi[i])/np.exp(logZ)
    # healthy = list(set(model.variables).difference(set(init_inf)))



    for index in range(N):
        var = model.variables[index]
        if var in init_inf: continue # skip

        conditioned_on_init_and_var = condition_on_seeds_from(model, init_inf+[var])
        if alg == "GBR":
            logZi = BucketRenormalization(conditioned_on_init_and_var, ibound=ibound).run()
        elif alg == "BE":
            logZi = BucketElimination(conditioned_on_init_and_var).run()

        # need the edge factors connected to infected nodes
        # J_vals = sum([fac.log_values[0][0] for fac in model.factors if 'F' in fac.name and set(init_inf).intersection(set(fac.variables))])

        J_val = model.get_factor('F({},{})'.format(init_inf[0][1:],var[1:])).log_values[0][0]
        # the update rule
        # norm = np.exp(-H_a)
        # norm = 1

        marg_prob = np.exp( logZi - logZ - H_a + J_val)
        CALI = 2*marg_prob-1
        utils.append_to_csv(filename, [model.variables[index], CALI])


    # results=Parallel(n_jobs=mp.cpu_count())(delayed(compute_PF_of_sub_GM)(conditioned_on_init, index, init_inf, ibound) for index in range(N-1))


    utils.append_to_csv(filename, ['whole GM',logZ])


'''
def compute_PF_of_modified_GM_BE(model, index):
    '
        Each computation done in parallel consists of
        1) updating the neighbors' magnetic field
        2) computing the partition function of the modified GM
    '
    var = model.variables[index] # collect the ith variable name
    copy = model.copy()


    # this modifies the GM copy by
    # removing var and factors connected to it
    # updating the neighbors' magnetic fields

    # print('removing {} and associated neighbors'.format(var))
    conditioned = condition_on_seeds_from(copy, [var], in_place=False)


    try:
        # compute partition function of the modified GM
        t1 = time.time()
        logZ_copy = BucketElimination(conditioned).run()
        t2 = time.time()
        print('partition function for {} is complete: {} (time taken: {})'.format(var, logZ_copy, t2 - t1))
        return [var, logZ_copy, t2 - t1]
    except Exception as e:
        print(e)
        return []

def compute_marginals_BE(case, model, params):
    init_inf, H_a, MU = params

    N = len(model.variables)


    # ==========================================
    # Compute partition function for GM
    # ==========================================
    try:
        t1 = time.time()
        logZ = BucketElimination(model).run()
        t2 = time.time()
        print('partition function = {}'.format(logZ))
        print('time taken for GBR = {}'.format(t2-t1))
    except Exception as e:
        raise Exception(e)
    # ==========================================

    results=Parallel(n_jobs=mp.cpu_count())(delayed(compute_PF_of_modified_GM_BE)(model, index) for index in range(N))

    # collect partition functions of sub-GMs
    logZi = []
    for index in range(N):
        logZi.append(results[index][1])

    # compute marginal probabilities formula conditioned on initial seed
    # ==========================================

    factors = model.get_factors_from([ith_object_name('B',i) for i in range(1,N+1)])
    magnetizations = [fac.log_values[0] for fac in factors]
    norm = np.exp(magnetizations)
    print(N)
    print(factors)
    print(norm[19])
    print(len(logZi))

    P = lambda i: norm[i]*np.exp(logZi[i])/np.exp(logZ)
    # ==========================================

    # write data to file
    filename = "{}_{}_CALI_init_inf={}_H_a={}_MU={}.csv".format("BE", case, init_inf, H_a, MU)

    utils.append_to_csv(filename, ['Tract', 'logZi', 'time', 'CALI'])
    for index in range(N):
        print(index)
        marg_prob = P(index)
        CALI = 2*P(index)-1
        utils.append_to_csv(filename, [results[index][0],results[index][1],results[index][2], CALI])
    utils.append_to_csv(filename, ['whole GM',logZ,t2-t1, 'N/A'])'''


'''def degree_distribution(model, G, params):
    'degree distribution'
    H_a, MU = params
    degree = [model.degree(var) for var in model.variables]
    weights = [G[i][j]['weight'] for i,j in G.edges]
    # counts, bins = np.histogram(weights)
    # plt.hist(weights, bins=100)
    plt.title('min value = {}'.format(np.min(weights)))
    maxJ = np.round(np.max(weights),3)
    minJ = np.round(np.min(weights),3)
    N = len(G.nodes)
    plt.plot(range(N), degree)
    plt.title(R"$\beta$ = {}, $\mu$ = {}," "\n" "max J = {}, min J = {}".format(H_a, MU, maxJ, minJ))
    plt.savefig('./results/H_a={}_MU={}_maxJ={}_minJ={}.png'.format(H_a, MU, maxJ, minJ))
    plt.show()
    # quit()'''

'''def generate_star(N):
    model = GraphicalModel()

    center = 0

    for i in range(N):
        model.add_variable(ith_object_name('V', i))
        val = np.random.uniform(0,1)
        log_values = [-val, val]
        # print(log_values)
        bucket = Factor(
            name = ith_object_name('B', i),
            variables = [ith_object_name('V', i)],
            log_values = np.array([-val, val]))
        model.add_factor(bucket)

        if i != center:
            beta = np.random.uniform(0,1)
            log_values = np.array([beta, -beta, -beta, beta]).reshape([2,2])
            factor = Factor(
                name = ijth_object_name('F', center, i),
                variables = [ith_object_name('V', center), ith_object_name('V', i)],
                log_values = log_values)
            model.add_factor(factor)
    return model'''


'''def extract_factor_matrix(model):
    J = np.zeros([N,N])
    for (a,b) in itertools.product(range(N), range(N)):
        if a < b:
            name = 'F({},{})'.format(a,b)
            fac = model.get_factor(name)
            J[a,b] = fac.log_values[1][1] if fac is not None else 0
            J[b,a] = J[a,b]
    return J'''

'''def extract_var_weights(model, nbr_num=-1):
    N = len(model.variables)
    # store magnetic fields
    H = np.zeros([N])
    for a in range(N):
        # you can skip neighbor nbr_num
        if a == nbr_num: continue
        # get the B factor
        fac = model.get_factor('B{}'.format(a))
        # store the beta value
        H[a] = fac.log_values[0] # collects positive beta
    return H'''

'''def threshold_GM(model, TAU, in_place = True):
    '
        Tau is a percentage of the maximum J value
        which we use as a minimum threshold to keep an edge or not
    '
    copy = model if inplace else model.copy()

    interactions = [fac.log_values[1] for fac in copy.factors if 'F' in fac.name]
    maxJ = np.max(interactions)
    for fac in copy.factors:
        if 'F' in fac.name and fac.log_values[1] <= Tau*maxJ:
            # get rid of the factor
            copy.remove_factor(fac)

    if not in_place:
        return copy'''
