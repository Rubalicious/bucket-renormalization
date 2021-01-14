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
    data = pd.read_csv('./{}/{}_top20_travel_numbers.csv'.format(case, case), header=None).values[1:]
    int_data = []
    for row in data:
        int_data.append([int(entry) for entry in row])
    data = np.array(int_data)

    # alternate way of estimating J
    J = -data*np.log(1-MU)
    return J

def extract_data(case, MU=0.002):
    # Read Data
    data = pd.read_csv('./{}/{}_travel_numbers.csv'.format(case, case), header=None).values
    print(data)
    # estimating J
    J = -data*np.log(1-MU)
    print(J)
    # MU = 1-np.exp(-J/np.max(data))

    minJ = np.round(np.min(J), 5)
    maxJ = np.round(np.max(J), 5)
    return J

def ith_object_name(prefix, i):
    return prefix + str(int(i))
def ijth_object_name(prefix, i,j):
    return prefix + '(' + str(int(i)) + ',' + str(int(j)) + ')'

def generate_graphical_model(case, J, init_inf, H_a, condition_on_init_inf=True):
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

    # Modify the graph by conditioning it on the initially infected nodes
    # removes the initially infected nodes too.
    if condition_on_init_inf:
        for var in init_inf:
            update_MF_of_neighbors_of(model, ith_object_name('V', var))

    return model

def get_neighbor_factors_of(model, var):
    '''
        Returns a list of 1-factors of a given variable var in the GM model
    '''
    adj_factors = model.get_adj_factors(var) # adj_factors contains a B factor and neighboring F factors
    factors = [fac for fac in adj_factors if 'F' in fac.name] # factors contains only F factors

    # collect variable names of neighbors
    var_names = []
    for fac in factors:
        for entry in fac.variables:
            var_names.append(entry.replace('V', 'B'))
    # this is a set e.g. ['B52', 'B81', 'B0']
    var_names = list(set(var_names))

    # remove the factor associated to var
    var_names.remove(var.replace('V', 'B'))

    # collection of B factors
    nbrs = model.get_factors_from(var_names)

    return nbrs

def update_MF_of_neighbors_of(model, var):
    '''
        This modifies the GM 'model' by
        1) updating the magnetic field of neighbors of variable 'var'
        2) removing the 'var' variable and associated edges
    '''
    # print('getting neighbors of {}'.format(var))
    # print('node {} has {} neighbors'.format(var, model.degree(var)-1))

    # get B-factors of the neighboring nodes (excluding index)
    nbrs = get_neighbor_factors_of(model, var)

    adj_factors = model.get_adj_factors(var) # adj_factors contains a B factor and neighboring F factors
    factors = [fac for fac in adj_factors if 'F' in fac.name] # factors contains only F factors

    # update the magnetic field of neighboring variables
    for nbr in nbrs:
        # collect the pair-wise factor containing the neighbor's bucket name
        fac = [f for f in factors if nbr.name.replace('B', '') in f.name].pop()
        # print(fac.log_values[1])
        # update the neighbor's magentic field
        nbr.log_values -= fac.log_values[1]
        # print("updated neighbor {} log value to {}".format(nbr.name, nbr.log_values))

    # remove variable var and associated factors
    model.remove_variable(var)
    model.remove_factors_from(adj_factors)


def compute_PF_of_modified_GM(model, index, ibound):
    '''
        Each computation done in parallel consists of
        1) updating the neighbors' magnetic field
        2) computing the partition function of the modified GM
    '''
    var = model.variables[index] # collect the ith variable name
    copy = model.copy()

    # this modifies the GM copy by
    # removing var and factors connected to it
    # updating the neighbors' magnetic fields

    # print('removing {} and associated neighbors'.format(var))
    update_MF_of_neighbors_of(copy, var)


    try:
        # compute partition function of the modified GM
        t1 = time.time()
        Z_copy = BucketRenormalization(copy, ibound=ibound).run()
        t2 = time.time()
        print('partition function for {} is complete: {} (time taken: {}) - ibound = {}'.format(var, Z_copy, t2 - t1, ibound))
        return [var, Z_copy, t2 - t1]
    except Exception as e:
        print(e)
        return []

def compute_marginals(case, model, params):
    init_inf, H_a, MU, ibound = params

    # ==========================================
    # Compute partition function for GM
    # ==========================================
    try:
        t1 = time.time()
        Z = BucketRenormalization(model, ibound=ibound).run()
        t2 = time.time()
        print('partition function = {}'.format(Z))
        print('time taken for GBR = {}'.format(t2-t1))
    except Exception as e:
        raise Exception(e)
    # ==========================================

    N = len(model.variables)

    # for testing
    # compute_PF_of_modified_GM(model, 0, ibound)

    results=[]
    results.append(
        Parallel(n_jobs=mp.cpu_count())(delayed(compute_PF_of_modified_GM)(model, index, ibound) for index in range(N))
    )

    # collect partition functions of sub-GMs
    Zi = []
    for index in range(N):
        Zi.append(results[0][index][1])

    # compute marginal probabilities formula conditioned on initial seed
    # ==========================================
    norm = [np.exp(model.get_factor(ith_object_name('B',i)).log_values[1]) for i in range(1,N+1)]
    print(list(enumerate(zip(norm, Zi))))

    P = lambda i: norm[i]*Zi[i]/Z
    # ==========================================

    # write data to file
    filename = "{}_ibound={}_{}_marg_prob_init_inf={}_H_a={}_MU={}.csv".format("GBR",ibound, case, init_inf, H_a, MU)

    utils.append_to_csv(filename, ['Tract', 'Z_i', 'time', 'P_i'])
    for index in range(N):
        marg_prob = P(index)
        utils.append_to_csv(filename, [results[0][index][0],results[0][index][1],results[0][index][2], marg_prob])
        print('{} complete'.format(index))
    utils.append_to_csv(filename, ['whole GM',Z,t2-t1, 'N/A'])

def compute_PF_of_modified_GM_BE(model, index):
    '''
        Each computation done in parallel consists of
        1) removing the index variable and associated edges
        2) updating the neighbors' magnetic field
        3) compute the partition function of the modified GM
    '''
    var = model.variables[index] # collect the ith variable name
    copy = model.copy()

    # this modifies the GM copy by
    # removing var and factors connected to it
    # updating the neighbors' magnetic fields

    update_MF_of_neighbors_of(copy, var)

    try:
        # compute partition function of the modified GM
        t1 = time.time()
        Z_copy = BucketElimination(copy).run()
        t2 = time.time()
        print('partition function for {} is complete: {} (time taken: {})'.format(var, Z_copy, t2 - t1))
        return [var, Z_copy, t2 - t1]
    except Exception as e:
        print(e)
        return []


def compute_marginals_BE(case, model, params):
    init_inf, H_a, MU = params

    # ==========================================
    # Compute partition function for GM
    # ==========================================
    try:
        t1 = time.time()
        Z = BucketElimination(model).run()
        t2 = time.time()
        print('partition function = {}'.format(Z))
        print('time taken for GBR = {}'.format(t2-t1))
    except Exception as e:
        raise Exception(e)
    # ==========================================

    N = len(model.variables)

    results=[]
    results.append(
        Parallel(n_jobs=mp.cpu_count())(delayed(compute_PF_of_modified_GM_BE)(model, index) for index in range(N))
    )

    # collect partition functions of sub-GMs
    Zi = []
    for index in range(N):
        Zi.append(results[0][index][1])

    # compute marginal probabilities formula conditioned on initial seed
    # ==========================================
    P = lambda i: Zi[i]/Z
    # ==========================================

    # write data to file
    filename = "{}_{}_marg_prob_init_inf={}_H_a={}_MU={}.csv".format("BE", case, init_inf, H_a, MU)

    utils.append_to_csv(filename, ['Tract', 'Z_i', 'time', 'P_i'])
    for index in range(N):
        marg_prob = P(index)
        utils.append_to_csv(filename, [results[0][index][0],results[0][index][1],results[0][index][2], marg_prob])
    utils.append_to_csv(filename, ['whole GM',Z,t2-t1, 'N/A'])


def degree_distribution(model, G, params):
    '''degree distribution'''
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
    # quit()

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
