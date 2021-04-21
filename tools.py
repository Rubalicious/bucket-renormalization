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

def extract_data_top10(MU=0.002):
    # Read Data
    data = pd.read_csv('./seattle/seattle_10sameArea_travel_numbers.csv', header=None).values
    int_data = []
    for row in data:
        int_data.append([int(entry) for entry in row])
    data = np.array(int_data)

    # alternate way of estimating J
    J = data*np.log(1/(1-MU))
    return J

def extract_data_top20(case, MU=0.002):
    # Read Data
    data = pd.read_csv('./seattle/seattle_top20_travel_numbers.csv', header=None).values
    int_data = []
    for row in data:
        int_data.append([int(entry) for entry in row])
    data = np.array(int_data)

    # alternate way of estimating J
    J = data*np.log(1/(1-MU))
    return J

def extract_data(file='seattle_20sameArea_travel_numbers.csv', MU=0.002):
    # Read Data
    data = pd.read_csv(file, header=None).values
    int_data = []
    for row in data:
        int_data.append([int(entry) for entry in row])
    data = np.array(int_data)
    # estimating J
    J = data*np.log(1/(1-MU))
    return J

def ith_object_name(prefix, i):
    return prefix + str(int(i))
def ijth_object_name(prefix, i,j):
    return prefix + '(' + str(int(i)) + ',' + str(int(j)) + ')'

def generate_graphical_model(J, H_a):
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
    # print(seed)
    healthy = [var for var in copy.variables if var not in seed]
    # modify magentic fields of neighbors of each seed
    for inf in seed:
        for sus in healthy:
            t = np.sort([int(inf[1:]), int(sus[1:])])
            fac = model.get_factor('F({},{})'.format(t[0], t[1]))
            if fac:
                copy.get_factor(sus.replace('V','B')).log_values -= fac.log_values[0]

    # remove all appropriate variables and edges
    for var in seed:
        adj_factors = copy.get_adj_factors(var)
        copy.remove_variable(var)
        copy.remove_factors_from(adj_factors)

    return copy

def compute_marginals(model, params, alg='GBR'):
    init_inf, H_a, MU, ibound = params

    N = len(model.variables)

    init_inf = [ith_object_name('V',var) for var in init_inf]
    conditioned_on_init = condition_on_seeds_from(model, init_inf)
    # print('model condition on initial seed')

    if alg == 'GBR':
        # print('running GBR ibound = {}'.format(ibound))
        t1 = time.time()
        logZ = BucketRenormalization(conditioned_on_init, ibound=ibound).run()
        t2 = time.time()
        # print('done. time taken = {}'.format(t2-t1))
    elif alg == 'BE':
        logZ = BucketElimination(conditioned_on_init).run()
    else:
        raise("Algorithm not defined")

    def subroutine_in_parallel(model, index, init_inf, ibound):
        var = model.variables[index]
        if var in init_inf:
            return +1

        conditioned_on_init_and_var = condition_on_seeds_from(model, init_inf+[var])
        if alg == "GBR":
            logZi = BucketRenormalization(conditioned_on_init_and_var, ibound=ibound).run()
        elif alg == "BE":

            logZi = BucketElimination(conditioned_on_init_and_var).run()

        # need the edge factors connected to infected nodes
        t0 = np.min([int(init_inf[0][1:]), int(var[1:])])
        t1 = np.max([int(init_inf[0][1:]), int(var[1:])])
        J_val = model.get_factor('F({},{})'.format(t0,t1)).log_values[0][0]

        marg_prob = np.exp( logZi - logZ - H_a + J_val )
        CALI = 2*marg_prob-1

        return CALI

    CALIs=Parallel(n_jobs=mp.cpu_count())(delayed(subroutine_in_parallel)(model, index, init_inf, ibound) for index in range(N))

    return CALIs
