import sys
import os
sys.path.extend(['graphical_model/'])
from generate_model import *
sys.path.extend(['inference/'])
from bucket_elimination import BucketElimination
from mean_field import MeanField
from belief_propagation import BeliefPropagation #, IterativeJoinGraphPropagation
from mini_bucket_elimination import MiniBucketElimination
from weighted_mini_bucket_elimination import WeightedMiniBucketElimination
from bucket_renormalization import BucketRenormalization
from uai_inference import UAIInference

# Inference protocols
MF_PROTOCOL = {
    'name': 'MF',
    'use_ibound': False,
    'algorithm': lambda model: MeanField(model),
    'run_args': {}}
BP_PROTOCOL = {
    'name': 'BP',
    'use_ibound': False,
    'algorithm': lambda model: BeliefPropagation(model),
    'run_args': {
        'max_iter': 1000,
        'converge_thr': 1e-5,
        'damp_ratio': 0.1}}
IJGP_PROTOCOL = {
    'name': 'IJGP',
    'use_ibound': True,
    'algorithm': lambda model, ibound: IterativeJoinGraphPropagation(
        model, ibound=ibound),
    'run_args': {
        'max_iter': 1000,
        'converge_thr': 1e-5,
        'damp_ratio': 0.1}}
MBE_PROTOCOL = {
    'name': 'MBE',
    'use_ibound': True,
    'algorithm': lambda model, ibound: MiniBucketElimination(
        model, ibound=ibound),
    'run_args': {}}
WMBE_PROTOCOL = {
    'name': 'WMBE',
    'use_ibound': True,
    'algorithm': lambda model, ibound: WeightedMiniBucketElimination(
        model, ibound=ibound),
    'run_args': {'max_iter': 0}}
MBR_PROTOCOL = {
    'name': 'MBR',
    'use_ibound': True,
    'algorithm': lambda model, ibound: BucketRenormalization(
        model, ibound=ibound),
    'run_args': {'max_iter': 0}}
GBR_PROTOCOL = {
    'name': 'GBR',
    'use_ibound': True,
    'algorithm': lambda model, ibound: BucketRenormalization(
        model, ibound=ibound),
    'run_args': {'max_iter': 1}}

# Model protocols
COMPLETE_PROTOCOL = {
    'generator': lambda size, delta: generate_complete(
        nb_vars=size, delta=delta),
    'true_inference': lambda model: BucketElimination(model).run()
    }
GMI_COMPLETE_PROTOCOL = {
    'generator': lambda size, delta, init_inf: generate_complete_gmi(
        nb_vars=size, delta=delta, init_inf=init_inf),
    'true_inference': lambda model: BucketElimination(model).run()
    }
GRID_PROTOCOL = {
    'generator': lambda size, delta: generate_grid(
        nb_vars=size**2, delta=delta),
    'true_inference': lambda model: BucketElimination(
        model).run(elimination_order_method='not_random')}
GMI_GRID_PROTOCOL = {
    'generator': lambda m, n, delta: generate_grid_gmi(
        m, n, delta=delta),
    'true_inference': lambda model: BucketElimination(
        model).run(elimination_order_method='not_random')}
SEATTLE_PROTOCOL = {
    'generator': lambda G, init_inf: generate_seattle(
        G=G, init_inf=init_inf),
    'true_inference': lambda model: BucketElimination(
        model).run()
    }
UAI_PROTOCOL = {
    'generator': lambda model_name: generate_uai(model_name=model_name),
    'true_inference': lambda model: UAIInference(model).run()}

inference_protocol_dict = {
    'mf': MF_PROTOCOL,
    'bp': BP_PROTOCOL,
    'ijgp': IJGP_PROTOCOL,
    'mbe': MBE_PROTOCOL,
    'wmbe': WMBE_PROTOCOL,
    'mbr': MBR_PROTOCOL,
    'gbr': GBR_PROTOCOL}

model_protocol_dict = {
    'complete': COMPLETE_PROTOCOL,
    'complete_gmi': GMI_COMPLETE_PROTOCOL,
    'grid': GRID_PROTOCOL,
    'grid_gmi': GMI_GRID_PROTOCOL,
    'seattle': SEATTLE_PROTOCOL,
    'uai': UAI_PROTOCOL}
