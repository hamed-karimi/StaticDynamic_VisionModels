import numpy as np
from os.path import exists as pexists
from os.path import join as pjoin
import os
import re
import psutil
import sys
import gc
import json
import time
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from os.path import isfile
import pandas as pd
from copy import deepcopy
from scipy.stats import zscore
import shutil
import json
from types import SimpleNamespace
from sklearn.linear_model import LinearRegression
import matplotlib.colors as mcolors

def get_masked_rdm(rdm, separate_segments=False):
    fmri_triggers = np.array([451, 441, 438, 488, 462, 439, 542, 338])
    fmri_timepoints_cum = np.cumsum(fmri_triggers)
    segments_vec_rdm = []
    start = 0
    for s in range(fmri_timepoints_cum.shape[0]):
        tri = upper_tri_masking(rdm[start:fmri_timepoints_cum[s], start:fmri_timepoints_cum[s]])
        segments_vec_rdm.append(tri)
        start = fmri_timepoints_cum[s]
    if separate_segments:
        return segments_vec_rdm
    else:
        masked_rdm = np.concatenate(segments_vec_rdm, axis=0)
        return masked_rdm

def get_params(model_dir):
    params_dir = os.path.join('{0}/Parameters.json'.format(model_dir))
    with open(params_dir, 'r') as json_file:
        params = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    return params

def pearson_corr(m1, m2):
    m1_z = zscore(m1, axis=0)
    m2_z = zscore(m2, axis=0)
    
    corr = np.matmul(np.transpose(m1_z), m2_z) / m1_z.shape[0]
    return corr

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return np.expand_dims(A[mask], axis=1)

from scipy import stats, linalg

def semi_partial_corr(x, y, z):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.


    Parameters
    ----------
    x : array-like, shape (n, p) -> (6474601, 1000)
        Array with the different variables. Each column of C is taken as a variable
    y : array-like, shape (p, ) -> (6474601, )
    
    z : array-like, shape(p, ) -> (6474601, n), n is number of layers
        control z in y
    Returns
    -------
    P : array-like, shape (n, )
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    z_intercept = np.hstack([np.ones((z.shape[0], 1)), z])
    beta = np.linalg.lstsq(z_intercept, y, rcond=None)[0]
    residuals = np.asarray(y - z_intercept.dot(beta), dtype=np.float64) # (6474601, 1000)
    p_corr = pearson_corr(residuals, x) # (1000, )

    return p_corr

def next_layer_rdm(model, mode, layer_names, separate_segments=False):
    model_dir = os.path.join('/data/karimike/Documents/ActionRecognition-2024/Models/', model, mode)
    for layer in layer_names[model][mode]:
        layer_rdm_list = []
        for seg in range(8):
            seg_rdm_dir = os.path.join(model_dir,
                                       'seg_{0}'.format(seg),
                                       'rdms')
            
            rdm_path = pjoin(seg_rdm_dir, '{0}.npy'.format(layer))
            seg_layer_rdm = np.load(rdm_path)
            tri = upper_tri_masking(seg_layer_rdm)
            layer_rdm_list.append(tri)
            # disorg_layer_rdm = disorganize_rdm(layer_rdm)
        if separate_segments:
            yield layer, layer_rdm_list
        else:
            layer_rdm = np.concatenate(layer_rdm_list, axis=0)
            yield layer, layer_rdm
        
def get_subject_rdm(subject, roi_name, separate_segments=False):
    subject_dir = '/data/karimike/Documents/forrest_study_fmri/Analysis/All Runs/All models/subject-{0:02d}/Selected Region-maxprob-RDMs/'.format(subject)
    roi_path = pjoin(subject_dir, '{0}.npy'.format(roi_name))
    roi_rdm = np.load(roi_path)
    masked = get_masked_rdm(roi_rdm, separate_segments)
    return masked

def predict_roi_rdm(train_x, train_y, test_x): # train_x: list(#seg, array(#layer, #time)), train_y: list(#seg, array(#layer, 1))
    rdm_num = train_x[0].shape[0]
    betas = np.zeros((len(train_x), rdm_num))
    intercepts = np.zeros((len(train_x), ))
    for seg in range(len(train_x)):
        seg_train_x = train_x[seg].squeeze().T
        seg_train_y = train_y[seg].squeeze()
        # print('X:', seg_train_x.shape, 'y:', seg_train_y.shape)
        lr = LinearRegression(positive=True)
        reg = lr.fit(seg_train_x, seg_train_y)
        seg_betas = reg.coef_ # (rdm_num, )
        seg_intercepts = reg.intercept_ # (1, )
        
        betas[seg, :] = seg_betas
        intercepts[seg]= seg_intercepts
        
    
    lr = LinearRegression()
    lr.coef_ = betas.mean(axis=0)
    lr.intercept_ = intercepts.mean()
    pred_test_y = lr.predict(test_x.squeeze().T)
    
    return pred_test_y, betas.mean(axis=0)


def get_models_layers_rdm(model_names: list, 
                          modes: list,
                          layer_names_dict):
    models_rdms_list = []
    for s in range(8):
        models_rdms_list.append([])
    for model, mode in zip(model_names, modes):
        for layer, layer_rdm in next_layer_rdm(model=model, 
                                               mode=mode, 
                                               layer_names=layer_names_dict, 
                                               separate_segments=True):
            for s in range(8):
                models_rdms_list[s].append(layer_rdm[s])
    
    models_rdms = [np.array(models_rdms_list[s]) for s in range(8)]
    return models_rdms
    # return np.array(models_rdms)
                
def predicted_segment_rdm(subject, 
                          roi_name, 
                          model_names: list, 
                          model_modes: list,  
                          layer_names_dict: dict, 
                          test_segment: int):
    
    subject_rdm = get_subject_rdm(subject, roi_name, separate_segments=True)
    model_layer_rdms = get_models_layers_rdm(model_names, 
                                                     model_modes, 
                                                     layer_names_dict)
    train_layer_rdms = []
    train_subject_rdms = []
    train_idx = [s for s in range(8) if s != test_segment] #[0, 1, 2, 3, 4, 5, 6] # segment indices
    print(train_idx)
    test_idx= [test_segment]
    print(test_idx)
    for idx in train_idx:
        train_layer_rdms.append(model_layer_rdms[idx])
        train_subject_rdms.append(subject_rdm[idx])
    for idx in test_idx: # only one index
        test_layer_rdms = model_layer_rdms[idx]
        test_subject_rdms = subject_rdm[idx]
        
    pred_test_subject_rdm, betas = predict_roi_rdm(train_layer_rdms, 
                                                   train_subject_rdms, 
                                                   test_layer_rdms)
    corr = pearson_corr(pred_test_subject_rdm, test_subject_rdms)
    return pred_test_subject_rdm, corr, betas

subject = int(sys.argv[1]) # [1, 2, 3, 4, 5, 9, 10, 14, 15, 16, 17, 18, 19, 20]
analysis = 'LinearRegression'
dnn_models = ['Spatial', 'TinyMotionNet', 'FlowClassifier', 'ImageMAE', 'VideoMAE', 'MVD']

modes = {'Spatial':('ImageNet-trained', 'HAA-trained'),
         'TinyMotionNet': tuple(['HAA-trained']),
         'FlowClassifier': tuple(['HAA-trained']),
         'ImageMAE': tuple(['pre-trained', 'fine-tuned']),
         'VideoMAE': tuple(['pre-trained', 'fine-tuned']), #'pre-trained'
         'MVD': tuple(['pre-trained'])} 
# Change this as desired
model_comb = {'static': 
              tuple([
                  ('Spatial', 'ImageNet-trained'), 
                  ('Spatial', 'HAA-trained'),
                  ('ImageMAE', 'pre-trained'),
                  ('ImageMAE', 'fine-tuned')
              ]), 
             'dynamic':
              tuple([
                  ('TinyMotionNet', 'HAA-trained'),
                  ('FlowClassifier', 'HAA-trained'),
                  ('VideoMAE', 'pre-trained'),
                  ('VideoMAE', 'fine-tuned'),
                  ('MVD', 'pre-trained')
             ])}
n_models = sum([len(modes[m]) for m in modes])
roi_names = ['ventral', 'dorsal', 'lateral', 'parietal_frontal']
layer_names = {}
base_dir = '/data/karimike/Documents/ActionRecognition-2024/Models'
pcorr = dict()
subject_res_dir = os.path.join(base_dir, analysis, 'subject-{0:02d}'.format(subject))
if not os.path.exists(subject_res_dir):
    os.makedirs(subject_res_dir)
    
for model in dnn_models:
    layer_names[model] = {}
    for at_mode in modes[model]:
        params = get_params(os.path.join(base_dir, model, at_mode))
        layer_names[model][at_mode] = params.MODULES
        
combined_corr = {}
combined_betas = {}
model_names = []
model_modes = []
for model in dnn_models:
    for mode in modes[model]:
        model_names.append(model)
        model_modes.append(mode)

print(model_names)
print(model_modes)
for roi_id in range(len(roi_names)):
    print(roi_names[roi_id])

    corr = np.zeros((8))
    for s in range(8):
        _, corr[s], betas = predicted_segment_rdm(subject=subject, 
                                                  roi_name=roi_names[roi_id], 
                                                  model_names=model_names, 
                                                  model_modes=model_modes, 
                                                  layer_names_dict=layer_names,
                                                  test_segment=s)

            
    combined_corr[roi_names[roi_id]] = corr.mean()
    combined_betas[roi_names[roi_id]] = betas

f = open(pjoin(subject_res_dir, '4_streams_linreg_all_models_combined.pkl'),'wb')
pickle.dump(combined_corr, f)
f.close()

f = open(pjoin(subject_res_dir, '4_streams_linreg_betas_all_models_combined.pkl'),'wb')
pickle.dump(combined_betas, f)
f.close()