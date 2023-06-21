#%%
import GPy
import numpy as np
import matplotlib.pyplot as plt
from functions import use_NARX_model_in_simulation, import_data, make_training_data, compute_rms
import pandas as pd
import os

######################################################################
save_name = 'sparse_GP_varying_inducing_points'
######################################################################
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', save_name)
if os.path.exists(save_dir):
    raise ValueError('Save name already exists, please change save_name')

# Defining global variables
na, nb = 5, 5

# Input data & functions
# for all
u_train, th_train, u_pred, th_pred, u_sim, th_sim = import_data()
X_train_all, Y_train_all, X_val, Y_val =  make_training_data(u_train, th_train, na, nb, train_amount=0.8)
# for prediction on submission file
X_sub_file = np.concatenate([u_pred[:,-nb-1:-1], th_pred[:,-na-1:-1]], axis=1)
Y_sub_file = th_pred[:,-1]
# for simulation on validation set:
u_val = list(u_train[-len(Y_val):])
th_val = list(th_train[-len(Y_val):])
fmodel = lambda u, y: model.predict(np.concatenate([u,y])[None,:])

val_prediction = pd.DataFrame(columns=['N_inducing_points', 'N_training_points', 'na', 'nb', 'noise_variance', 'kern_variance', 'lengthscale', 'rms_rad', 'rms_deg', 'nrms'])
submission_prediction = pd.DataFrame(columns=['N_inducing_points', 'N_training_points', 'na', 'nb', 'noise_variance', 'kern_variance', 'lengthscale', 'rms_rad', 'rms_deg', 'nrms'])
val_simulation = pd.DataFrame(columns=['N_inducing_points', 'N_training_points', 'na', 'nb', 'noise_variance', 'kern_variance', 'lengthscale', 'rms_rad', 'rms_deg', 'nrms'])

# Define dictionary for saving results
split_dict = {'val_pred': {'Y_pred': None, 'Y_gt': Y_val, 'results': val_prediction},\
'sub_pred': {'Y_pred': None, 'Y_gt': Y_sub_file, 'results': submission_prediction},\
'val_sim': {'Y_pred': None, 'Y_gt': Y_val, 'results': val_simulation}}

# list where key is number of training points and list contains number of inducing points for that number of training points
N_inducing_points_list = {50: [1, 5, 10, 25],
                          100: [1, 10, 25, 50],
                          1000: [10, 50, 100, 500],
                          10000: [10, 100, 500, 1000, 5000],
                          50000: [10, 100, 500, 1000, 5000, 10000],
                          len(Y_train_all): [10, 100, 500, 1000, 5000, 10000, 50000]}

for N_training in N_inducing_points_list.keys():
    X_train = X_train_all[0:N_training]
    Y_train = Y_train_all[0:N_training]
    for N_inducing_points in N_inducing_points_list[N_training]:
        model = GPy.models.SparseGPRegression(X_train, Y_train, num_inducing=N_inducing_points)
        model.randomize()
        model.Z.unconstrain()
        model.optimize('bfgs')
        
        # PREDICTION on VALIDATION data
        split_dict['val_pred']['Y_pred'] = model.predict(X_val)[0][:,0]
        # PREDICTION on SUBMISSION file
        split_dict['sub_pred']['Y_pred'] = model.predict(X_sub_file)[0][:,0]
        # SIMULATION on VALIDATION data
        split_dict['val_sim']['Y_pred'] = use_NARX_model_in_simulation(u_val, th_val, fmodel, na, nb)[0]
        
        
        # Compute and save results
        for set in split_dict.keys():
            rms_rad, rms_deg, nrms = compute_rms(split_dict[set]['Y_pred'], split_dict[set]['Y_gt'])
        
            split_dict[set]['results'] = split_dict[set]['results'].append({'N_inducing_points': N_inducing_points, 'N_training_points': N_training,\
                'na': na, 'nb': nb, 'noise_variance': model.Gaussian_noise.variance.item(), 'kern_variance': model.kern.variance.item(),\
                'lengthscale': model.kern.lengthscale.item(), 'rms_rad': rms_rad, 'rms_deg': rms_deg, 'nrms': nrms}, ignore_index=True)

            # Save results
            if not os.path.exists(save_dir): os.mkdir(save_dir)
            save_path = os.path.join(save_dir, set+'.csv')
            split_dict[set]['results'].to_csv(save_path, index=False)
            
        
        