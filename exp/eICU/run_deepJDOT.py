import getpass
user_id = getpass.getuser()

import sys
sys.path.append(f"/home/{user_id}/projects/OTTEHR/")
sys.path.append(f"/home/{user_id}/projects/unbalanced_gromov_wasserstein/")
sys.path.append(f"/home/{user_id}/projects/OTTEHR/competitors/deepJDOT")


from ast import literal_eval
import dnn
from mimic_common import *
import numpy as np
import os
import ot
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import time

import dnn
from Deepjdot import Deepjdot


n_components = 50
#%% feature extraction and regressor function definition
def feat_ext(main_input, l2_weight=0.0):
    net = dnn.Dense(2*n_components, activation='relu', name='fe')(main_input)
    net = dnn.Dense(n_components, activation='relu', name='feat_ext')(net)
    return net
    

def regressor(model_input, l2_weight=0.0):
    net = dnn.Dense(int(n_components/2), activation='relu', name='rg')(model_input)
    net = dnn.Dense(1, activation='linear', name='rg_output')(net)
    return net



data = 'eICU'
data_dir = os.path.join(os.path.expanduser("~"), f"projects/OTTEHR/data/{data}")
output_dir = os.path.join(os.path.expanduser("~"), f"projects/OTTEHR/outputs/{data}")
print(f"Will save outputs to {output_dir}")

""" 
Read in the original dataframe
"""
df = pd.read_csv(os.path.join(data_dir, "admission_patient_diagnosis_ICD.csv"), index_col=None, header=0, converters={'ICD codes': literal_eval})

suffix = None

iterations = 100
trans_metric = 'deepJDOT'

n_components = 50

suffix = None

# Update group_name and groups to appropriate values 
group_name = 'hospitalid'
groups = [420, 264, 243, 338, 73, 458, 167, 443, 208, 300]

group_1_count = 120
group_2_count = 100
type = 'cat'


for group_1 in groups:
    for group_2 in groups:
        if group_1 == group_2:
            continue
        score_path = os.path.join(output_dir, f"{group_name}_{group_2}_to_{group_1}_{trans_metric}.csv")
        if os.path.exists(score_path):
            print(f"{score_path} exists!")
            continue
        maes = []
        rmses = []
        for i in range(iterations):
            start_time = time.time()
            selected_df = select_samples(df, group_name, type, group_1, group_2, group_1_count, group_2_count)
            code_feature_name = 'ICD codes'
            label_name = 'duration'
            source_data, source_labels, target_data, target_labels = gen_code_feature_label(selected_df, group_name, type, group_1, group_2, code_feature_name, label_name)
            n_dim = np.shape(source_data)
            optim = tf.keras.optimizers.SGD(learning_rate=0.001)
            # optim= tf.keras.optimizers.legacy.SGD(learning_rate=0.1)
                
            # #%% Feature extraction as a keras model
            main_input = dnn.Input(shape=(n_dim[1],))
            fe = feat_ext(main_input)
            # # feature extraction model
            fe_model = dnn.Model(main_input, fe, name= 'fe_model')
            # # Classifier model as a keras model
            # rg_input = dnn.Input(shape =(fe.get_shape().as_list()[1],))  # input dim for the classifier 
            rg_input = dnn.Input(shape=(fe.shape[1],))
            net = regressor(rg_input)
            # # classifier keras model
            rg_model = dnn.Model(rg_input, net, name ='regressor')
            # #%% source model
            ms = dnn.Input(shape=(n_dim[1],))
            fes = feat_ext(ms)
            nets = regressor(fes)
            source_model = dnn.Model(ms, nets)
            source_model.compile(optimizer=optim, loss='mean_squared_error', metrics=['accuracy'])
            source_model.fit(source_data, source_labels, batch_size=128, epochs=100, validation_data=(target_data, target_labels))
            source_acc = source_model.evaluate(source_data, source_labels)
            target_acc = source_model.evaluate(target_data, target_labels)
            print("source loss & acc using source model", source_acc)
            print("target loss & acc using source model", target_acc)

            #%% Target model
            main_input = dnn.Input(shape=(n_dim[1],))
            # feature extraction model
            ffe=fe_model(main_input)
            # classifier model
            net = rg_model(ffe)
            # target model with two outputs: predicted class prob, and intermediate layers
            model = dnn.Model(inputs=main_input, outputs=[net, ffe])
            model.set_weights(source_model.get_weights())


            #%% deepjdot model and training
            from Deepjdot import Deepjdot

            batch_size=128
            sample_size=50
            sloss = 2.0; tloss=1.0; int_lr=0.002; jdot_alpha=5.0
            # DeepJDOT model initalization
            optim = tf.keras.optimizers.legacy.SGD(learning_rate=0.001)
            al_model = Deepjdot(model, batch_size,  optim, allign_loss=1.0,
                                sloss=sloss,tloss=tloss,int_lr=int_lr,jdot_alpha=jdot_alpha,
                                lr_decay=True,verbose=1)
            
            # DeepJDOT model fit
            print("source_data shape is:", source_data.shape, "source_labels shape is:", source_labels.shape, "target_data shape is:", target_data.shape)
            h,t_loss,tacc = al_model.fit(source_data, source_labels, target_data,
                                        n_iter=100, target_label=target_labels)


            #%% accuracy assesment
            tarmodel_sacc = al_model.evaluate(source_data, source_labels)    
            rmse, mae = al_model.evaluate(target_data, target_labels)
            print("target loss & acc using source+target model", "rmse is:", rmse, "mae is:", mae)
            print(rmse.numpy())
            rmses.append(rmse.numpy())
            maes.append(mae.numpy())
            print("time for one iteration is:", time.time()-start_time)


        print("rmses is:", rmses)
        print("maes is:", maes)
        save_results(rmses, maes, score_path)


