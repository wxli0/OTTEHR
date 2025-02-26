import getpass
user_id = getpass.getuser()

import sys
sys.path.append(f"/home/{user_id}/OTTEHR/")
sys.path.append(f"/home/{user_id}/unbalanced_gromov_wasserstein/")
sys.path.append(f"/home/{user_id}/OTTEHR/competitors/deepJDOT")


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
from tensorflow.keras import layers


import dnn
from Deepjdot import Deepjdot


n_components = 50
#%% feature extraction and regressor function definition
def feat_ext(main_input, l2_weight=0.0):
    net = dnn.Dense(2*n_components, activation='relu', name='fe')(main_input)
    net = dnn.Dense(n_components, activation='relu', name='feat_ext')(net)
    return net
    

def regressor(model_input, l2_weight=0.0):
    net = dnn.Dense(n_components/2, activation='relu', name='rg')(model_input)
    net = dnn.Dense(1, activation='linear', name='rg_output')(net)
    return net




output_dir = mimic_output_dir
print(f"Will save outputs to {output_dir}")


source_count = 120
target_count = 100
iterations = 100
trans_metric = 'deepJDOT'

# # For insurance experiments
# group_name = 'insurance'
# groups = ['Self_Pay', 'Private', 'Government', 'Medicare', 'Medicaid']
# source = 'Self_Pay'
# target_groups = ['Private']
# type = 'cat'
# append_features = []
# suffix = None


# For age experiment, update group_name and groups to appropriate values 
group_name = 'age'
source = [50, 70]
target_groups = [[25, 45], [30, 50], [35, 55], [45, 65], [50, 70], [55, 75], [60, 80]]
type = 'cts'
append_features = ['age_normalized']
suffix = "with_age"


# groups.reverse()


""" 
Read in the original dataframe
"""

admid_diagnosis_df = pd.read_csv(os.path.join(output_dir, f"admission_patient_diagnosis_ICD.csv"), index_col=0, header=0, converters={'ICD codes': literal_eval})
if suffix is not None:
    admid_diagnosis_df = pd.read_csv(os.path.join(output_dir, f"admission_patient_diagnosis_ICD_{suffix}.csv"), index_col=0, header=0, converters={'ICD codes': literal_eval})
    admid_diagnosis_df['age_normalized'] = admid_diagnosis_df['age'] / 80

print(admid_diagnosis_df)



for target in target_groups:
    # if source == target:
    #     continue

    
    print(f"source is: {source}, target is: {target}")
    score_path = os.path.join(output_dir, f"{group_name}_{target}_to_{source}_{trans_metric}.csv")
    # if os.path.exists(score_path):
    #     continue

    maes = []
    rmses = []
    for i in range(iterations):
        selected_df = select_samples(admid_diagnosis_df, group_name, type, source, target, source_count, target_count)
        code_feature_name = 'ICD codes'
        label_name = 'duration'
        source_data, source_labels, target_data, target_labels = gen_code_feature_label(selected_df, group_name, type, source, target, code_feature_name, label_name, append_features=append_features)
        n_dim = np.shape(source_data)
        print("n_dim is:", n_dim)
        optim = tf.keras.optimizers.legacy.SGD(lr=0.01)
            
        # #%% Feature extraction as a keras model
        main_input = dnn.Input(shape=(n_dim[1],))
        print("main_input is:", main_input)
        fe = feat_ext(main_input)
        # # feature extraction model
        fe_model = dnn.Model(main_input, fe, name= 'fe_model')
        # # Classifier model as a keras model
        rg_input = dnn.Input(shape =(fe.get_shape().as_list()[1],))  # input dim for the classifier 
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
        al_model = Deepjdot(model, batch_size, optim,allign_loss=1.0,
                            sloss=sloss,tloss=tloss,int_lr=int_lr,jdot_alpha=jdot_alpha,
                            lr_decay=True,verbose=1)
        # DeepJDOT model fit
        h,t_loss,tacc = al_model.fit(source_data, source_labels, target_data,
                                    n_iter=100, target_label=target_labels)


        #%% accuracy assesment
        tarmodel_sacc = al_model.evaluate(source_data, source_labels)    
        rmse, mae = al_model.evaluate(target_data, target_labels)
        print("target loss & acc using source+target model", "rmse is:", rmse, "mae is:", mae)
        print(rmse.numpy())
        rmses.append(rmse.numpy())
        maes.append(mae.numpy())
    

    print("rmses is:", rmses)
    print("maes is:", maes)
    save_results(rmses, maes, score_path)


