import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import networkx as nx
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from CINN_customized_structure_v2 import CINN



### Load data
data_name = 'TEP_Faulty_Training_Sample.csv'
target_variable_name = 'xmeas_15'

df = pd.read_csv(data_name)
columns_names = list(df.columns)
node_labels = columns_names
target_node = list(df.columns).index(target_variable_name)  

print('target variable name:',target_variable_name, ', node ID: ', target_node, node_labels)


# 'layers_of_intermediate_nodes': [[22,23,24,25,31],[5,7],[8],[6],[21],[10],[12,26,13],[28],[14],[16],[29],[18]]

causality_structure = { 'root_nodes':[0,1,2,3,4,9,11,20], 'intermediate_nodes':[5,6,7,8,10,12,13,14,16,18,21,22,23,24,25,26,28,29,31], 'leaf_nodes':[15,17,19,27,30,32], 
'layers_of_intermediate_nodes': [[22,23,24,25,31],[5,7],[8],[6],[21],[10],[12,26,13],[28],[14],[16],[29],[18]] }

print(causality_structure)



### Load causality structure into neural network
scaler = StandardScaler()
scaler.fit(df)
data = scaler.transform(df)

train, test, _ , _ = train_test_split(data, data, test_size=0.2, random_state=42)

# without root_nodes (root_nodes are [4,7])
nodes_list = [5,6,7,8,10,12,13,14,16,18,21,22,23,24,25,26,28,29,31,15,17,19,27,30,32]

# To check target_node position in the nodes_list 
target_node_index = nodes_list.index(target_node)  # 5

root_nodes = [0,1,2,3,4,9,11,20]
intermediate_nodes = [5,6,7,8,10,12,13,14,16,18,21,22,23,24,25,26,28,29,31]
leaf_nodes = [15,17,19,27,30,32]

kf = KFold(n_splits = 10, random_state = 2, shuffle=True)

X_test = test[:, root_nodes]
y_test_int = test[:, intermediate_nodes]
y_test_output = test[:, leaf_nodes]


batch_size = 256
no_epochs = 600
results = []
mse_best = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print(f"Fold {fold}:")
    
    X_train = train[train_idx][:, root_nodes]
    X_val = train[val_idx][:, root_nodes]
    
    y_train_int = train[train_idx][:, intermediate_nodes]
    y_train_output = train[train_idx][:, leaf_nodes]
    
    y_val_int = train[val_idx][:, intermediate_nodes]
    y_val_output = train[val_idx][:, leaf_nodes]

    model = CINN(causality_structure, target_node, batch_size, 'original')
    # plot the architecture of NN model
    tf.keras.utils.plot_model(model.model, to_file= 'architecture.png', show_shapes=True, show_layer_names=True)
    model.assign_data(X_train, y_train_int, y_train_output, X_val, y_val_int, y_val_output,
                      X_test, y_test_int, y_test_output)

    initial_weights = model.model.get_weights()
    optim = tf.keras.optimizers.Adam()
    model.train(no_epochs, optim)

    # Choose the best weights on the validation data from 10 fold results 
    model.model.set_weights(model.best_weights)
    y = model.y_test_combined[:, target_node_index]
    y_pred = model.model.predict(model.X_test)[:, target_node_index]
    model.model.reset_states()
    
    mse = mean_squared_error(y, y_pred)
    results.append([fold, 'No_PCGrad', mse])
    print (fold, mse)

df_results = pd.DataFrame(results, columns = ['Run', 'Method', 'MSE'])

df_results.to_csv('results.csv', index = False)

print (df_results['MSE'].mean())
print (df_results['MSE'].std()) 
