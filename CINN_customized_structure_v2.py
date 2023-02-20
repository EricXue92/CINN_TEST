import tensorflow as tf
from keras import regularizers
from keras.layers import Dense, Input, Dropout
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error

tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

class CINN:
    def __init__(self, casuality_structure, target_node, batch_size, method, seed=0):
        
        self.casuality_structure = casuality_structure
        self.input_shape = len(self.casuality_structure['root_nodes'])
        # return the number of intermediate and leaf nodes 
        self.no_intermediates = len(self.casuality_structure['intermediate_nodes'])
        self.no_outputs = len(self.casuality_structure['leaf_nodes'])

        self.input_nodes = self.casuality_structure['root_nodes']
        self.intermediate_nodes = self.casuality_structure['intermediate_nodes']
        self.layers_of_intermediate_nodes = self.casuality_structure['layers_of_intermediate_nodes']
        self.output_nodes = self.casuality_structure['leaf_nodes']

        self.nodes_list = self.intermediate_nodes + self.output_nodes

        self.target_node_index = self.nodes_list.index(target_node)
        # 
        self.spacer_index = self.no_intermediates


        self.seed = seed

        if target_node in self.intermediate_nodes:
            self.target_node_flag = 'intermediate'
        else:
            self.target_node_flag = 'output'

        self.batch_size = batch_size
        self.method = method
        self.loss = []

        # return the model
        self.model = self.build_causality_informed_model()

    def build_causality_informed_model(self):
        tf.print ('Number of intermediate layers: ', len(self.layers_of_intermediate_nodes))

        if len(self.layers_of_intermediate_nodes) > 1:
            inputs = Input(shape=self.input_shape)
            curr = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=regularizers.l2(0.0001))(inputs)
            curr = Dropout(0.2)(curr)
            ## intermediate_outputs
            curr_intermediate = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=regularizers.l2(0.0001))(curr)
            for i in range(len(self.layers_of_intermediate_nodes)):
                tf.print (i, self.layers_of_intermediate_nodes[i])
                current_intermediate_layer_nodes = len(self.layers_of_intermediate_nodes[i])

                cur_input = curr_intermediate
                curr_intermediate = Dense(current_intermediate_layer_nodes, activation='linear', name='intermediate_layer_' + str(i))(cur_input)
                if i == 0:
                    intermediate_outputs = curr_intermediate
                else:
                    intermediate_outputs = tf.concat([intermediate_outputs, curr_intermediate], axis=1)
                
            ## combine intermediate outputs with another layer to generate final outputs
            curr_ = Dense(16, activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=regularizers.l2(0.0001))(curr)
            curr_ = Dense(8, activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=regularizers.l2(0.0001))(tf.concat([curr_, curr_intermediate], 1))

            outputs = Dense(self.no_outputs, activation='linear', name='final_outputs')(curr_)

            return keras.Model(inputs=inputs, outputs=tf.concat([intermediate_outputs, outputs], axis=1))

        else:
            inputs = Input(shape=self.input_shape)
            curr = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01),
            kernel_regularizer=regularizers.l2(0.0001))(inputs)
            curr = Dropout(0.2)(curr)

            ## intermediate_outputs
            curr_intermediate = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01),
            kernel_regularizer=regularizers.l2(0.0001))(curr)

            intermediate_outputs = Dense(self.no_intermediates, activation='linear', name='intermediate_outputs')(curr_intermediate)

            ## combine intermediate outputs with another layer to generate final outputs
            curr_ = Dense(16, activation='relu', 
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0001), kernel_regularizer=regularizers.l2(0.0001))(curr)
            curr_ = Dense(8, activation='relu', 
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0001), kernel_regularizer=regularizers.l2(0.0001))(tf.concat([curr_, intermediate_outputs], 1))

            outputs = Dense(self.no_outputs, activation='linear', name='final_outputs')(curr_)

            return keras.Model(inputs=inputs, outputs=tf.concat([intermediate_outputs, outputs], axis=1))

    # get the extra infection causal loss 
    # def gradient_loss(self, x_batch):

    #         # Operations are recorded if they are executed within this context manager 
    #         # and at least one of their inputs is being "watched".

    #         # x = tf.constant(3.0)
    #         # with tf.GradientTape() as g:
    #         #   g.watch(x)
    #         #   y = x * x
    #         # dy_dx = g.gradient(y, x)
    #         # print(dy_dx) 

    #     with tf.GradientTape(persistent=True) as g:
    #         g.watch(x_batch)
    #         y_intermediate = self.model(x_batch)[:, :self.spacer_index][:, -1] ### MEDV
    #         y_output = self.model(x_batch)[:, self.spacer_index:][:, -1] ### Proportion of black     ????

    #     g_intermediate = g.gradient(y_intermediate, x_batch)
    #     g_output = g.gradient(y_output, x_batch)

    #     # ### MEDV w.r.t. the crime rate (negative)

    #     # Computes the mean of elements across dimensions of a tensor.
    #     causality_loss_1 = tf.reduce_mean( tf.square( tf.maximum(0.0, g_intermediate[:, 0] - 0.01) ) )
        
    #     ### MEDV w.r.t. lower status of the population (negative)
    #     causality_loss_2 = tf.reduce_mean(tf.square(tf.maximum(0.0, g_intermediate[:, -1] - 0.01)))
        
    #     ### MEDV w.r.t. RM (positive)
    #     causality_loss_3 = tf.reduce_mean(tf.square(tf.maximum(0.0, -1*g_intermediate[:, 3] - 0.01)))

    #     ###   
    #     causality_loss_4 = tf.reduce_mean(tf.square(tf.maximum(0.0, g_output[:, 0] - 0.01)))

    #     ### MEDV w.r.t. NOX (negative)
    #     causality_loss_5 = tf.reduce_mean(tf.square(tf.maximum(0.0, g_intermediate[:, 2] - 0.01)))

    #     return 32*(causality_loss_1 + causality_loss_2 + causality_loss_3 + causality_loss_4 + causality_loss_5)

    # Get the total loss value 
    def loss_fn(self, x_batch, y_batch):
        y_pred = self.model(x_batch)

        intermediate_loss = tf.reduce_mean(tf.square(y_batch[:, :self.spacer_index] - y_pred[:, :self.spacer_index]), axis=0)

        output_loss = tf.reduce_mean(tf.square(y_batch[:, self.spacer_index:] - y_pred[:, self.spacer_index:]), axis=0)

        # get the extra causal infection loss 
        # causality_loss = self.gradient_loss(x_batch)

        if self.method == 'original':
            return [tf.reduce_mean(intermediate_loss) + tf.reduce_mean(output_loss)]
        else:
            print ('PCGrad starts ...............')
            if self.domain_knowledge_flag:
                ## check if there are nodes in the intermediate layers
                if len(self.intermediate_nodes) > 0:
                    return [tf.reduce_mean(intermediate_loss), tf.reduce_mean(output_loss) + causality_loss]
                else:
                    return [tf.reduce_mean(output_loss) + causality_loss]
            else:
                if len(self.intermediate_nodes) > 0:
                    return [tf.reduce_mean(intermediate_loss), tf.reduce_mean(output_loss)]
                else:
                    return [tf.reduce_mean(output_loss)]

    def assign_data(self, X_train, y_train_int, y_train_output, X_val, y_val_int, y_val_output, X_test, y_test_int, y_test_output):
        self.X_train = X_train.astype('float32')
        self.y_train_int = y_train_int.astype('float32')
        self.y_train_output = y_train_output.astype('float32')

        self.X_val = X_val.astype('float32')
        self.y_val_int = y_val_int.astype('float32')
        self.y_val_output = y_val_output.astype('float32')

        self.y_val_combined = np.concatenate((self.y_val_int, self.y_val_output), axis = 1)
        
        self.X_test = X_test.astype('float32')
        self.y_test_int = y_test_int.astype('float32')
        self.y_test_output = y_test_output.astype('float32')

        self.y_test_combined = np.concatenate((self.y_test_int, self.y_test_output), axis = 1)
        
        self.X_train = tf.convert_to_tensor(self.X_train, dtype=tf.float32)
        self.y_train_int = tf.convert_to_tensor(self.y_train_int, dtype=tf.float32)
        self.y_train_output = tf.convert_to_tensor(self.y_train_output, dtype=tf.float32)

        self.y_train_combined = np.concatenate((self.y_train_int, self.y_train_output), axis = 1)
 
    # Get the loss and gradient 
    def get_grad(self, x_batch, y_batch):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(x_batch, y_batch)

            # x = tf.constant(3.0)
            # with tf.GradientTape() as g:
            #   g.watch(x)
            #   y = x * x
            # dy_dx = g.gradient(y, x)
            # print(dy_dx) 

            # gradient(
            #     target,
            #     sources,
            #     output_gradients=None,
            #     unconnected_gradients=tf.UnconnectedGradients.NONE )

        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, g

    # def get_grad_by_PCG(self, x_batch, y_batch):
    #     with tf.GradientTape() as tape:
    #         loss = self.loss_fn(x_batch, y_batch)
            
    #         assert type(loss) is list

    #         loss = tf.stack(loss)
    #         tf.random.shuffle(loss)

    #         grads_task = tf.vectorized_map(lambda x: tf.concat([tf.reshape(grad, [-1,]) 
    #                             for grad in tape.gradient(x, self.model.trainable_variables)
    #                             if grad is not None], axis=0), loss)
        
    #     num_tasks = len(loss)

    #     # Compute gradient projections.
    #     def proj_grad(grad_task):
    #         for k in range(num_tasks):
    #             inner_product = tf.reduce_sum(grad_task*grads_task[k])
    #             proj_direction = inner_product / tf.reduce_sum(grads_task[k]*grads_task[k])
    #             grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
    #         return grad_task

    #     proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)

    #     # Unpack flattened projected gradients back to their original shapes.
    #     proj_grads = []
    #     for j in range(num_tasks):
    #         start_idx = 0
    #         for idx, var in enumerate(self.model.trainable_variables):
    #             grad_shape = var.get_shape()
    #             flatten_dim = np.prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
    #             proj_grad = proj_grads_flatten[j][start_idx:start_idx+flatten_dim]
    #             proj_grad = tf.reshape(proj_grad, grad_shape)
    #             if len(proj_grads) < len(self.model.trainable_variables):
    #                 proj_grads.append(proj_grad)
    #             else:
    #                 proj_grads[idx] += proj_grad             
    #             start_idx += flatten_dim

    #     grads_and_vars = list(zip(proj_grads, self.model.trainable_variables))
        
    #     del tape
    #     return loss, proj_grads

    def train(self, N, optimizer):

        @tf.function
        def train_step(x_batch, y_batch):
            if self.method == 'original':
                loss, grad_theta = self.get_grad(x_batch, y_batch)
                
            # if self.method == 'PCG_gradient':
            #     loss, grad_theta = self.get_grad_by_PCG(x_batch, y_batch)
            
            # Perform gradient descent step ( Update weights )
            optimizer.apply_gradients( zip(grad_theta, self.model.trainable_variables) )
            
            return loss, grad_theta

        ### Perform gradient descent optimization

        # tf.data.Dataset.from_tensor_slices () Return : Return the objects of sliced elements.
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train_combined))
        train_dataset = train_dataset.shuffle(buffer_size=1024, seed=self.seed).batch(self.batch_size)
        
        best_value = 1e9

        for epoch in range(N):
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss, grad_theta = train_step(x_batch_train, y_batch_train)

            self.loss.append(loss)

            y_pred_val = self.model(self.X_val)

            if self.target_node_flag == 'output':
                val_loss = mean_squared_error(y_pred_val[:, self.spacer_index:], self.y_val_combined[:, self.spacer_index:])

            elif self.target_node_flag == 'intermediate':
                #tf.print (tf.shape(self.y_val_combined), tf.shape(y_pred_val), tf.shape(self.y_val_combined), self.spacer_index)
                val_loss = mean_squared_error(y_pred_val[:, :self.spacer_index], self.y_val_combined[:, :self.spacer_index])

            if val_loss < best_value:
                best_value = val_loss

                self.min_val_loss = best_value
                self.best_weights = self.model.get_weights()
                #print("Saving model")
                self.count = 0
            else:
                self.count += 1