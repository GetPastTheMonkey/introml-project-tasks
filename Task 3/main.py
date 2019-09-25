import numpy as np
import h5py
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import scale

# Load training data
with h5py.File('train.h5', 'r') as f:
    hf = f['train']
    train_x = np.array(hf['block0_values'])
    train_y = np.array(hf['block1_values'])

# Load testing data
with h5py.File('test.h5', 'r') as f:
    hf = f['test']
    test_x = np.array(hf['block0_values'])

#config
kbests = [ 75, 80, 85, 90, 95, 100]
#folds = 6
#alphas = [1e-4]
neurons = [350, 400, 450, 475, 500, 525, 550, 600, 650]
activations_1 = ['relu']
activations_2 = ['tanh']
activations_3 = ['relu']
dropouts = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4]
max_acc = 0
best_k = 0
best_neuron = 0
best_act_1 = 0
best_act_2 = 0
best_act_3 = 0
best_drop = 0
#solvers = ['adam']


# Apply configs
for k1 in kbests:
    for neuron in neurons:
        for act_1 in activations_1:
            for act_2 in activations_2:
                for act_3 in activations_3:
                    for drop in dropouts:
                        # Select features
                        selector = SelectKBest(k=k1)
                        selector.fit(train_x, train_y)
                        train_x1 = selector.transform(train_x)
                        test_x1 = selector.transform(test_x)

                        # Scale data
                        train_x1 = scale(train_x1)
                        test_x1 = scale(test_x1)

                        # Set y to categorical
                        train_y1 = to_categorical(train_y)

                        # Create model
                        model = Sequential()
                        # Input layer
                        model.add(Dense(neuron, input_dim=k1, kernel_initializer="uniform"))
                        model.add(BatchNormalization())
                        model.add(Activation(act_1))
                        model.add(Dropout(drop))
                        # Hidden layer 1
                        model.add(Dense(neuron, kernel_initializer="uniform"))
                        model.add(BatchNormalization())
                        model.add(Activation(act_2))
                        model.add(Dropout(drop))
                        # Hidden layer 2
                        model.add(Dense(neuron, kernel_initializer="uniform"))
                        model.add(BatchNormalization())
                        model.add(Activation(act_3))
                        model.add(Dropout(drop))
                        # Hidden layer 3
                        model.add(Dense(neuron, kernel_initializer="uniform"))
                        model.add(BatchNormalization())
                        model.add(Activation('tanh'))
                        model.add(Dropout(drop))

                        # Output layer
                        model.add(Dense(5, kernel_initializer="uniform"))
                        model.add(BatchNormalization())
                        model.add(Activation("softmax"))

                        # Compile model
                        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])


                        # Fit data
                        hist = model.fit(train_x1, train_y1, validation_split=0.1, epochs=60, batch_size=512, verbose=2)

                        #train_loss = hist.history['loss']
                        #val_loss = hist.history['val_loss']
                        #train_acc = hist.history['acc']
                        val_acc = hist.history['val_acc'][-1]

                        print("\n=====Finished with:\tk={}\tneuron={}\tact_1={}\tact_2={}\tact_3={}\tdrop={}=====\n".format(k1, neuron, act_1, act_2, act_3, drop))

                        if val_acc >= max_acc:
                            print("\n*****Got new max_acc:\t{}*****".format(val_acc))
                            test_y = model.predict_classes(test_x1, batch_size=32)
                            max_acc = val_acc
                            best_k = k1
                            best_neuron = neuron
                            best_act_1 = act_1
                            best_act_2 = act_2
                            best_act_3 = act_3
                            best_drop = drop

                            print("Best Parameters:\tk={}\tneuron={}\tact_1={}\tact_2={}\tact_3={}\tdrop={}\n".format(k1, neuron, act_1, act_2, act_3, drop))

                            # Write output file
                            column_id = train_y1.shape[0]
                            with open('result.csv', 'w') as f:
                                f.write('Id,y\n')
                                for val in test_y:
                                    f.write('{},{}\n'.format(column_id, val))
                                    column_id += 1
                        else:
                            print("\n-----Nothing changed. max_acc:\t{}-----".format(max_acc))
                            print("Best Parameters:\tk={}\tneuron={}\tact_1={}\tact_2={}\tact_3={}\tdrop={}\n".format(best_k, best_neuron, best_act_1, best_act_2, best_act_3, best_drop))



