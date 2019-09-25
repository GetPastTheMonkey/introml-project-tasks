import h5py
import numpy as np
from operator import itemgetter
from keras.layers import BatchNormalization
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


config_for_unlabeled = {
    "neurons": [300, 400, 500],
    "dropouts": [0.1, 0.2, 0.3, 0.4],
    "components": [1],
    "layers": [3, 4, 5]
}
config_for_test_set = {
    "neurons": [300, 400, 500, 600],
    "dropouts": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "components": [1],
    "layers": [2, 3, 4, 5, 6]
}

def load_data_normalized():
    # Load labeled data
    with h5py.File('train_labeled.h5', 'r') as h5_file:
        hf = h5_file['train']
        x_train = np.array(hf['block0_values'])
        y_train = np.array(hf['block1_values']).astype("int")

    # Load unlabeled data
    with h5py.File("train_unlabeled.h5", "r") as h5_file:
        hf = h5_file["train"]
        x_unlabeled = np.array(hf['block0_values'])

    # Load testing data
    with h5py.File('test.h5', 'r') as h5_file:
        hf = h5_file['test']
        x_test = np.array(hf['block0_values'])

    # Bring to 0 mean
    x_train -= np.mean(x_train, axis=0)
    x_unlabeled -= np.mean(x_unlabeled, axis=0)
    x_test -= np.mean(x_test, axis=0)

    return x_train, y_train, x_unlabeled, x_test


def get_best_models(x_train_u, x_train, y_train, neurons, dropouts, components, layers):
    autoenc = Sequential()
    # encode
    autoenc.add(Dense(100, input_dim=x_train.shape[1], kernel_initializer="lecun_uniform"))
    autoenc.add(BatchNormalization())
    autoenc.add(Activation("tanh"))
    autoenc.add(Dense(50, kernel_initializer="lecun_uniform"))
    autoenc.add(BatchNormalization())
    autoenc.add(Activation("relu"))

    #decode
    autoenc.add(Dense(50, kernel_initializer="lecun_uniform"))
    autoenc.add(BatchNormalization())
    autoenc.add(Activation("tanh"))
    autoenc.add(Dense(x_train.shape[1], kernel_initializer="lecun_uniform"))
    autoenc.add(BatchNormalization())
    autoenc.add(Activation("relu"))

    autoenc.compile(optimizer="adadelta", loss="mse")
    autoenc.fit(x_train_u, x_train_u, batch_size=256, epochs=10, shuffle=True, validation_split=0.20)

    models = dict()
    y = np_utils.to_categorical(y_train, 10)
    best_val_acc = None

    # Loop through all possible combinations
    for comps in components:
        for dropout in dropouts:
            for neuron in neurons:
                for layer in layers:

                    # Define model
                    model = Sequential()
                    for i in range(layer):
                        if i == 0:
                            model.add(autoenc.layers[0])
                            model.add(autoenc.layers[1])
                            model.add(autoenc.layers[2])
                            model.add(Dense(neuron, kernel_initializer="lecun_uniform"))
                        else:
                            model.add(Dense(neuron, kernel_initializer="lecun_uniform"))
                        model.add(BatchNormalization())
                        if i % 2 == 0:
                            model.add(Activation("relu"))
                        else:
                            model.add(Activation("tanh"))
                        model.add(Dropout(dropout))

                    # Add output layer
                    model.add(Dense(10, kernel_initializer="lecun_uniform"))
                    model.add(Activation("softmax"))

                    # Compile and fit model
                    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
                    hist = model.fit(x_train, y, batch_size=256, epochs=60, verbose=2, shuffle=True,
                                     validation_split=0.1)
                    val_acc = hist.history['val_acc'][-1]

                    # Check if layer already in models dict
                    if layer in models.keys():
                        # Check if current model is better
                        _, so_far_best = models[layer]
                        if so_far_best < val_acc:
                            models[layer] = (model, val_acc)
                    else:
                        # Insert current model
                        models[layer] = (model, val_acc)

                    if best_val_acc is None or best_val_acc < val_acc:
                        print("NEW BEST MODEL: comp={}, drop={}, neuron={}, layer={}. Got accuracy {}".format(
                            comps, dropout, neuron, layer, val_acc))
                        best_val_acc = val_acc
                    else:
                        print("Finished model: comp={}, drop={}, neuron={}, layer={}. Got accuracy {}. Best is {}".format(
                            comps, dropout, neuron, layer, val_acc, best_val_acc))

    # Return one model for each layer count
    return_list = []
    for model, _ in models.values():
        return_list.append(model)
    return return_list


def predict_classes(models, x):
    probabilities = None
    for model in models:
        probas = model.predict_proba(x, batch_size=256)
        if probabilities is None:
            probabilities = probas
        else:
            probabilities += probas
    probabilities /= len(models)
    result = np.argmax(probabilities, axis=1)
    return result.reshape((len(result), 1))


def main(config_1, config_2):
    x_l, y_l, x_u, x_t = load_data_normalized()
    print("GENERATING MODELS FOR UNLABELED DATA")
    models_for_unlabeled = get_best_models(x_u, x_l, y_l, **config_1)
    y_t = predict_classes(models_for_unlabeled, x_t)

    #x_final = np.concatenate((x_l, x_u), axis=0)
    #y_final = np.concatenate((y_l, y_u), axis=0)

    #print("GENERATING MODELS FOR TESTING DATA")
    #models_for_testset = get_best_models(x_final, y_final, **config_2)
    #y_t = predict_classes(models_for_testset, x_t)

    with open("result.csv", "w") as f:
        f.write('Id,y\n')
        column_id = 30000
        for val in y_t:
            f.write('{},{}\n'.format(column_id, str(val)[1]))
            column_id += 1


main(config_for_unlabeled, config_for_test_set)
