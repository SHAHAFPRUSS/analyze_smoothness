from random_forest import WaveletsForestRegressor
import pickle
import numpy as np
import time


def regressor(path):
    #input layer pickle file path
    y_c = pickle.load(open(path, 'rb'))
    y = np.eye(10)[y_c.astype(np.int32)][:, :9]
    #del y_c
    input = pickle.load(open(path, 'rb'))

    #regressor
    regressor = WaveletsForestRegressor(regressor='random_forest', trees=10, depth=None, features='sqrt', seed=21)
    rf = regressor.fit(input, y)

    #alpha paramenter
    alpha_input, _ = rf.evaluate_smoothness()
    print(f'the alpha input:{alpha_input}')

    del input
    return alpha_input, y


def get_smoothness_dict(model, alpha_input, y):
    alpha_dict = {'input': alpha_input}
    model == 'Toy_1'
    #layers = ['conv1_relu', 'conv2_relu', 'conv4_relu', 'conv6_relu',  'conv8_relu']
    layers = ['conv1_relu']
    for layer in layers:
        X = pickle.load(open('data/' + model + '_' + layer, 'rb'))
        rf = regressor.fit(X, y)
        alpha, MSE_errors = rf.evaluate_smoothness()
        print('\n', model, ': ', layer, ': ', alpha)

        alpha_dict[layer] = alpha
        pickle.dump(alpha_dict, open('results/alpha_dict_' + model, 'wb'))
        if layer == 'fc1':
            pickle.dump(MSE_errors, open('results/MSE_errors_no_misslab_' + model, 'wb'))

        del X

if __name__ == '__main__':
    input_path = 'C:/Users/IMOE001/Desktop/shahaf/mathematical_foundations_of_machine_learning/models/Toy_1_conv1_relu.pkl'
    alpha_input, y = regressor(input_path)
    get_smoothness_dict('Toy', alpha_input, y)
