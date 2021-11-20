import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import random
import itertools
import csv


def write_results(
        res=None, model=None,
        save_plot_loss=None, save_plot_metric=None, save_result=None,
        validation=False,
        test=False,
        show=False
    ):
    """

    @param res: results to write
    @param model: NeuralNetwork instance
    @param save_plot_loss: path where save loss
    @param save_plot_metric: path where save metric
    @param save_result: path where save numerical result
    @param validation: True if you want to save validation results
    @param test: True if you want to save test results
    @param show: True if you want to show plots

    """

    if save_plot_loss is not None:
        model.plot_loss(val=validation, test=test, show=show, path=save_plot_loss)
    if save_plot_metric is not None:
        model.plot_metric(val=validation, test=test, show=show, path=save_plot_metric)
    if save_result is not None:
        if validation and test:
            results = pd.DataFrame(res, index=['Training set', 'Validation set', 'Test set'])
        elif validation:
            results = pd.DataFrame(res, index=['Training set', 'Validation set'])
        elif test:
            results = pd.DataFrame(res, index=['Training set', 'Test set'])
        else:
            results = pd.DataFrame(res, index=['Training set'])
        results.to_csv(save_result, index=True)


def set_style_plot(style='seaborn', fig_size=(12, 10)):
    """

    @param style: style used for the plot
    @param fig_size: size the plots
    """
    plt.style.use(style)
    mpl.rcParams['figure.figsize'] = fig_size


def change_output_value(targets, old_value, new_value):
    """

    @param targets: target vector
    @param old_value: value to update
    @param new_value: updated value
    @return:
    """
    targets[targets == old_value] = new_value
    return targets


def train_test_split(X, Y, test_size=0.25, shuffle=False):
    """

    @param X: samples
    @param Y: targets
    @param test_size: size of the test (%)
    @param shuffle: True if you want shuffle data,
                    False otherwise
    @return: (X_train, Y_train, X_test, Y_test)
    """
    n_samples = X.shape[0]
    if shuffle:
        idx = np.random.permutation(n_samples)
        X = X[idx]
        Y = Y[idx]
    split = int(n_samples*test_size)
    X_test = X[:split]
    Y_test = Y[:split]
    X_train = X[split:n_samples]
    Y_train = Y[split:n_samples]
    return X_train, Y_train, X_test, Y_test


def train_val_test_split(X, Y, val_size=0.25, test_size=0.25, shuffle=False):
    """

    @param X: samples
    @param Y: targets
    @param test_size: size of the validation set (%)
    @param test_size: size of the test set (%)
    @param shuffle: True if you want shuffle data,
                    False otherwise
    @return: (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    """
    n_samples = X.shape[0]
    if shuffle:
        idx = np.random.permutation(n_samples)
        X = X[idx]
        Y = Y[idx]
    split_test = int(n_samples*test_size)
    split_val = int(n_samples*val_size)
    X_test = X[:split_test]
    Y_test = Y[:split_test]
    X_val = X[split_test:split_test+split_val]
    Y_val = Y[split_test:split_test+split_val]
    X_train = X[split_test+split_val:n_samples]
    Y_train = Y[split_test+split_val:n_samples]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def train_val_test_split_k_fold(X, Y,  test_size=0.25, shuffle=False,k_fold=5):
    """

        @param X: samples
        @param Y: targets
        @param test_size: size of the test set (%)
        @param shuffle: True if you want shuffle data,False otherwise
        @param k_fold : number of folds which to separate train and validation
        @return: (X_test,Y_test,folds_X,folds_Y)
        """
    n_samples = X.shape[0]
    if shuffle:
        idx = np.random.permutation(n_samples)
        X = X[idx]
        Y = Y[idx]
    split_test = int(n_samples * test_size)
    X_test = X[:split_test]
    Y_test = Y[:split_test]
    X_train=X[split_test+1:]
    Y_train = Y[split_test + 1:]
    split_val=int(len(X_train)/k_fold)
    folds_X=list()
    folds_Y=list()
    if(k_fold!=1):
        for i in range(0,k_fold):
            folds_X.append(X_train[i*split_val:split_val+(i*split_val)])
            folds_Y.append(Y_train[i*split_val:split_val+(i*split_val)])
    return (X_test,Y_test,folds_X,folds_Y)


def write_blind_result(results, path):
    """

    @param results: nparray of 2D dimension to write on csv results
    @param path: path of csv file
    @return: boolean success or failure
    """
    f = open("out/blind/results.csv", "w")
    f.write("# Riccardo Amadio Samuel Fabrizi \n")
    f.write("# Group Nickname \n")
    f.write("# ML-CUP19 \n")
    f.write("# 02/11/2019 \n")
    for i in range(1,results[0].size+1):
        f.write("{},{},{}\n".format(i,results[0][i-1],results[1][i-1]))
    f.close()


def generate_hyperparameters_combination(PARAMS, _random=False, max_evals=0, path_params=None):
    if _random:
        hyperaparams=list()
        for i in range(max_evals):
            hyperaparams.append({k: random.sample(v, 1)[0] for k, v in PARAMS.items()})
        return hyperaparams
    if not _random:
        Input = PARAMS
        f = open(path_params, 'w')
        with f:
            writer = csv.writer(f)
            Output = [writer.writerow([{key: value} for (key, value) in zip(Input, values)])
                      for values in itertools.product(*Input.values())]








