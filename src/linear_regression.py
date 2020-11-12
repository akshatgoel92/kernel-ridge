#!/usr/bin/env python
# coding: utf-8
# Import packages
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data():
    '''
    ------------------------
    Input: None
    Output: Assignment data
    ------------------------
    '''
    return np.array([1, 2, 3, 4]), np.array([3, 2, 0, 5])  


def get_polynomial_basis(x, k):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Polynomial basis features
    Given a dimension this iterates
    through orders [0 to k-1]
    ------------------------
    '''
    grid = np.meshgrid(x, np.arange(k)) 
    return np.power(grid[0], grid[1]).T


def get_sol(X, Y):
    '''
    ------------------------
    Input: Polynomial features
    Output: Least squares solution
    ------------------------
    '''
    return np.linalg.solve(X.T @ X, X.T @ Y)


def get_predictions(X, beta_hat):
    '''
    ------------------------
    Input: 
           1) Polynomial features
           2) Least squares coefficients
    Output: Predictions
    ------------------------
    '''
    return X @ beta_hat


def get_mse(Y, Y_hat):
    '''
    ------------------------
    Input: True values and predicted values
    Output: Mean squared error
    ------------------------
    '''
    return np.sum(np.power(Y-Y_hat, 2))/max(Y.shape)


def get_ln_mse(mse):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    '''
    return np.log(mse)


def run_polynomial_regression(k, x, y):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    ''' 
    phi_x = get_polynomial_basis(x, k)
    beta_hat = get_sol(phi_x, y)
    y_hat = get_predictions(phi_x, beta_hat)
    
    mse = get_mse(y, y_hat)
    ln_mse = get_ln_mse(mse)
    
    results = {'beta_hat': beta_hat, 'y_hat': y_hat, 
               'mse': mse, 'ln_mse': ln_mse, 
               'degree': k-1, 'dim': k}
    
    return(results)


def get_final_results(results):
    '''
    ------------------------
    Input: True values and predicted values
    Output: Mean squared error
    ------------------------
    '''
    mse = pd.DataFrame([result['mse'] for result in results], columns = ['MSE'])
    mse['degree'] = [result['degree'] for result in results]
    mse.set_index('degree', inplace = True)
    
    return(mse)


def plot_regression_predictions(path, title, start_k, end_k, 
                                results, x, y, get_basis,  
                                x_lab = "X", y_lab="Y", add_data=True):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    ''' 
    plt.clf()

    # Create dimensions to iterate over
    dims = range(start_k, end_k + 1)
    
    # Make grids for plot
    x_grid = np.linspace(-5, 5, 100000)
    
    # These iterate through orders/degrees [0 to k-1] using dimension
    x_basis = [get_basis(x_grid, dim) for dim in dims]
    y_grid = [get_predictions(basis, result['beta_hat']) for basis, result in zip(x_basis, results)]
    
    # Plots
    for i, y_pred in enumerate(y_grid):
        plt.plot(x_grid, y_pred, label = str(i + start_k))

    # Overlay the observed data if the add_data option is switched on is passed through
    # Set the axes
    if add_data:
        plt.plot(x, y, "r.")
        axes = plt.gca()
        axes.set_xlim([0,1])
        axes.set_ylim([-1.5,1.5])

    # Add legend
    # Set the axes
    if not add_data: 
        plt.legend(title="k")
        axes = plt.gca()
        axes.set_xlim([0,5])
        axes.set_ylim([-5,8])
    
    
    # Add annotations
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    
    # Display and save plot
    plt.savefig(path)
    plt.show()


def main(start_k = 1, end_k = 4):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    ''' 
    get_basis = get_polynomial_basis

    x, y = get_data()
    results = [run_polynomial_regression(dim, x, y) for dim in range(start_k, end_k + 1)]
    
    # For plot
    title = 'Polynomial Basis Fits'
    path = os.path.join('.', 'figs', '1_1.png')
    
    df = get_final_results(results)
    plot_regression_predictions(path, title, start_k, end_k, results, x, y, get_basis, add_data = False)
    
    return(results, df)



if __name__ == '__main__':
    main()