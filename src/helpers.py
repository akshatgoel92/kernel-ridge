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


def get_basis(x, k):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Polynomial basis features
    ------------------------
    '''
    grid = np.meshgrid(x, np.arange(k + 1)) 
    return np.power(grid[0], grid[1]).T


def get_sol(phi_x, y):
    '''
    ------------------------
    Input: Polynomial features
    Output: Least squares solution
    ------------------------
    '''
    return np.linalg.inv(phi_x.T @ phi_x) @ phi_x.T @ y


def get_predictions(phi_x, beta_hat):
    '''
    ------------------------
    Input: 
           1) Polynomial features
           2) Least squares coefficients
    Output: Predictions
    ------------------------
    '''
    return phi_x @ beta_hat


def get_mse(y, y_hat):
    '''
    ------------------------
    Input: True values and predicted values
    Output: Mean squared error
    ------------------------
    '''
    return 1/max(y.shape)*np.sum(np.power(y-y_hat, 2))



def get_final_results_df(results):
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


def run_regression(k, x, y, loss_func):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    ''' 
    phi_x = get_basis(x, k)
    beta_hat = get_sol(phi_x, y)
    y_hat = get_predictions(phi_x, beta_hat)
    
    mse = loss_func(y, y_hat)
    results = {'beta_hat': beta_hat, 'y_hat': y_hat, 'mse': mse, 'degree': k}
    
    return(results)


def plot_results(path, title, k, results, x_lab = "X", y_lab="Y"):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    ''' 
    # Make grids for plot
    x_grid = np.linspace(-5, 5, 100000)
    x_basis = [get_basis(x_grid, degree) for degree in range(k + 1)]
    y_grid = [get_predictions(basis, results[deg]['beta_hat']) for basis, deg in zip(x_basis, range(k + 1))]
    
    # Plots
    for degree in range(k + 1):
        plt.plot(x_grid, y_grid[degree], label = str(degree))
    
    # Add annotations
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.legend()
    
    axes = plt.gca()
    axes.set_xlim([0,5])
    axes.set_ylim([-5,8])

    
    # Display and save plot
    plt.show()
    plt.savefig(path)