# Import packages
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import run_regression, get_mse


def get_noisy_basis(x, loc = 0, var = 0.007): 
    '''
    ---------------------
    Input:
    Output: 
    ---------------------
    '''
    return np.sin(2*math.pi*x)**2 + np.random.normal(0, np.sqrt(var))


def get_true_basis(x):
    '''
    ---------------------
    Input:
    Output: 
    ---------------------
    '''
    return np.sin(2*math.pi*x)**2


def get_training_data(n, lower, upper, var):
    '''
    ---------------------
    Input:
    Output: 
    ---------------------
    '''
    np.random.seed(1291803123)
    x = np.random.uniform(lower, upper, n)
    y = np.array([get_true_basis(example) for example in x])
    y_hat = np.array([get_noisy_basis(example, var) for example in x])
    
    return x, y, y_hat


def plot_data(x, y_hat, path):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    '''
    # Close any figures currently open
    plt.clf()

    # Make grids of x and y points to plot
    x_grid = np.linspace(0, 5, 1000000)
    y_grid = np.array([get_true_basis(example) for example in x_grid])

    # Title of the plot
    title = "True basis function vs. Noisy data points"

    # Plot the true curve
    plt.plot(x_grid, y_grid, label = "True function")
    plt.plot(x, y_hat, "ro", )
    
    # Add annotations
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    
    
    axes = plt.gca()
    axes.set_xlim([0,1.6])
    axes.set_ylim([0,1.6])

    
    # Display and save plot
    plt.savefig(path)


def get_ln_mse(y, y_hat, tolerance=1.220446049250313e-5):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    '''
    return np.sum(2*np.log(y-y_hat + tolerance))


def run_polynomial_regression(x, y, lower = 1, upper = 18):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    '''
    results = np.array([run_regression(k, x, y_hat, get_mse) for k in range(lower, upper)])
    ln_mse = np.log(np.array([result['mse'] for result in results]))
    
    return ln_mse



def plot_training_loss(mse, highest_k, path):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    '''
    # Close any currently open plots
    plt.clf()

    # Title of the plot
    title = "Training error vs. polynomial degree"

    # Make grids of x and y points to plot
    x_grid = np.arange(1, highest_k, 1)
    
    # Plot the true curve
    plt.plot(x_grid, mse, "ro")
    
    # Add annotations
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    
    
    # Get current access
    axes = plt.gca()
    axes.set_xlim([0,20])
    axes.set_ylim([-3,3])

    
    # Display and save plot
    plt.savefig(path)


if __name__ == '__main__':


    # Get and plot initial data
    x, y, y_hat = get_training_data(30, 0, 1, 0.07)
    plot_data(x, y_hat, os.path.join("figs", '1_2_data.png'))
    
    # Run regressions
    lower = 1 
    upper = 18
    ln_mse = run_polynomial_regression(x, y_hat, lower, upper) 

    # Plot training loss
    plot_training_loss(ln_mse, upper, os.path.join("figs", '1_2_training_loss.png'))
    