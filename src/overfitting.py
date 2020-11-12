# coding: utf-8
# Import packages
import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.linear_regression import get_polynomial_basis, get_sol, get_predictions 
from src.linear_regression import get_mse, get_ln_mse, run_polynomial_regression, plot_regression_predictions


def get_true_function(x): 
    '''
    ---------------------
    Input:
    Output: 
    ---------------------
    '''
    y_true = np.power(np.sin(2*math.pi*x),2)
    return y_true


def add_noise(y_true, loc = 0, sd = 0.07):
    '''
    ---------------------
    Input:
    Output: 
    ---------------------
    '''
    y_obs = y_true + np.random.normal(loc, sd, y_true.shape)
    return(y_obs)


def get_sin_basis(x, k):
    '''
    ---------------------
    Input:
    Output: 
    ---------------------
    '''
    dims = np.arange(1, k + 1)
    grid = np.meshgrid(x, dims) 
    return(np.sin(2*math.pi*grid[0]*grid[1]).T)


def run_sin_regression(k, x, y):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    ''' 
    phi_x = get_sin_basis(x, k)
    beta_hat = get_sol(phi_x, y)
    y_hat = get_predictions(phi_x, beta_hat)
    
    mse = get_mse(y, y_hat)
    ln_mse = get_ln_mse(mse)
    
    results = {'beta_hat': beta_hat, 'y_hat': y_hat, 
               'mse': mse, 'ln_mse': ln_mse, 
               'degree': k-1, 'dim': k}
    
    return(results)


def get_data(n, min_x, max_x, sd):
    '''
    ---------------------
    Input:
    Output: 
    ---------------------
    '''
    x = np.random.uniform(min_x, max_x, n)
    
    y_true = get_true_function(x)
    y_obs = add_noise(y_true, sd = sd)
    
    return x, y_true, y_obs



def plot_data(x, y_obs, path):
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
    y_grid = np.array([get_true_function(example) for example in x_grid])

    # Title of the plot
    title = "True basis function vs. Noisy data points"

    # Plot the true curve and then plot the observed values
    plt.plot(x_grid, y_grid, label = "True function")
    plt.plot(x, y_obs, "r.")
    
    # Add annotations
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    
    # Format the axes
    axes = plt.gca()
    axes.set_xlim([0,1.1])
    axes.set_ylim([0,1.1])

    
    # Display and save plot
    plt.savefig(path)
    plt.show()
    

def plot_regression_loss(losses, highest_k, path, title, xlab = "Basis dimension", ylab = "Log MSE"):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    '''
    # Close any currently open plots
    plt.clf()

    # Make grids of x and y points to plot
    x_grid = np.arange(1, highest_k + 1)
    
    # Plot the true curve
    plt.plot(x_grid, losses, "r.")
    
    # Add annotations
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    
    # Get current access
    axes = plt.gca()
    axes.set_xlim([0,20])
    axes.set_ylim([min(losses) - 0.1, max(losses) + 0.1])

    # Show plots as a check
    plt.savefig(path)
    plt.show()
    

def get_test_mse(x_test, y_test, results, get_basis, k = 18):
    '''
    ---------------------
    Input: Parameters needed for data
    Output: output
    ---------------------
    '''
    # Get features
    x_basis = [get_basis(x_test, dim) for dim in range(1, k + 1)]
    y_preds = [get_predictions(basis, results[deg]['beta_hat']) for basis, deg in zip(x_basis, range(k + 1))]
    mse_test = [get_mse(y_test, y_pred) for y_pred in y_preds]
    ln_mse_test = [get_ln_mse(mse) for mse in mse_test]

    return(mse_test, ln_mse_test)


def execute_data_plots(x, y, path): 
    '''
    ---------------------
    Input: Parameters needed for data
    Output: output
    ---------------------
    '''
    plot_data(x, y, path)


def execute_poly_plots(x, y, path, run_regression, get_basis, title, dims = [2, 5, 10, 12, 14, 18]):
    '''
    ---------------------
    Input: Parameters needed for data
    Output: output
    ---------------------
    '''
    for k in dims:
        fig_title = title.format(k)
        fig_path = path.format(k)
        results = [run_regression(k, x, y)]
        plot_regression_predictions(fig_path, fig_title, k, k+1, results, x, y, get_basis)


def execute_train_loss_plots(x, y, start_dim, end_dim, path, run_regression, title):
    '''
    ---------------------
    Input: Parameters needed for data
    Output: output
    ---------------------
    '''
    dims = range(start_dim, end_dim + 1)
    results = np.array([run_regression(k, x, y) for k in dims])
    ln_mse = np.array([result['ln_mse'] for result in results])
    plot_regression_loss(ln_mse, end_dim, path, title)

    return(results)


def execute_test_loss_plots(x_test, y_test, results, end_dim, path, get_basis, title):
    '''
    ---------------------
    Input: Parameters needed for data
    Output: output
    ---------------------
    '''
    mse_test, ln_mse_test = get_test_mse(x_test, y_test, results, get_basis)
    plot_regression_loss(ln_mse_test, end_dim, path, title)
    return(ln_mse_test)


def main(basis = 'polynomial',
         n_runs = 1,
         path_data_plot =  os.path.join(".", "figs", '1_2_data.png'), 
         path_poly_plot = os.path.join(".", "figs", '1_2_results_dim_{}_{}.png'), 
         path_train_loss = os.path.join(".", "figs", "1_2_train_loss_{}.png"), 
         path_test_loss = os.path.join(".", "figs", '1_2_test_loss_{}.png'),
         path_train_loss_multiple = os.path.join(".", "figs", '1_2_train_loss_100_runs_{}.png'),
         path_test_loss_multiple = os.path.join(".", "figs", '1_2_test_loss_100_runs_{}.png')):
    '''
    ---------------------
    Input: Parameters needed for data
    Output: output
    ---------------------
    '''
    
    # Set the random seed to reproduce
    np.random.seed(1291239)

    # Set parameters
    min_x = 0
    max_x = 1 
    sigma = 0.07 
    n_train_samples = 30
    n_test_samples = 1000

    # Store dimensions of regression basis functions
    # Note that these are dimensions and not degrees
    # Degree of polynomial = Dimension of polynomial - 1
    start_dim = 1
    end_dim = 18

    # Insert basis name into file paths
    path_poly_plot = path_poly_plot.format({}, basis) 
    path_train_loss = path_train_loss.format(basis)
    path_test_loss = path_test_loss.format(basis)
    path_train_loss_multiple = path_train_loss_multiple.format(basis)
    path_test_loss_multiple = path_test_loss_multiple.format(basis)

    # Set the basis generator and regression function depending on the option passed
    if basis == 'polynomial':
        get_basis = get_polynomial_basis
        run_regression = run_polynomial_regression
    
    elif basis == 'sin':
        get_basis = get_sin_basis
        run_regression = run_sin_regression

    # This is for the initial single run
    if n_runs == 1:
        
        # Get dataset
        x, y_true, y_obs = get_data(n_train_samples, min_x, max_x, sigma)
        x_test, y_test_true, y_test_obs = get_data(n_test_samples, min_x, max_x, sigma)

        # Make plots
        execute_data_plots(x, y_obs, path_data_plot)
        execute_poly_plots(x, y_obs, path_poly_plot, run_regression, get_basis, 
                           title = "{} basis regression results: k = {}".format(basis.title(), {}))

        # Store results
        results = execute_train_loss_plots(x, y_obs, start_dim, end_dim, path_train_loss, run_regression, 
                                           title = "{} Basis Train Error".format(basis.title()))
        
        ln_mse_test = execute_test_loss_plots(x_test, y_test_obs, results, end_dim, path_test_loss, get_basis, 
                                              title = "{} Basis Test Error".format(basis.title()))

    # This is for multiple runs to return average MSE
    elif n_runs > 1:
        
        # Create list to store results
        results = []
        
        # Call the process for the no. of runs input by the user
        for counter in range(n_runs):
            
            # Print progress
            print("Doing the {}th run...".format(counter))
            
            # Make new datasets
            x, y_true, y_obs = get_data(n_train_samples, min_x, max_x, sigma)
            x_test, y_test_true, y_test_obs = get_data(n_test_samples, min_x, max_x, sigma)

            # Store dimensions to iterate over and then run polynomial regressions
            dims = range(start_dim, end_dim + 1)
            results_from_single_run = [run_regression(k, x, y_obs) for k in dims]

            # Get results on test set
            # First create the testing features
            mse_test, ln_mse_test = get_test_mse(x_test, y_test_obs, results_from_single_run, get_basis)

            # Append this to results
            for result, mse, ln_mse in zip(results_from_single_run, mse_test, ln_mse_test):
                result['mse_test'] = mse
                result['ln_mse_test'] = ln_mse
                
            # Store results
            results.append(results_from_single_run)

        
        # Store the average
        mse = [np.mean(np.array([results[run][degree]['mse'] for run in range(n_runs)])) 
               for degree in range(end_dim)]
        
        mse_test = [np.mean(np.array([results[run][degree]['mse_test'] for run in range(n_runs)])) 
                    for degree in range(end_dim)]

        # Take logs
        ln_mse = np.log(mse)
        ln_mse_test = np.log(mse_test)

        # Now make plots 
        plot_regression_loss(ln_mse, end_dim, path_train_loss_multiple, title = "{} Basis Train Error".format(basis.title()))
        plot_regression_loss(ln_mse_test, end_dim, path_test_loss_multiple, title = "{} Basis Test Error".format(basis.title()))


    # Return results
    return(results)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='List the content of a folder')
    
    parser.add_argument('basis',
                        type=str, 
                        help='Whether to use polynomial or sin basis...')

    parser.add_argument('n_runs',
                         type=int, 
                         help='No. of runs..')
    
    args = parser.parse_args()
    
    basis = args.basis
    n_runs = args.n_runs
    results = main(basis=basis, n_runs=n_runs)