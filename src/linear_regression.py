# Import packages
import os
import numpy as np
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
    Output: Assignment data
    ------------------------
    '''
    grid = np.meshgrid(x, np.arange(k + 1)) 
    return np.power(grid[0], grid[1]).T


def get_sol(phi_x, y):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    '''
    return np.linalg.inv(phi_x.T @ phi_x) @ phi_x.T @ y


def get_predictions(phi_x, beta_hat):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    '''
    return phi_x @ beta_hat


def get_mse(y, y_hat):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    '''
    return np.sum(np.power(y-y_hat, 2))



def plot_results(path, title, k, results):
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
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    
    axes = plt.gca()
    axes.set_xlim([0,5])
    axes.set_ylim([-5,8])

    
    # Display and save plot
    plt.show()
    plt.savefig(path)


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
    
    #Get loss and results 
    mse = loss_func(y, y_hat)
    results = {'beta_hat': beta_hat, 'y_hat': y_hat, 'mse': mse}
    
    return(results)


def main(k):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    ''' 
    title = 'Polynomial Basis Fits'
    path = os.path.join('.', 'figs', '1_1.png')
    
    x, y = get_data()
    results = [run_regression(degree, x, y, get_mse) for degree in range(k + 1)]
    plot_results(path, title, k, results)
    
    return(results)



if __name__ == '__main__':

    results = main(k = 3)