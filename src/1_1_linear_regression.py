# Import packages
import numpy as np
import matplotlib.pyplot as plt

# Set plot size to make plots bigger
plt.rcParams["figure.figsize"] = (10,5)


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
    return np.power(np.sum(y-y_hat), 2) 


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
    y_grid = [get_predictions(basis, results[degree]['beta_hat']) for basis, degree in zip(x_basis, range(k + 1))]
    
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


def run_regression(k):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    ''' 
    # Store data
    x, y = get_data()
    
    # Get basis
    phi_x = get_basis(x, k)
    
    # Get solution
    beta_hat = get_sol(phi_x, y)
    
    # Get predictions
    y_hat = get_predictions(phi_x, beta_hat)
    
    # Get loss value
    mse = get_mse(y, y_hat)
    
    # Store results in a dictionary
    results = {'beta_hat': beta_hat, 'y_hat': y_hat, 'mse': mse}

    return(results)


def main(k):
    '''
    ------------------------
    Input: Dataset and degree
    Output: Assignment data
    ------------------------
    ''' 
    # Store output figures here
    path = '../figs/1_1.png'
    
    # Title of results plot
    title = 'Polynomial Basis Fits'

    # Get results
    results = [run_regression(degree) for degree in range(k + 1)]
    
    # Plot and save results
    plot_results(path, title, k, results)
    
    # Return the results
    return(results)


if __name__ == '__main__':

    results = main(k = 3)