""" gp.py
Bayesian optimisation of loss functions.
https://github.com/thuijskens/bayesian-optimization
"""

import numpy as np
import sklearn.gaussian_process as gp
from pathos.multiprocessing import Pool

from scipy.stats import norm
from scipy.optimize import minimize
import datetime
import os
import time
import argparse

from fitness_functions.fitness_function import RastriginFunction, FullyConnected, ConvNet


def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        exp_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        exp_improvement[sigma == 0.0] = 0.0

    return -1 * exp_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(patience=-1, n_iters=-1, sample_loss=None, bounds=None, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7, pool_size=1):
    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.
    Arguments:
    ----------
        patience: integer
            Number of nonimproved iterations before exit.
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    iteration = 0
    no_improvement = 0
    best_fitness = 100

    n_params = bounds.shape[0]
    print("Create {} initial points".format(n_pre_samples))

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)

        if pool_size > 1:
            p = Pool(pool_size)
            losses = p.map(sample_loss, x_list)
            y_list = losses
            p.terminate()
        else:
            for params in x_list:
                y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)

        if pool_size > 1:
            p = Pool(pool_size)
            losses = p.map(sample_loss, x_list)
            y_list = losses
            p.terminate()
        else:
            for params in x_list:
                y_list.append(sample_loss(params))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    print("Use bayesian optimization to sample points.")
    while True:
        print("Iteration: {}".format(iteration + 1))
        model.fit(xp, yp)

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=False, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=False, bounds=bounds, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

        if cv_score < best_fitness:
            best_fitness = cv_score
            no_improvement = 0
        else:
            no_improvement += 1

        iteration += 1

        if iteration == n_iters or patience == no_improvement:
            break

    return xp, yp


if __name__ == "__main__":
    """python -m utils.bayesian_optimization --ff fc --patience 30 --n_pre 16"""

    parser = argparse.ArgumentParser(description="Compile and run a bayesian optimization.")

    parser.add_argument('--ff', help='Set the fitness function. rf: Rastrigin, fc: FullyConnected, cn: ConvNet', choices=['rf', 'fc', 'cn'], default='rf', type=str)
    parser.add_argument('--patience', help='Patience of the searching.', default=10, type=int)
    parser.add_argument('--n_pre', help='Number of initial points.', default=12, type=int)
    parser.add_argument('--random', help='Random search.', default=0, type=int)
    parser.add_argument('--pool_size', help='Pool size for multiprocessing.', default=1, type=int)
    args = parser.parse_args()
    parameter_dict = {"fitness_function": args.ff,
                      "patience": args.patience,
                      "n_pre": args.patience,
                      "random": args.random,
                      "pool_size": args.pool_size}

    fitness_func = None
    phenotypes_func = None
    if parameter_dict["fitness_function"] == "rf":
        fitness_func = RastriginFunction()
        exit()
    elif parameter_dict["fitness_function"] == "fc":
        fitness_func = FullyConnected()
        phenotypes_func = FullyConnected.log_result
    elif parameter_dict["fitness_function"] == "cn":
        fitness_func = ConvNet()
        phenotypes_func = ConvNet.log_result

    bounds = np.array([[0, 1]] * 11)
    sample_loss = fitness_func.calculate

    start = time.time()
    genes, fitness_values = bayesian_optimisation(patience=parameter_dict.get("patience"),
                                                  n_iters=-1, sample_loss=sample_loss,
                                                  bounds=bounds,
                                                  n_pre_samples=parameter_dict.get("n_pre"),
                                                  random_search=parameter_dict.get("random"),
                                                  pool_size=parameter_dict.get("pool_size"))
    running_time = time.time() - start

    # [print("{}\n{}\n".format(param, fv)) for param, fv in zip(genes, fitness_values)]

    num_of_eval = len(fitness_values)
    min_index = np.argmin(fitness_values)

    best_dict = fitness_func.genotype_to_phenotype(genes[min_index])
    phenotypes_dict = phenotypes_func(best_dict, fitness_values[min_index])

    result_dict = {"name": parameter_dict["fitness_function"],
                   "running time": running_time,
                   "number of evaluation": num_of_eval,
                   "best fitness value": fitness_values[min_index],
                   "best genes": genes[min_index],
                   "patience": args.patience,
                   "n_pre": args.patience,
                   "random": args.random
                   }
    result_dict = {**result_dict, **phenotypes_dict}

    path = "./logs/bayesian-{}-{}".format(parameter_dict["fitness_function"], datetime.datetime.now().strftime('%y-%m-%d-%H:%M'))

    if not os.path.exists(path):
        os.mkdir(path)

    file_path = path + "/result.txt"

    with open(file_path, 'w+') as result_file:
        result_file.write(str(result_dict))
