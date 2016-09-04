import itertools
import math
import numpy as np


def run_iterations(iterator, max_iterations, abs_tol=1e-20):
    """ Run iterative optimization method such as gradient descent. Stop early if cost doesn't change. """
    previous_cost = None
    limited_iterator = itertools.islice(iterator, max_iterations)

    for i, (x, cost) in enumerate(limited_iterator):
        if previous_cost is not None and math.isclose(cost, previous_cost, abs_tol=abs_tol):
            break
        previous_cost = cost

    return i, x, cost


def gradient_descent(cost, gradient, initial_value, step_size):
    x = initial_value

    while True:
        x -= step_size * gradient(x)
        yield x, cost(x)


def newtons_method(cost, gradient, hessian, initial_value, step_size):
    x = initial_value

    while True:
        if np.isscalar(x):
            x -= step_size * gradient(x) / hessian(x)
        else:
            H = hessian(x)
            x -= step_size * np.linalg.inv(H) @ gradient(x)
            # More efficient implementation
            # x = np.linalg.solve(H, H @ x - step_size * gradient(x))
        yield x, cost(x)


def linear_least_square(A, b):
    def cost(x):
        return np.linalg.norm(A @ x - b) ** 2 / 2.0

    def gradient(x):
        return A.T @ (A @ x - b)

    i, x, cost = run_iterations(gradient_descent(cost, gradient, np.zeros(A.shape[1]), 1.0), 10000)
    return x
