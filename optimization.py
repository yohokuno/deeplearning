import itertools
import math


def run_iterations(iterator, max_iterations, abs_tol=1e-20):
    """ Run iterative optimization method such as gradient descent. Stop early if cost doesn't change. """
    previous_cost = None
    limited_iterator = itertools.islice(iterator, max_iterations)

    for i, (x, cost) in enumerate(limited_iterator):
        if previous_cost is not None and math.isclose(cost, previous_cost, abs_tol=abs_tol):
            break
        previous_cost = cost

    return i, x, cost


def gradient_descent_scalar(cost, derivative, initial_value, step_size):
    x = initial_value

    while True:
        x -= step_size * derivative(x)
        yield x, cost(x)


def newtons_method_scalar(cost, derivative, second_derivative, initial_value, step_size):
    x = initial_value

    while True:
        x -= step_size * derivative(x) / second_derivative(x)
        yield x, cost(x)
