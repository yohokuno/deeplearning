import numpy as np


def forward(inputs, functions):
    units = inputs

    for function, parents in functions:
        arguments = [units[parent] for parent in parents]
        units.append(function(*arguments))

    return units


def identity(i):
    def function(x):
        return x

    def gradient(_):
        return 1

    return function, gradient, [i]


def multiply(i, j):
    def function(x, y):
        return x * y

    def gradient(x, y):
        return y, x

    return function, gradient, [i, j]


def back_propagation(inputs, functions):
    # An unit is tuple of unit value, gradient, and indices of children.
    units = [(input, None, []) for input in inputs]

    for function, gradient, parents in functions:
        arguments = []

        for parent in parents:
            arguments.append(units[parent][0])
            units[parent][2].append(len(units))

        value = function(*arguments)
        units.append((value, gradient, []))

    gradients = [0 for _ in range(len(units))]
    gradients[len(units)-1] = 1

    for j in range(len(units) - 1, 0, -1):
        value, _, children = units[j-1]

        for index in range(len(children)):
            child = children[index]
            gradients[j-1] += gradients[child] # * gradient_ij

    return gradients
