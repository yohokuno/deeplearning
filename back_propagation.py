import numpy as np


def forward(inputs, functions):
    units = inputs

    for function, parents in functions:
        arguments = [units[parent] for parent in parents]
        units.append(function(*arguments))

    return units[-1]
