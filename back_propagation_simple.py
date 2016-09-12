def identity(i):
    def function(x):
        return x

    def gradient(_):
        return [1]

    return function, gradient, [i]


def multiply(i, j):
    def function(x, y):
        return x * y

    def gradient(x, y):
        return y, x

    return function, gradient, [i, j]


def add(i, j):
    def function(x, y):
        return x + y

    def gradient(x, y):
        return 1, 1

    return function, gradient, [i, j]


def back_propagation_simple(inputs, functions):
    # An unit is tuple of unit value, local gradient and indices of children.
    units = [[input, [], []] for input in inputs]

    for function, gradient, parents in functions:
        arguments = []

        for parent in parents:
            arguments.append(units[parent][0])
            units[parent][2].append(len(units))

        units.append([function(*arguments), [], []])

        for parent, local_gradient in zip(parents, gradient(*arguments)):
            units[parent][1].append(local_gradient)

    gradients = [0 for _ in range(len(units))]
    gradients[len(units)-1] = 1

    for j in range(len(units) - 1, 0, -1):
        value, local_gradient, children = units[j-1]

        for index in range(len(children)):
            child = children[index]
            gradients[j-1] += gradients[child] * local_gradient[index]

    return gradients
