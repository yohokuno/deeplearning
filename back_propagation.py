def identity(x):
    return lambda x: x, lambda _: [1], x


def multiply(x, y):
    return lambda x, y: x * y, lambda x, y: [y, x], [x, y]


def forward(inputs, functions):
    units = inputs

    for function, _, parents in functions:
        arguments = [units[parent] for parent in parents]
        units.append(function(*arguments))

    return units


def back_propagation(inputs, functions):
    units = forward(inputs, functions)
    gradients = [1]
    for i in range(len(units), 0, -1):
        unit = units[i-1]
        _, gradient, parents = functions[i-len(inputs)]
        arguments = [units[parent] for parent in parents]
        gradients.insert(0, sum(gradients[parent] * gradient(arguments)[parent] for parent in parents))
    return gradients
