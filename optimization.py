
def gradient_descent_scalar(cost_function, derivative_function, initial_value, iterations, learning_rate):
    x = initial_value

    for i in range(iterations):
        derivative = derivative_function(x)
        x -= learning_rate * derivative
        cost = cost_function(x)
        yield x, cost
