import numpy as np


class Unit:
    def __init__(self, *parents):
        self.parents = list(parents)
        for i, parent in enumerate(parents):
            parent.add_child(self, i)

        self.children = []

    def add_parent(self, parent):
        self.parents.append(parent)
        parent.add_child(self, len(self.parents) - 1)

    def get_parents(self):
        return self.parents

    def add_child(self, child, index):
        # A child is a pair of child unit and index of self unit within parents of child unit
        self.children.append((child, index))

    def get_children(self):
        return self.children

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def __sub__(self, other):
        return Subtract(self, other)

    def __pow__(self, power, modulo=None):
        return Power(self, power)

    def __matmul__(self, other):
        return MatrixMultiply(self, other)


class Variable(Unit):
    def __init__(self, value=None):
        self.value = value
        super().__init__()

    def evaluate(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def __add__(self, other):
        return super().__add__(other)

    def __mul__(self, other):
        return super().__mul__(other)

ZERO = Variable(0)
ONE = Variable(1)


class Add(Unit):
    def evaluate(self):
        return sum(parent.evaluate() for parent in self.parents)

    def get_gradient(self, index):
        return ONE


class Subtract(Unit):
    def evaluate(self):
        return self.parents[0].evaluate() - sum(parent.evaluate() for parent in self.parents[1:])

    def get_gradient(self, index):
        if index == 0:
            return ONE
        else:
            return Variable(-1.0)


class Multiply(Unit):
    def evaluate(self):
        result = 1.0
        for parent in self.parents:
            result *= parent.evaluate()
        return result

    def get_gradient(self, index):
        if len(self.parents) == 2:
            return self.parents[abs(index - 1)]
        else:
            return Multiply(*(self.parents[i] for i in range(len(self.parents)) if i != index))


class Power(Unit):
    def __init__(self, target, power):
        # TODO: support case when power is another unit
        self.power = power
        super().__init__(target)

    def evaluate(self):
        return self.parents[0].evaluate() ** self.power

    def get_gradient(self, index):
        if self.power == 2:
            return Variable(2) * self.parents[0]
        return Variable(self.power) * Power(self.parents[0], self.power - 1)


class Sum(Unit):
    def evaluate(self):
        return sum(np.sum(parent.evaluate()) for parent in self.parents)

    def get_gradient(self, index):
        return ONE


class MatrixMultiply(Unit):
    def evaluate(self):
        if type(self.parents[0].evaluate()) in (int, float) or type(self.parents[1].evaluate()) in (int, float):
            return self.parents[0].evaluate() * self.parents[1].evaluate()
        return self.parents[0].evaluate() @ self.parents[1].evaluate()

    def get_gradient(self, index):
        if index == 0:
            return self.parents[1]
        else:
            return Transpose(self.parents[0])


class Transpose(Unit):
    def evaluate(self):
        result = self.parents[0].evaluate()
        if type(result) in (int, float):
            return result
        else:
            return result.T

    def get_gradient(self, index):
        return ONE


class Repeat(Unit):
    def __init__(self, parent, repeat):
        super().__init__()
        self.repeat = repeat
        for i in range(repeat):
            self.add_parent(parent)

    def evaluate(self):
        return np.array([parent.evaluate() for parent in self.parents])

    def get_gradient(self, index):
        result = np.zeros(len(self.parents))
        result[index] = 1
        return Variable(result)


class Relu(Unit):
    def evaluate(self):
        A = np.expand_dims(self.parents[0].evaluate(), 0)
        return np.max(np.concatenate([A, np.zeros(A.shape)]), 0)

    def get_gradient(self, index):
        return ReluGradient(self.parents[index])


class ReluGradient(Unit):
    def evaluate(self):
        parent_value = self.parents[0].evaluate()
        result = np.sign(parent_value) / 2.0 + 0.5
        if type(parent_value) in (int, float):
            return result
        if len(parent_value.shape) == 1:
            return np.diag(result)
        elif len(parent_value.shape) == 2:
            return np.diagflat(parent_value).reshape(parent_value.shape * 2)

    def get_gradient(self, index):
        return ZERO


def differentiate(target, variable):
    if variable is target:
        return ONE

    gradient = ZERO

    for child, index in variable.get_children():
        local_gradient = child.get_gradient(index)
        child_gradient = differentiate(target, child)

        if child_gradient is ONE:
            current_gradient = local_gradient
        elif local_gradient is ONE:
            current_gradient = child_gradient
        elif local_gradient is ZERO or child_gradient is ZERO:
            current_gradient = ZERO
        else:
            current_gradient = local_gradient @ child_gradient

        if current_gradient is not ZERO:
            if gradient is ZERO:
                gradient = current_gradient
            else:
                gradient += current_gradient

    return gradient
