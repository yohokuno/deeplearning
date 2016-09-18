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


class Variable(Unit):
    def __init__(self, value=None):
        self.value = value
        super().__init__()

    def evaluate(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def __add__(self, other):
        if self.evaluate() == 0:
            return other
        return super().__add__(other)

    def __mul__(self, other):
        if self.evaluate() == 1:
            return other
        return super().__mul__(other)


class Add(Unit):
    def evaluate(self):
        return sum(parent.evaluate() for parent in self.parents)

    def get_gradient(self, index):
        # TODO: do not call evaluate() when getting gradient
        if type(self.parents[index].evaluate()) in (int, float):
            return Variable(1.0)
        else:
            return Variable(np.ones(self.parents[index].evaluate().shape))

    def __add__(self, other):
        self.add_parent(other)
        return self


class Subtract(Unit):
    def __init__(self, left, right):
        super().__init__(left, right)

    def evaluate(self):
        return self.parents[0].evaluate() - self.parents[1].evaluate()

    def get_gradient(self, index):
        if index == 0:
            sign = 1
        else:
            sign = -1
        # TODO: do not call evaluate() when getting gradient
        if type(self.parents[0].evaluate()) in (int, float):
            return Variable(sign)
        else:
            return Variable(sign * np.ones(self.parents[index].evaluate().shape))


class Multiply(Unit):
    def evaluate(self):
        result = 1.0
        for parent in self.parents:
            result *= parent.evaluate()
        return result

    def get_gradient(self, index):
        if len(self.parents) == 1:
            # TODO: do not call evaluate() when getting gradient
            if type(self.parents[0].evaluate()) in (int, float):
                return Variable(1.0)
            else:
                return Variable(np.ones(self.parents[0].shape))
        elif len(self.parents) == 2:
            return self.parents[abs(index - 1)]
        else:
            return Multiply(*(self.parents[i] for i in range(len(self.parents)) if i != index))

    def __mul__(self, other):
        self.add_parent(other)
        return self


class Power(Unit):
    def __init__(self, target, power):
        self.power = power
        super().__init__(target)

    def evaluate(self):
        return self.parents[0].evaluate() ** self.power

    def get_gradient(self, index):
        if self.power == 2:
            return Variable(2) * self.parents[0]
        return Variable(self.power) * Power(self.parents[0], self.power - 1)


class Relu(Unit):
    def evaluate(self):
        result = 0
        for parent in self.parents:
            result += max(0, parent.evaluate())
        return result

    def get_gradient(self, index):
        if self.parents[index] > 0:
            return Variable(1)
        else:
            return Variable(0)


def differentiate(target, variable):
    if variable is target:
        return Variable(1)

    gradient = Variable(0)

    for child, index in variable.get_children():
        local_gradient = child.get_gradient(index)
        child_gradient = differentiate(target, child)
        gradient += child_gradient * local_gradient

    return gradient
